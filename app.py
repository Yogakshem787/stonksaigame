"""
PortfolioPulse — Model Portfolio Platform Backend
===================================================
Features:
- Google OAuth + Email/Password auth (JWT)
- Model portfolios (1 per user, ₹100 virtual cash)
- Buy/sell stocks at live NSE prices
- Public portfolio publishing (with category + bio)
- Competitions (create, invite, leaderboard, auto-winner)
- Comments on public portfolios
- Daily price refresh via yfinance
- PostgreSQL (Render-compatible)
"""

import os, time, math, logging, hashlib, secrets, json, re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, date
from functools import wraps

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import jwt as pyjwt
import requests
import yfinance as yf

# ═══════ APP SETUP ═══════
app = Flask(__name__)

ALLOWED_ORIGINS = [
    "https://portfoliopulse.in",
    "https://www.portfoliopulse.in",
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5501",
    "null",  # file:// local testing
]
_extra = os.environ.get("EXTRA_ORIGINS", "")
if _extra:
    ALLOWED_ORIGINS += [o.strip() for o in _extra.split(",") if o.strip()]

CORS(app,
     resources={r"/api/*": {"origins": ALLOWED_ORIGINS}},
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True,
     max_age=3600)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", secrets.token_hex(32))
db_url = os.environ.get("DATABASE_URL", "sqlite:///portfoliopulse.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
elif db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 280,
    "pool_pre_ping": True,
    "pool_size": 5,
    "max_overflow": 10,
}

db = SQLAlchemy(app)

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5500")
ADMIN_EMAILS = [e.strip() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pp")

# ═══════ CORS HELPERS ═══════
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")
    if origin in ALLOWED_ORIGINS or not origin:
        response.headers["Access-Control-Allow-Origin"] = origin or "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Vary"] = "Origin"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin-allow-popups"
    response.headers["Cross-Origin-Embedder-Policy"] = "unsafe-none"
    return response

@app.route("/api/<path:path>", methods=["OPTIONS"])
def handle_options(path):
    r = app.make_default_options_response()
    origin = request.headers.get("Origin", "")
    r.headers["Access-Control-Allow-Origin"] = origin or "*"
    r.headers["Access-Control-Allow-Credentials"] = "true"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    r.headers["Access-Control-Max-Age"] = "3600"
    return r

# ═══════ DATABASE MODELS ═══════

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name = db.Column(db.String(255), default="")
    password_hash = db.Column(db.String(255), default="")
    google_id = db.Column(db.String(255), default="")
    avatar_url = db.Column(db.String(500), default="")
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)
    login_count = db.Column(db.Integer, default=0)
    # Profile for portfolio managers
    bio = db.Column(db.Text, default="")
    linkedin_url = db.Column(db.String(500), default="")

    portfolio = db.relationship("ModelPortfolio", backref="owner", uselist=False, cascade="all, delete-orphan")
    competitions_created = db.relationship("Competition", backref="creator", foreign_keys="Competition.creator_id", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "avatar": self.avatar_url,
            "isAdmin": self.is_admin,
            "bio": self.bio,
            "linkedin": self.linkedin_url,
            "createdAt": self.created_at.isoformat(),
            "hasGoogle": bool(self.google_id),
            "hasPassword": bool(self.password_hash),
        }


class ModelPortfolio(db.Model):
    """Each user can have 1 model portfolio with ₹100 virtual cash."""
    __tablename__ = "model_portfolios"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True, nullable=False)
    name = db.Column(db.String(255), default="My Portfolio")
    description = db.Column(db.Text, default="")
    category = db.Column(db.String(50), default="flexicap")  # smallcap, midcap, largecap, flexicap
    is_public = db.Column(db.Boolean, default=False)
    cash = db.Column(db.Float, default=100.0)  # virtual ₹100 starting cash
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Cached performance
    total_value = db.Column(db.Float, default=100.0)
    total_return_pct = db.Column(db.Float, default=0.0)
    last_refreshed = db.Column(db.DateTime, nullable=True)

    holdings = db.relationship("PortfolioHolding", backref="portfolio", lazy=True, cascade="all, delete-orphan")
    comments = db.relationship("PortfolioComment", backref="portfolio", lazy=True, cascade="all, delete-orphan")

    def to_dict(self, include_holdings=False):
        owner = self.owner
        d = {
            "id": self.id,
            "userId": self.user_id,
            "ownerName": owner.name if owner else "",
            "ownerAvatar": owner.avatar_url if owner else "",
            "ownerBio": owner.bio if owner else "",
            "ownerLinkedin": owner.linkedin_url if owner else "",
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "isPublic": self.is_public,
            "cash": round(self.cash, 4),
            "totalValue": round(self.total_value, 4),
            "totalReturnPct": round(self.total_return_pct, 2),
            "createdAt": self.created_at.isoformat(),
            "lastRefreshed": self.last_refreshed.isoformat() if self.last_refreshed else None,
            "holdingsCount": len(self.holdings),
            "commentsCount": len(self.comments),
        }
        if include_holdings:
            d["holdings"] = [h.to_dict() for h in self.holdings]
        return d


class PortfolioHolding(db.Model):
    """A stock holding in a model portfolio."""
    __tablename__ = "portfolio_holdings"
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey("model_portfolios.id"), nullable=False)
    symbol = db.Column(db.String(30), nullable=False)
    name = db.Column(db.String(255), default="")
    weight = db.Column(db.Float, nullable=False)  # % weight e.g. 25.0
    duration = db.Column(db.String(20), default="12m")  # e.g. "1m","3m","6m","12m",... "36+m"
    purchase_price = db.Column(db.Float, nullable=False)  # locked at buy time
    current_price = db.Column(db.Float, default=0)
    purchase_date = db.Column(db.Date, default=date.today)
    is_sold = db.Column(db.Boolean, default=False)
    sold_price = db.Column(db.Float, nullable=True)
    sold_date = db.Column(db.Date, nullable=True)

    def pnl(self):
        """P&L in virtual ₹ terms: (CMP - Purchase) * 100 * weight%"""
        if self.is_sold:
            return round((self.sold_price - self.purchase_price) / self.purchase_price * 100 * (self.weight / 100), 4)
        if not self.current_price:
            return 0
        return round((self.current_price - self.purchase_price) / self.purchase_price * 100 * (self.weight / 100), 4)

    def pnl_pct(self):
        ref = self.sold_price if self.is_sold else self.current_price
        if not ref or not self.purchase_price:
            return 0
        return round((ref - self.purchase_price) / self.purchase_price * 100, 2)

    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "name": self.name,
            "weight": self.weight,
            "duration": self.duration,
            "purchasePrice": self.purchase_price,
            "currentPrice": self.current_price,
            "purchaseDate": self.purchase_date.isoformat(),
            "isSold": self.is_sold,
            "soldPrice": self.sold_price,
            "soldDate": self.sold_date.isoformat() if self.sold_date else None,
            "pnl": self.pnl(),
            "pnlPct": self.pnl_pct(),
        }


class PortfolioComment(db.Model):
    __tablename__ = "portfolio_comments"
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey("model_portfolios.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    author = db.relationship("User", foreign_keys=[user_id])

    def to_dict(self):
        return {
            "id": self.id,
            "portfolioId": self.portfolio_id,
            "userId": self.user_id,
            "authorName": self.author.name if self.author else "Anonymous",
            "authorAvatar": self.author.avatar_url if self.author else "",
            "text": self.text,
            "createdAt": self.created_at.isoformat(),
        }


class Competition(db.Model):
    __tablename__ = "competitions"
    id = db.Column(db.Integer, primary_key=True)
    creator_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, default="")
    invite_code = db.Column(db.String(20), unique=True, nullable=False, index=True)
    max_participants = db.Column(db.Integer, default=20)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    winner_user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    participants = db.relationship("CompetitionParticipant", backref="competition", lazy=True, cascade="all, delete-orphan")
    winner = db.relationship("User", foreign_keys=[winner_user_id])

    def to_dict(self, include_participants=False):
        d = {
            "id": self.id,
            "creatorId": self.creator_id,
            "creatorName": self.creator.name if self.creator else "",
            "name": self.name,
            "description": self.description,
            "inviteCode": self.invite_code,
            "maxParticipants": self.max_participants,
            "startDate": self.start_date.isoformat(),
            "endDate": self.end_date.isoformat(),
            "isActive": self.is_active,
            "participantCount": len(self.participants),
            "winnerId": self.winner_user_id,
            "winnerName": self.winner.name if self.winner else None,
            "createdAt": self.created_at.isoformat(),
            "daysLeft": max(0, (self.end_date - date.today()).days),
            "hasStarted": date.today() >= self.start_date,
            "hasEnded": date.today() > self.end_date,
        }
        if include_participants:
            d["participants"] = [p.to_dict() for p in self.participants]
        return d


class CompetitionParticipant(db.Model):
    __tablename__ = "competition_participants"
    id = db.Column(db.Integer, primary_key=True)
    competition_id = db.Column(db.Integer, db.ForeignKey("competitions.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    joined_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Snapshot of return at join time (for leaderboard delta)
    baseline_value = db.Column(db.Float, default=100.0)

    user = db.relationship("User")

    def to_dict(self):
        portfolio = ModelPortfolio.query.filter_by(user_id=self.user_id).first()
        return {
            "userId": self.user_id,
            "userName": self.user.name if self.user else "",
            "userAvatar": self.user.avatar_url if self.user else "",
            "joinedAt": self.joined_at.isoformat(),
            "currentValue": portfolio.total_value if portfolio else 100.0,
            "returnPct": portfolio.total_return_pct if portfolio else 0.0,
            "portfolioId": portfolio.id if portfolio else None,
        }

# ═══════ AUTH HELPERS ═══════

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def make_token(user_id):
    payload = {"uid": user_id, "exp": datetime.utcnow() + timedelta(days=30)}
    return pyjwt.encode(payload, app.config["SECRET_KEY"], algorithm="HS256")

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Authentication required"}), 401
        try:
            data = pyjwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            g.user = User.query.get(data["uid"])
            if not g.user:
                return jsonify({"error": "User not found"}), 401
        except pyjwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except Exception:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return wrapper

def optional_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        g.user = None
        if token:
            try:
                data = pyjwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
                g.user = User.query.get(data["uid"])
            except Exception:
                pass
        return f(*args, **kwargs)
    return wrapper

# ═══════ STOCK PRICE HELPERS ═══════

_price_cache = {}  # symbol -> (price, timestamp)
CACHE_TTL = 3600  # 1 hour

def get_stock_price(symbol):
    """Fetch NSE stock price via yfinance. Returns price or None."""
    now = time.time()
    if symbol in _price_cache:
        price, ts = _price_cache[symbol]
        if now - ts < CACHE_TTL:
            return price

    ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
        if price and price > 0:
            _price_cache[symbol] = (price, now)
            return price
        # fallback: history
        hist = t.history(period="2d")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
            _price_cache[symbol] = (price, now)
            return price
    except Exception as e:
        log.warning(f"[PRICE] {symbol}: {e}")
    return None

def get_stock_name(symbol):
    """Try to get company name from yfinance."""
    ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return info.get("longName") or info.get("shortName") or symbol
    except Exception:
        return symbol

def refresh_portfolio_prices(portfolio):
    """Update current prices for all active holdings and recalculate portfolio value."""
    active = [h for h in portfolio.holdings if not h.is_sold]
    total_invested = 100.0  # starting capital
    total_weight_deployed = sum(h.weight for h in active)
    cash = portfolio.cash

    current_portfolio_value = cash  # start with cash
    for h in active:
        price = get_stock_price(h.symbol)
        if price:
            h.current_price = price
        # value of this holding = weight% * 100 * (CMP/purchase)
        if h.purchase_price > 0:
            current_portfolio_value += (h.weight / 100) * 100 * (h.current_price / h.purchase_price)

    portfolio.total_value = round(current_portfolio_value, 4)
    portfolio.total_return_pct = round((current_portfolio_value - 100) / 100 * 100, 2)
    portfolio.last_refreshed = datetime.utcnow()
    db.session.commit()

# ═══════ AUTH ROUTES ═══════

@app.route("/api/auth/google", methods=["POST"])
def google_auth():
    data = request.json or {}
    credential = data.get("credential") or data.get("token")
    if not credential:
        return jsonify({"error": "No credential"}), 400
    try:
        r = requests.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={credential}", timeout=10)
        info = r.json()
        if info.get("aud") != GOOGLE_CLIENT_ID and GOOGLE_CLIENT_ID:
            return jsonify({"error": "Invalid token audience"}), 401
        email = info.get("email")
        if not email:
            return jsonify({"error": "No email in token"}), 400
    except Exception as e:
        return jsonify({"error": f"Google auth failed: {e}"}), 400

    user = User.query.filter_by(email=email).first()
    is_new = False
    if not user:
        user = User(
            email=email,
            name=info.get("name", ""),
            google_id=info.get("sub", ""),
            avatar_url=info.get("picture", ""),
            is_admin=email in ADMIN_EMAILS,
        )
        db.session.add(user)
        is_new = True
    else:
        if not user.google_id:
            user.google_id = info.get("sub", "")
        if not user.avatar_url:
            user.avatar_url = info.get("picture", "")
        if not user.name:
            user.name = info.get("name", "")

    user.last_login = datetime.utcnow()
    user.login_count = (user.login_count or 0) + 1
    db.session.commit()

    return jsonify({"token": make_token(user.id), "user": user.to_dict(), "isNew": is_new})


@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.json or {}
    email = (data.get("email") or "").strip().lower()
    name = (data.get("name") or "").strip()
    pw = data.get("password") or ""
    if not email or not pw:
        return jsonify({"error": "Email and password required"}), 400
    if len(pw) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 400
    user = User(
        email=email,
        name=name or email.split("@")[0],
        password_hash=hash_password(pw),
        is_admin=email in ADMIN_EMAILS,
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({"token": make_token(user.id), "user": user.to_dict(), "isNew": True})


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    email = (data.get("email") or "").strip().lower()
    pw = data.get("password") or ""
    user = User.query.filter_by(email=email).first()
    if not user or not user.password_hash or user.password_hash != hash_password(pw):
        return jsonify({"error": "Invalid email or password"}), 401
    user.last_login = datetime.utcnow()
    user.login_count = (user.login_count or 0) + 1
    db.session.commit()
    return jsonify({"token": make_token(user.id), "user": user.to_dict()})


@app.route("/api/auth/me", methods=["GET"])
@require_auth
def get_me():
    return jsonify({"user": g.user.to_dict()})


@app.route("/api/auth/profile", methods=["PUT"])
@require_auth
def update_profile():
    data = request.json or {}
    if "name" in data:
        g.user.name = data["name"].strip()
    if "bio" in data:
        g.user.bio = data["bio"].strip()
    if "linkedin_url" in data:
        g.user.linkedin_url = data["linkedin_url"].strip()
    db.session.commit()
    return jsonify({"user": g.user.to_dict()})

# ═══════ STOCK SEARCH ═══════

@app.route("/api/stocks/search", methods=["GET"])
def search_stocks():
    q = (request.args.get("q") or "").strip().upper()
    if len(q) < 1:
        return jsonify({"results": []})
    # Use yfinance search
    try:
        results = []
        # Try direct NSE ticker first
        ticker_ns = q + ".NS"
        t = yf.Ticker(ticker_ns)
        info = t.fast_info
        price = getattr(info, "last_price", None)
        if price and price > 0:
            full_info = t.info
            results.append({
                "symbol": q,
                "name": full_info.get("longName") or full_info.get("shortName") or q,
                "price": round(price, 2),
                "exchange": "NSE",
            })
        if not results:
            # Try yfinance search
            search_r = requests.get(
                f"https://query2.finance.yahoo.com/v1/finance/search?q={q}&lang=en-US&region=IN&quotesCount=8&newsCount=0&listsCount=0",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=5
            )
            quotes = search_r.json().get("quotes", [])
            for qt in quotes:
                if qt.get("exchange") in ("NSI", "BSE") or qt.get("symbol", "").endswith(".NS") or qt.get("symbol", "").endswith(".BO"):
                    sym = qt.get("symbol", "").replace(".NS", "").replace(".BO", "")
                    results.append({
                        "symbol": sym,
                        "name": qt.get("longname") or qt.get("shortname") or sym,
                        "price": 0,
                        "exchange": "NSE",
                    })
        return jsonify({"results": results[:8]})
    except Exception as e:
        log.warning(f"[SEARCH] {q}: {e}")
        return jsonify({"results": []})


@app.route("/api/stocks/price/<symbol>", methods=["GET"])
def get_price(symbol):
    symbol = symbol.upper()
    price = get_stock_price(symbol)
    if price:
        name = get_stock_name(symbol)
        return jsonify({"symbol": symbol, "price": round(price, 2), "name": name})
    return jsonify({"error": "Could not fetch price"}), 404

# ═══════ PORTFOLIO ROUTES ═══════

@app.route("/api/portfolio", methods=["GET"])
@require_auth
def get_my_portfolio():
    pf = ModelPortfolio.query.filter_by(user_id=g.user.id).first()
    if not pf:
        return jsonify({"portfolio": None})
    # Refresh if not refreshed today
    if not pf.last_refreshed or pf.last_refreshed.date() < date.today():
        refresh_portfolio_prices(pf)
    return jsonify({"portfolio": pf.to_dict(include_holdings=True)})


@app.route("/api/portfolio", methods=["POST"])
@require_auth
def create_portfolio():
    if ModelPortfolio.query.filter_by(user_id=g.user.id).first():
        return jsonify({"error": "You already have a portfolio. Each user can only have 1 model portfolio."}), 400
    data = request.json or {}
    pf = ModelPortfolio(
        user_id=g.user.id,
        name=data.get("name", "My Portfolio").strip() or "My Portfolio",
        description=data.get("description", "").strip(),
        category=data.get("category", "flexicap"),
        cash=100.0,
    )
    db.session.add(pf)
    db.session.commit()
    return jsonify({"portfolio": pf.to_dict(include_holdings=True)}), 201


@app.route("/api/portfolio/settings", methods=["PUT"])
@require_auth
def update_portfolio_settings():
    pf = ModelPortfolio.query.filter_by(user_id=g.user.id).first()
    if not pf:
        return jsonify({"error": "Portfolio not found"}), 404
    data = request.json or {}
    if "name" in data:
        pf.name = data["name"].strip() or pf.name
    if "description" in data:
        pf.description = data["description"].strip()
    if "category" in data:
        pf.category = data["category"]
    if "isPublic" in data:
        pf.is_public = bool(data["isPublic"])
    db.session.commit()
    return jsonify({"portfolio": pf.to_dict(include_holdings=True)})


@app.route("/api/portfolio/buy", methods=["POST"])
@require_auth
def buy_stock():
    pf = ModelPortfolio.query.filter_by(user_id=g.user.id).first()
    if not pf:
        return jsonify({"error": "Create a portfolio first"}), 400
    data = request.json or {}
    symbol = (data.get("symbol") or "").strip().upper()
    weight = float(data.get("weight") or 0)
    duration = data.get("duration") or "12m"

    if not symbol or weight <= 0 or weight > 100:
        return jsonify({"error": "Invalid symbol or weight"}), 400

    # Check total weight
    active_holdings = [h for h in pf.holdings if not h.is_sold]
    existing = next((h for h in active_holdings if h.symbol == symbol), None)
    used_weight = sum(h.weight for h in active_holdings)
    if existing:
        used_weight -= existing.weight
    if used_weight + weight > 100:
        return jsonify({"error": f"Weight exceeds 100%. You have {round(100 - used_weight, 2)}% available."}), 400

    # Fetch live price
    price = get_stock_price(symbol)
    if not price:
        return jsonify({"error": f"Could not fetch price for {symbol}. Please verify the NSE symbol."}), 400

    name = data.get("name") or get_stock_name(symbol)

    if existing:
        # Update existing holding
        existing.weight = weight
        existing.duration = duration
    else:
        h = PortfolioHolding(
            portfolio_id=pf.id,
            symbol=symbol,
            name=name,
            weight=weight,
            duration=duration,
            purchase_price=price,
            current_price=price,
            purchase_date=date.today(),
        )
        db.session.add(h)

    # Update cash (cash = 100 - deployed weight as proxy)
    all_active = [h for h in pf.holdings if not h.is_sold]
    if existing:
        all_active_after = [h for h in all_active if h.symbol != symbol]
        total_weight = sum(h.weight for h in all_active_after) + weight
    else:
        total_weight = sum(h.weight for h in all_active) + weight
    pf.cash = round(max(0, 100 - total_weight), 4)
    db.session.commit()
    refresh_portfolio_prices(pf)
    return jsonify({"portfolio": pf.to_dict(include_holdings=True)}), 201


@app.route("/api/portfolio/sell/<int:holding_id>", methods=["POST"])
@require_auth
def sell_stock(holding_id):
    pf = ModelPortfolio.query.filter_by(user_id=g.user.id).first()
    if not pf:
        return jsonify({"error": "Portfolio not found"}), 404
    h = PortfolioHolding.query.filter_by(id=holding_id, portfolio_id=pf.id).first()
    if not h or h.is_sold:
        return jsonify({"error": "Holding not found or already sold"}), 404

    price = get_stock_price(h.symbol)
    if not price:
        price = h.current_price or h.purchase_price

    h.is_sold = True
    h.sold_price = price
    h.sold_date = date.today()

    # Return weight to cash (as virtual cash increase from P&L)
    pnl_return = (price - h.purchase_price) / h.purchase_price * (h.weight)
    pf.cash = round(pf.cash + h.weight + pnl_return, 4)
    db.session.commit()
    refresh_portfolio_prices(pf)
    return jsonify({"portfolio": pf.to_dict(include_holdings=True)})


@app.route("/api/portfolio/refresh", methods=["POST"])
@require_auth
def force_refresh_portfolio():
    pf = ModelPortfolio.query.filter_by(user_id=g.user.id).first()
    if not pf:
        return jsonify({"error": "Portfolio not found"}), 404
    # Clear cache to force fresh prices
    for h in pf.holdings:
        if h.symbol in _price_cache:
            del _price_cache[h.symbol]
    refresh_portfolio_prices(pf)
    return jsonify({"portfolio": pf.to_dict(include_holdings=True)})

# ═══════ PUBLIC PORTFOLIOS ═══════

@app.route("/api/public/portfolios", methods=["GET"])
@optional_auth
def get_public_portfolios():
    category = request.args.get("category")
    sort = request.args.get("sort", "return")  # return, recent, name
    page = int(request.args.get("page", 1))
    per_page = 12

    q = ModelPortfolio.query.filter_by(is_public=True)
    if category and category != "all":
        q = q.filter_by(category=category)

    if sort == "return":
        q = q.order_by(ModelPortfolio.total_return_pct.desc())
    elif sort == "recent":
        q = q.order_by(ModelPortfolio.updated_at.desc())
    else:
        q = q.order_by(ModelPortfolio.name)

    total = q.count()
    portfolios = q.offset((page - 1) * per_page).limit(per_page).all()

    return jsonify({
        "portfolios": [p.to_dict(include_holdings=False) for p in portfolios],
        "total": total,
        "page": page,
        "pages": math.ceil(total / per_page),
    })


@app.route("/api/public/portfolios/<int:pid>", methods=["GET"])
@optional_auth
def get_public_portfolio(pid):
    pf = ModelPortfolio.query.get(pid)
    if not pf:
        return jsonify({"error": "Not found"}), 404
    # Allow owner to see their own regardless, others need it public
    if not pf.is_public and (not g.user or g.user.id != pf.user_id):
        return jsonify({"error": "Portfolio is private"}), 403
    return jsonify({"portfolio": pf.to_dict(include_holdings=True)})


@app.route("/api/public/portfolios/<int:pid>/comments", methods=["GET"])
def get_comments(pid):
    comments = PortfolioComment.query.filter_by(portfolio_id=pid).order_by(PortfolioComment.created_at.desc()).all()
    return jsonify({"comments": [c.to_dict() for c in comments]})


@app.route("/api/public/portfolios/<int:pid>/comments", methods=["POST"])
@require_auth
def add_comment(pid):
    pf = ModelPortfolio.query.get(pid)
    if not pf or not pf.is_public:
        return jsonify({"error": "Portfolio not found or private"}), 404
    data = request.json or {}
    text = (data.get("text") or "").strip()
    if not text or len(text) > 1000:
        return jsonify({"error": "Comment must be 1-1000 characters"}), 400
    c = PortfolioComment(portfolio_id=pid, user_id=g.user.id, text=text)
    db.session.add(c)
    db.session.commit()
    return jsonify({"comment": c.to_dict()}), 201

# ═══════ COMPETITIONS ═══════

@app.route("/api/competitions", methods=["GET"])
@optional_auth
def list_competitions():
    page = int(request.args.get("page", 1))
    per_page = 10
    q = Competition.query.filter_by(is_active=True).order_by(Competition.created_at.desc())
    total = q.count()
    comps = q.offset((page - 1) * per_page).limit(per_page).all()
    return jsonify({
        "competitions": [c.to_dict() for c in comps],
        "total": total,
        "page": page,
        "pages": math.ceil(total / per_page) if total else 1,
    })


@app.route("/api/competitions", methods=["POST"])
@require_auth
def create_competition():
    # Must have a portfolio to create a competition
    pf = ModelPortfolio.query.filter_by(user_id=g.user.id).first()
    if not pf:
        return jsonify({"error": "You need a portfolio to create a competition"}), 400
    data = request.json or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Competition name required"}), 400
    try:
        start_date = datetime.strptime(data.get("startDate", ""), "%Y-%m-%d").date()
        end_date = datetime.strptime(data.get("endDate", ""), "%Y-%m-%d").date()
    except Exception:
        return jsonify({"error": "Invalid dates. Use YYYY-MM-DD format."}), 400
    if end_date <= start_date:
        return jsonify({"error": "End date must be after start date"}), 400

    invite_code = secrets.token_urlsafe(8).upper()[:10]
    comp = Competition(
        creator_id=g.user.id,
        name=name,
        description=(data.get("description") or "").strip(),
        invite_code=invite_code,
        max_participants=min(int(data.get("maxParticipants", 20)), 100),
        start_date=start_date,
        end_date=end_date,
    )
    db.session.add(comp)
    db.session.flush()
    # Creator auto-joins
    participant = CompetitionParticipant(
        competition_id=comp.id,
        user_id=g.user.id,
        baseline_value=pf.total_value,
    )
    db.session.add(participant)
    db.session.commit()
    return jsonify({"competition": comp.to_dict(include_participants=True)}), 201


@app.route("/api/competitions/<int:cid>", methods=["GET"])
@optional_auth
def get_competition(cid):
    comp = Competition.query.get(cid)
    if not comp:
        return jsonify({"error": "Not found"}), 404
    # Auto-declare winner if ended
    if comp.is_active and date.today() > comp.end_date:
        _declare_winner(comp)
    data = comp.to_dict(include_participants=True)
    # Sort leaderboard
    data["participants"].sort(key=lambda x: x["returnPct"], reverse=True)
    for i, p in enumerate(data["participants"]):
        p["rank"] = i + 1
    return jsonify({"competition": data})


@app.route("/api/competitions/join/<invite_code>", methods=["POST"])
@require_auth
def join_competition(invite_code):
    comp = Competition.query.filter_by(invite_code=invite_code.upper()).first()
    if not comp:
        return jsonify({"error": "Invalid invite code"}), 404
    if not comp.is_active:
        return jsonify({"error": "This competition has ended"}), 400
    if date.today() > comp.end_date:
        return jsonify({"error": "Competition has ended"}), 400
    if len(comp.participants) >= comp.max_participants:
        return jsonify({"error": "Competition is full"}), 400

    existing = CompetitionParticipant.query.filter_by(
        competition_id=comp.id, user_id=g.user.id
    ).first()
    if existing:
        return jsonify({"error": "Already joined this competition"}), 400

    pf = ModelPortfolio.query.filter_by(user_id=g.user.id).first()
    if not pf:
        return jsonify({"error": "You need a portfolio to join a competition"}), 400

    participant = CompetitionParticipant(
        competition_id=comp.id,
        user_id=g.user.id,
        baseline_value=pf.total_value,
    )
    db.session.add(participant)
    db.session.commit()
    return jsonify({"competition": comp.to_dict(include_participants=True)})


@app.route("/api/competitions/my", methods=["GET"])
@require_auth
def my_competitions():
    participations = CompetitionParticipant.query.filter_by(user_id=g.user.id).all()
    comp_ids = [p.competition_id for p in participations]
    comps = Competition.query.filter(Competition.id.in_(comp_ids)).all()
    return jsonify({"competitions": [c.to_dict() for c in comps]})


def _declare_winner(comp):
    """Find participant with highest portfolio return and mark as winner."""
    best = None
    best_return = -999999
    for p in comp.participants:
        pf = ModelPortfolio.query.filter_by(user_id=p.user_id).first()
        ret = pf.total_return_pct if pf else 0
        if ret > best_return:
            best_return = ret
            best = p.user_id
    if best:
        comp.winner_user_id = best
    comp.is_active = False
    db.session.commit()

# ═══════ LEADERBOARD ═══════

@app.route("/api/leaderboard", methods=["GET"])
def leaderboard():
    top = (ModelPortfolio.query
           .filter_by(is_public=True)
           .order_by(ModelPortfolio.total_return_pct.desc())
           .limit(50)
           .all())
    return jsonify({"leaderboard": [p.to_dict() for p in top]})

# ═══════ ADMIN ═══════

@app.route("/api/admin/stats", methods=["GET"])
@require_auth
def admin_stats():
    if not g.user.is_admin:
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify({
        "users": User.query.count(),
        "portfolios": ModelPortfolio.query.count(),
        "publicPortfolios": ModelPortfolio.query.filter_by(is_public=True).count(),
        "competitions": Competition.query.count(),
        "activeCompetitions": Competition.query.filter_by(is_active=True).count(),
    })

# ═══════ HEALTH ═══════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})

# ═══════ INIT ═══════

with app.app_context():
    db.create_all()
    log.info("[DB] Tables created/verified")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
