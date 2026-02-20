# PortfolioPulse — Setup Guide

## What You've Got
- `index.html` — Complete frontend (single file, works directly in browser)
- `app.py` — Flask backend with all APIs
- `requirements.txt`, `Procfile`, `runtime.txt`, `render.yaml` — Deployment files

---

## Step 1: Deploy the Backend on Render.com (Free)

1. Create a free account at **render.com**
2. Click **New → Web Service**
3. Connect your GitHub repo (upload all backend files: `app.py`, `requirements.txt`, `Procfile`, `runtime.txt`, `render.yaml`)
4. Render will auto-detect settings from `render.yaml`
5. Add a **PostgreSQL database** on Render (free tier)
6. Set environment variables in Render dashboard:
   - `DATABASE_URL` → auto-filled from the database
   - `SECRET_KEY` → any random string (e.g. paste from: https://randomkeygen.com)
   - `GOOGLE_CLIENT_ID` → from Google Console (Step 2)
   - `FRONTEND_URL` → URL where your `index.html` is hosted
   - `ADMIN_EMAILS` → your email address
7. Deploy! Render will give you a URL like `https://portfoliopulse-api.onrender.com`

---

## Step 2: Set Up Google Sign-In (Optional but Recommended)

1. Go to **console.cloud.google.com**
2. Create a new project → APIs & Services → Credentials
3. Create **OAuth 2.0 Client ID** (Web Application type)
4. Add your frontend URL to Authorized JavaScript Origins
5. Copy the Client ID

---

## Step 3: Update index.html

Open `index.html` and find these two lines near the top of the JavaScript:

```javascript
const API = 'https://your-backend-url.onrender.com';  // ← CHANGE THIS
const GOOGLE_CLIENT_ID = 'YOUR_GOOGLE_CLIENT_ID';      // ← CHANGE THIS
```

Replace with your actual Render URL and Google Client ID.

---

## Step 4: Host index.html

**Easiest options:**
- **GitHub Pages** (free): Upload `index.html` to a GitHub repo, enable Pages
- **Netlify** (free): Drag & drop `index.html` at netlify.com/drop
- **Vercel** (free): Similar to Netlify

---

## Features Summary

### Portfolio Management
- Create 1 model portfolio per user
- Start with ₹100 virtual cash
- Add stocks by NSE symbol (auto-fetches live price at time of purchase)
- Purchase price is LOCKED at buy time — cannot be changed
- Track daily P&L: (CMP - Buy Price) × 100 × weight%
- Sell stocks (updates cash, logs sold price)
- Weight gauge shows deployed vs. available allocation

### Public Portfolios
- Publish portfolio with category (Large/Mid/Small/Flexi cap)
- Add investment description, bio, LinkedIn profile
- Anyone can browse and comment
- Leaderboard by total return %

### Competitions
- Create competition: set name, dates, max participants
- Get unique invite code to share
- Friends join with the code
- Live leaderboard during competition
- Auto-declares winner when competition ends

### Stock Data
- Uses Yahoo Finance (yfinance) — free, no API key needed
- Supports all NSE-listed stocks
- Prices cached for 1 hour (refreshed on demand)

---

## Troubleshooting

**"Could not fetch price"** — Make sure you're using the NSE ticker symbol (e.g., RELIANCE, TCS, INFY, HDFC, not HDFC.NS — the .NS is added automatically)

**Prices showing 0** — The free Render instance sleeps after inactivity. First request after sleep may take 30-60 seconds.

**Google Sign-In not working** — Make sure GOOGLE_CLIENT_ID is set in both the backend env vars and in index.html

---

## Architecture

```
Frontend (index.html) ←→ Backend (app.py on Render) ←→ PostgreSQL DB
                                      ↓
                              Yahoo Finance API (yfinance)
```

All in one HTML file, all backend in one Python file. Simple and maintainable!
