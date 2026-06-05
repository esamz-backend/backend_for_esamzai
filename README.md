# eSAMz v9.1 — Rust Backend (Render.com)

## Repo structure
```
esamz-backend/
├── Cargo.toml        ← Rust dependencies
├── render.yaml       ← Render auto-detects this
├── .gitignore
└── src/
    └── main.rs       ← full backend
```

---

## Deploy to Render (web only, no terminal)

### Step 1 — Push to GitHub
1. Go to **github.com** → click **+** → **New repository**
2. Name it `esamz-backend` → click **Create repository**
3. Click **uploading an existing file**
4. Upload ALL files keeping the folder structure:
   - `Cargo.toml` → root
   - `render.yaml` → root
   - `.gitignore` → root
   - `src/main.rs` → inside a folder called `src`
5. Click **Commit changes**

---

### Step 2 — Create Render account
1. Go to **render.com**
2. Click **Get Started for Free**
3. Sign up with **GitHub** (easiest — links repos automatically)

---

### Step 3 — Create Web Service
1. In Render dashboard click **New +** → **Web Service**
2. Click **Connect a repository**
3. Select your `esamz-backend` repo
4. Click **Connect**

---

### Step 4 — Configure service settings
Fill in exactly:

| Field            | Value                      |
|------------------|----------------------------|
| Name             | `esamz-backend`            |
| Region           | Frankfurt (closest to India) |
| Branch           | `main`                     |
| Runtime          | `Rust`                     |
| Build Command    | `cargo build --release`    |
| Start Command    | `./target/release/esamz`   |
| Instance Type    | **Free**                   |
| Health Check Path| `/health`                  |

---

### Step 5 — Add environment variables
Scroll down to **Environment Variables** section and add these one by one:

| Key                | Value                  |
|--------------------|------------------------|
| `GOOGLE_API_KEY`   | your actual key        |
| `SERPER_API_KEY`   | your actual key        |
| `KV_REST_API_URL`  | your actual URL        |
| `KV_REST_API_TOKEN`| your actual token      |
| `PRIVACY_MODE`     | `false`                |
| `ENVIRONMENT`      | `production`           |
| `RUST_LOG`         | `esamz=info`           |

---

### Step 6 — Deploy
1. Click **Create Web Service**
2. Render will start building (takes ~5-8 minutes first time — Rust compiles slowly)
3. Watch the build logs — you'll see `Listening on http://0.0.0.0:PORT`
4. Your URL will be: `https://esamz-backend.onrender.com`

---

### Step 7 — Update your frontend
Change the API URL in your frontend JavaScript:
```js
// Change this one line
const API_URL = "https://esamz-backend.onrender.com"
```

---

### Step 8 — Stop it sleeping (free tier fix)

Free Render services sleep after 15 minutes idle. Fix with UptimeRobot:

1. Go to **uptimerobot.com** → Sign up free
2. Click **Add New Monitor**
3. Monitor Type: **HTTP(s)**
4. URL: `https://esamz-backend.onrender.com/health`
5. Monitoring Interval: **5 minutes**
6. Click **Create Monitor**

Done — your backend now stays awake 24/7 for free.

---

## Every future deploy
Just edit any file on GitHub → Render auto-deploys within seconds.
No terminal ever needed.

---

## Your live URLs
- Backend API: `https://esamz-backend.onrender.com`
- Health check: `https://esamz-backend.onrender.com/health`
- Chat endpoint: `https://esamz-backend.onrender.com/api/chat`
