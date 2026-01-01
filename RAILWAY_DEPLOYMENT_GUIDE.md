# Railway Backend Deployment Guide - Step by Step

Follow these exact steps to deploy your RAG chatbot backend to Railway.

## ✅ Prerequisites Check

- [x] Code pushed to GitHub ✅
- [x] `railway.json` exists ✅
- [x] `backend/Dockerfile` exists ✅
- [x] `backend/requirements.txt` exists ✅
- [ ] Railway account (we'll create this)
- [ ] Environment variables ready (from your .env file)

---

## Step 1: Sign Up for Railway

1. Go to: **https://railway.app**
2. Click **"Start a New Project"** or **"Login"**
3. Choose **"Login with GitHub"**
4. Authorize Railway to access your GitHub account
5. You'll get $5/month free credit (enough for this project)

---

## Step 2: Create New Project from GitHub

1. **Click "New Project"** button (top right)
2. Select **"Deploy from GitHub repo"**
3. You'll see a list of your repositories
4. Find and click: **"Ibrahimsabir/hackathoneQ4-ebook"**
5. Railway will automatically detect your `railway.json` configuration

---

## Step 3: Configure the Service

After selecting your repo:

1. Railway will show "Configure Service"
2. **Service Name:** Keep default or name it `rag-chatbot-api`
3. **Root Directory:** Leave empty (Railway will use railway.json config)
4. Click **"Deploy"**

Railway will start building using your Dockerfile.

---

## Step 4: Add Environment Variables

While it's building, add your environment variables:

1. Click on your service (in Railway dashboard)
2. Go to **"Variables"** tab
3. Click **"+ New Variable"** and add each one:

**Add these 5 variables:**

Copy the values from your local `.env` file:

```
Variable Name: COHERE_API_KEY
Value: <copy from your .env file>

Variable Name: QDRANT_URL
Value: <copy from your .env file>

Variable Name: QDRANT_API_KEY
Value: <copy from your .env file>

Variable Name: OPENROUTER_API_KEY
Value: <copy from your .env file>

Variable Name: OPENAI_API_KEY
Value: <copy from your .env file>
```

**How to copy values:**
- Open your local `.env` file
- Copy each value (the part after the `=`)
- Paste into Railway variable value field

4. Click **"Add"** for each variable

⚠️ **Important:** After adding all variables, Railway will automatically redeploy.

---

## Step 5: Generate Public Domain

1. In your service dashboard, go to **"Settings"** tab
2. Scroll down to **"Networking"** section
3. Click **"Generate Domain"**
4. Railway will give you a URL like: `https://rag-chatbot-api-production-xxxx.up.railway.app`
5. **Copy this URL** - you'll need it for the next step!

---

## Step 6: Wait for Deployment

1. Go to **"Deployments"** tab
2. Watch the build logs (click on the latest deployment)
3. Wait for status to show **"SUCCESS"** (usually 2-3 minutes)
4. Look for this line in logs:
   ```
   INFO:     Uvicorn running on http://0.0.0.0:8000
   INFO:     Application startup complete.
   ```

---

## Step 7: Test Your Backend

Once deployed, test the health endpoint:

**Replace `YOUR-RAILWAY-URL` with your actual Railway URL:**

```bash
curl https://YOUR-RAILWAY-URL.railway.app/api/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "vector_store": "connected",
  "llm_service": "available",
  "timestamp": "2026-01-01T..."
}
```

**If you see this ✅ Your backend is live!**

---

## Step 8: Update Frontend to Use Production Backend

Now we need to update your chatbot to use the Railway backend URL instead of localhost.

**I'll help you do this in the next step** - just tell me your Railway URL and I'll update the code!

---

## What You'll Get

After completing all steps:

- ✅ **Frontend on Vercel:** `https://your-site.vercel.app`
- ✅ **Backend on Railway:** `https://your-backend.railway.app`
- ✅ **Fully working chatbot** accessible from anywhere!
- ✅ **Professional UI** with custom design
- ✅ **No local server needed** - everything in the cloud

---

## Troubleshooting

**Build fails:**
- Check environment variables are set correctly
- Look at build logs for specific errors
- Ensure all variables are added

**Backend starts but health check fails:**
- Verify Qdrant URL is accessible
- Check API keys are valid
- Review application logs in Railway dashboard

**CORS errors:**
- Our FastAPI app already has CORS enabled for all origins
- Should work automatically

---

## Cost

**Railway Free Tier:**
- $5/month credit (resets monthly)
- Enough for development and light production use
- Sleeps after inactivity (wakes automatically on request)

**If you exceed free tier:**
- Costs ~$5-10/month for this size app
- Can upgrade as needed

---

## Ready?

Follow the steps above, and once you have your **Railway URL**, send it to me and I'll update the frontend chatbot to use it!

Then we'll push one final commit and your entire system will be **100% deployed and working!** 🚀
