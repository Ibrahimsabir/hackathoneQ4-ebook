# Deployment Guide

This guide will help you deploy the Physical AI & Humanoid Robotics ebook with RAG chatbot to production.

## Architecture

- **Frontend (Docusaurus)**: Deployed on Vercel
- **Backend (FastAPI)**: Deployed on Railway or Render
- **Vector Database**: Qdrant Cloud (already configured)

## Option 1: Deploy Backend to Railway (Recommended)

Railway offers free tier with excellent Docker support.

### Steps:

1. **Sign up for Railway**: https://railway.app
   - Connect your GitHub account

2. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `Ibrahimsabir/hackathoneQ4-ebook`

3. **Configure Environment Variables**:
   - Go to project → Variables tab
   - Add these variables:
     ```
     COHERE_API_KEY=<your_cohere_key>
     QDRANT_URL=<your_qdrant_url>
     QDRANT_API_KEY=<your_qdrant_key>
     OPENROUTER_API_KEY=<your_openrouter_key>
     OPENAI_API_KEY=<your_openai_key>
     PORT=8000
     ```

4. **Deploy**:
   - Railway will automatically detect `railway.json` and `backend/Dockerfile`
   - Wait for deployment to complete (2-3 minutes)
   - You'll get a URL like: `https://your-app.railway.app`

5. **Test Backend**:
   ```bash
   curl https://your-app.railway.app/api/health
   ```

## Option 2: Deploy Backend to Render

Render also offers a free tier for web services.

### Steps:

1. **Sign up for Render**: https://render.com
   - Connect your GitHub account

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect to `Ibrahimsabir/hackathoneQ4-ebook`
   - Render will detect `render.yaml` automatically

3. **Configure**:
   - Name: `rag-chatbot-backend`
   - Environment: Python 3
   - Build Command: `cd backend && pip install -r requirements.txt`
   - Start Command: `cd backend && uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`

4. **Add Environment Variables**:
   - In Render dashboard → Environment tab
   - Add all the API keys from `.env.example`

5. **Deploy**:
   - Click "Create Web Service"
   - Wait for deployment (3-5 minutes)
   - You'll get a URL like: `https://rag-chatbot-backend.onrender.com`

## Step 3: Update Frontend to Use Deployed Backend

After backend is deployed, update the chatbot to use the production URL:

### Edit `src/pages/chatbot.tsx`:

**Line 57:** Change from:
```typescript
const response = await fetch('http://localhost:8000/api/ask', {
```

**To:**
```typescript
const backendUrl = process.env.NODE_ENV === 'production'
  ? 'https://your-railway-or-render-url.app'
  : 'http://localhost:8000';
const response = await fetch(`${backendUrl}/api/ask`, {
```

### Or use environment variable:

1. **Add to `docusaurus.config.ts`** (customFields section):
```typescript
customFields: {
  BACKEND_URL: process.env.BACKEND_URL || 'http://localhost:8000',
},
```

2. **Update chatbot.tsx** to use:
```typescript
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
// In component:
const { siteConfig } = useDocusaurusContext();
const backendUrl = siteConfig.customFields.BACKEND_URL;
```

3. **Add to Vercel Environment Variables**:
```
BACKEND_URL=https://your-deployed-backend.railway.app
```

## Step 4: Redeploy Frontend

After updating the backend URL:

```bash
git add src/pages/chatbot.tsx docusaurus.config.ts
git commit -m "Update chatbot to use production backend URL"
git push origin main
```

Vercel will auto-deploy the updated frontend.

## Verification Checklist

After deployment:

- [ ] Frontend loads on Vercel URL
- [ ] Homepage shows new professional UI
- [ ] Custom SVG icons display correctly
- [ ] Backend health check works: `https://your-backend-url/api/health`
- [ ] Chatbot page loads
- [ ] Chatbot can send/receive messages
- [ ] Mobile responsive design works
- [ ] Dark mode functions properly

## Cost Breakdown

- **Vercel (Frontend)**: Free tier (sufficient for this project)
- **Railway (Backend)**: Free tier includes $5/month credit (enough for light usage)
- **Render (Backend)**: Free tier (sleeps after 15min inactivity, wakes on request)
- **Qdrant Cloud**: Free tier (already configured)
- **OpenRouter**: Free tier for Mistral model

## Troubleshooting

**Backend won't start:**
- Check environment variables are set correctly
- Verify Qdrant URL is accessible from deployment platform
- Check logs for specific errors

**Chatbot not connecting:**
- Verify CORS is enabled (already configured in main.py)
- Check backend URL in chatbot.tsx
- Test backend health endpoint directly

**Slow responses:**
- Free tier may have cold starts
- Consider upgrading to paid tier for production use

## Next Steps

1. Choose Railway or Render for backend deployment
2. Deploy backend following steps above
3. Get your backend URL
4. Update frontend with backend URL
5. Test end-to-end functionality
6. Monitor and optimize as needed

Need help with any specific step? Let me know!
