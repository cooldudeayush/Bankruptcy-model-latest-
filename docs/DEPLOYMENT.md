# Deploying the Bankruptcy Risk Predictor

Yes, this app is easy to deploy. It is a single Streamlit app, the five models together are about 5 MB of text,
and it needs no database or GPU. Pick one of the options below.

| Option | Cost | Effort | Best for |
|---|---|---|---|
| **1. Streamlit Community Cloud** | Free | ~5 minutes, no code | Sharing a public demo link (recommended) |
| **2. Hugging Face Spaces** | Free | ~10 minutes | A demo on your Hugging Face profile |
| **3. Docker (Render, Railway, Cloud Run, Azure, VPS)** | Free tier or paid | ~15 minutes | Full control, private or company hosting |

---

## Option 1: Streamlit Community Cloud (recommended)

The repository is already set up for it: `app.py` is at the root, `requirements.txt` lists the Python packages,
and `packages.txt` installs the OpenMP library that LightGBM needs.

1. Make sure the code is on GitHub (`https://github.com/cooldudeayush/Bankruptcy-model-latest-`).
   The repository must be **public**, or you must grant Streamlit access to private repositories.
2. Go to **https://share.streamlit.io** and sign in with your GitHub account.
3. Click **Create app**, then **Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `cooldudeayush/Bankruptcy-model-latest-`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** choose a name, e.g. `bankruptcy-risk`
5. Open **Advanced settings** and choose **Python 3.12**.
6. Click **Deploy**. The first build takes 2–4 minutes. You get a link like
   `https://bankruptcy-risk.streamlit.app` that anyone can open.

Every `git push` to `main` redeploys the app automatically.

**If the build fails:** open "Manage app" → logs at the bottom right. The most common causes are a wrong main
file path or a Python version below 3.11 (pandas 3 needs 3.11+).

---

## Option 2: Hugging Face Spaces

1. Create a free account at https://huggingface.co and click **New Space**.
2. Choose a name, select the **Docker** SDK (blank template) and **CPU basic (free)** hardware.
3. Push this repository to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
4. Spaces expect the app on port **7860**. Add these lines at the top of `README.md` (the Space's config):
   ```yaml
   ---
   title: Bankruptcy Risk Predictor
   sdk: docker
   app_port: 7860
   ---
   ```
   and change the Dockerfile's last line to use port 7860, or set the `PORT` variable to `7860` in the Space settings.
5. The Space builds the Dockerfile and shows the app on its page.

---

## Option 3: Docker (any container host)

A `Dockerfile` is included. Test it locally first:

```bash
docker build -t bankruptcy-app .
docker run -p 8501:8501 bankruptcy-app
```

Open http://localhost:8501. The container reads the `PORT` variable if the host sets one.

**Render (free tier):** New → Web Service → connect the GitHub repo → Runtime **Docker** → Create.
Render sets `PORT` automatically. Free services sleep after inactivity, so the first visit can take ~30 seconds.

**Google Cloud Run:**
```bash
gcloud run deploy bankruptcy-app --source . --region europe-west1 --allow-unauthenticated
```

**Railway / Azure Container Apps / Fly.io:** point them at the repository. They detect the Dockerfile.

---

## Updating the models

1. `pip install -r requirements-train.txt`
2. `python -m bankruptcy.train` (about 15–20 minutes on a laptop)
3. `pytest`
4. Commit the changed `models/` and `reports/` folders and push. The hosted app picks up the new models on redeploy.

## Before using it for real decisions

- The models learned from Polish companies (2000–2013). Ratios from other countries, sectors or years may behave
  differently, so check the results on your own data first.
- Treat the output as a screening signal, not financial advice.
