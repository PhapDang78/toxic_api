# toxic_api# Moderation API (text + image)

This repo provides a FastAPI app for text toxicity and image SafeSearch moderation.

Quick start:
1. Create a `.env` file in the project root (you can copy from `.env.example`):
   ```
   GOOGLE_API_KEY=your_google_vision_api_key_here
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   uvicorn main:app --reload
   ```

Notes:
- `.env` is ignored by git (see `.gitignore`).
- The app uses `python-dotenv` to automatically load environment variables from `.env`.
- If `GOOGLE_API_KEY` is not set, image moderation endpoint will raise an error explaining it.