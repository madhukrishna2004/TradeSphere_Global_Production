import os
from src.app import app  # ensure src/__init__.py exists

# ✅ This ensures Gunicorn can see `app` directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
