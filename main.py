import os
from src.app import app

if __name__ == "__main__":
    # Get the port Render provides, default to 5000 if running locally
    port = int(os.environ.get("PORT", 5000))
    
    # Bind to all network interfaces (0.0.0.0) so Render can reach it
    app.run(host="0.0.0.0", port=port)
