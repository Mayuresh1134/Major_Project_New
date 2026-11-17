"""Start MuseAssist: python app.py, then open http://127.0.0.1:5000"""
import os

from museassist import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host=os.environ.get("MUSE_HOST", "127.0.0.1"), port=int(os.environ.get("MUSE_PORT", "5000")),
            debug=os.environ.get("MUSE_DEBUG") == "1", threaded=True)
