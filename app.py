# app.py  (root)
import sys
import os

# Add subfolder to Python import path
sys.path.append(os.path.join(os.getcwd(), "Heart_Disease"))

# Import actual app module
from Heart_Disease import app as heart_app

if __name__ == "__main__":
    heart_app.main()
