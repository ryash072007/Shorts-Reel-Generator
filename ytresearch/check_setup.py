
"""
Diagnostic script to check if your environment is properly set up
for the YouTube Shorts Research tool.
"""

import os
from dotenv import load_dotenv
import sys
from pathlib import Path

def check_setup():
    """Check if the environment is properly set up."""
    print("YouTube Shorts Research Tool - Setup Check")
    print("------------------------------------------")
    
    # Check if .env file exists
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found. Please copy .env.example to .env and fill in your API keys.")
        print("   Command: cp .env.example .env")
        return False
        
    # Load environment variables
    load_dotenv()
    
    # Check required API keys
    youtube_api_key = os.environ.get('YOUTUBE_API_KEY')
    if not youtube_api_key:
        print("❌ YOUTUBE_API_KEY not found in .env file. This is required.")
        has_error = True
    else:
        print("✅ YouTube API key found.")
        
    # Check optional API keys
    reddit_client_id = os.environ.get('REDDIT_CLIENT_ID')
    reddit_client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
    if not reddit_client_id or not reddit_client_secret:
        print("⚠️ Reddit API credentials not found. Some features will be limited.")
    else:
        print("✅ Reddit API credentials found.")
        
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        print("⚠️ Groq API key not found. Advanced AI analysis will not be available.")
    else:
        print("✅ Groq API key found.")
    
    # Check for required Python packages
    try:
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import requests
        from googleapiclient.discovery import build
        print("✅ Core Python packages found.")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("   Please run: pip install pandas numpy matplotlib seaborn google-api-python-client requests")
        return False
        
    # Check optional packages
    try:
        import praw
        import wordcloud
        print("✅ Optional Python packages found.")
    except ImportError as e:
        print(f"⚠️ Missing optional package: {e}")
        print("   For full functionality, run: pip install praw wordcloud")
    
    print("\nSetup check completed. If you see any errors above, please address them.")
    return True

if __name__ == "__main__":
    check_setup()
