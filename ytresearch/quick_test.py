
"""
Quick test script for the YouTube Shorts Research tool.
This runs a minimal analysis to test if everything is working.
"""

from research import YouTubeRedditResearcher
import os
from dotenv import load_dotenv

def run_quick_test():
    """Run a quick test of the YouTube Shorts Research tool."""
    print("Running quick test of the YouTube Shorts Research tool...")
    
    # Load environment variables
    load_dotenv()
    
    # Check if YouTube API key is available
    youtube_api_key = os.environ.get('YOUTUBE_API_KEY')
    if not youtube_api_key:
        print("Error: YouTube API key not found in .env file.")
        return False
    
    try:
        # Initialize the researcher
        researcher = YouTubeRedditResearcher()
        
        # Run a small search
        print("Searching for a few YouTube Shorts...")
        results = researcher.search_reddit_shorts(
            query_terms=["reddit", "AITA"],
            max_results=5
        )
        
        if not results:
            print("No search results found. Something might be wrong with your API key or search terms.")
            return False
            
        print(f"Found {len(results)} videos. Test successful!")
        print("\nExample video:")
        video = results[0]
        print(f"Title: {video['title']}")
        print(f"Channel: {video['channel_title']}")
        print(f"Views: {video['view_count']:,}")
        
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_quick_test()
    if success:
        print("\nQuick test completed successfully! The tool appears to be working.")
    else:
        print("\nQuick test failed. Please check the error messages above.")
