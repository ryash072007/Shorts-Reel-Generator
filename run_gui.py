"""
Launch the Reddit Meme to Video Generator GUI

This script is a simple launcher for the GUI application.
"""

import os
import sys
from redditmeme2video.gui import main

if __name__ == "__main__":
    # Print usage information
    print("\nReddit Meme to Video Generator")
    print("=============================")
    print("1. Click 'Analyze Memes' to fetch and select memes")
    print("2. Select the memes you want to include in your video")
    print("3. Edit captions with the SSML Editor if desired")
    print("4. Reorder memes by dragging in the selection window")
    print("5. Click 'Generate Video' to create your video\n")
    
    # Launch the GUI
    main()
