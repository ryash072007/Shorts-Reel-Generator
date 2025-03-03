"""
Configuration settings for the YouTube Shorts Research tool.
Modify the settings here to customize your research.
"""

# Research mode: "niche", "channel", or "all"
MODE = "niche"

# Your channel ID for personal analysis
# Example: "UCF9rg6gieaxyM3fY4VHyIXA"
CHANNEL_ID = "UCF9rg6gieaxyM3fY4VHyIXA"  # Your channel ID

# Search settings
SEARCH_KEYWORDS = ["reddit", "memes", "relationship", "historymemes", "funny"]
MAX_RESULTS = 100  # Maximum number of videos to search for
MAX_CHANNELS = 30  # Maximum number of channels to analyze

# Output settings
OUTPUT_BASE_FILENAME = "youtube_research_results"
GENERATE_VISUALIZATIONS = True  # Set to True to display visual charts

# Advanced analysis - Groq is always enabled for detailed insights
USE_GROQ_ANALYSIS = True  # Always using Groq for deeper analysis
