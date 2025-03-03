# YouTube Shorts Research Tool

This tool helps you research and analyze successful YouTube Shorts content, with a focus on Reddit-based content. It can identify trending topics, untapped niches, and analyze your own channel's performance.

## Setup

1. Install required dependencies:
```
pip install pandas numpy matplotlib seaborn google-api-python-client tqdm praw wordcloud requests python-dotenv
```

2. Set up your API keys:
   - Copy `.env.example` to `.env`
   - Fill in your API keys (at minimum, you need a YouTube API key)

## Getting API Keys

### YouTube API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to "APIs & Services" > "Library"
4. Search for and enable "YouTube Data API v3"
5. Go to "Credentials" > "Create Credentials" > "API key"
6. Copy your new API key to the `.env` file

### Reddit API (optional)
1. Go to [Reddit's App Preferences](https://www.reddit.com/prefs/apps)
2. Click "create another app..."
3. Select "script" as the application type
4. Fill in the required fields
5. Copy the client ID and secret to the `.env` file

### Groq API (optional)
1. Go to [Groq Console](https://console.groq.com/)
2. Sign up and get an API key
3. Copy the API key to the `.env` file

## Usage

### Configuration

Simply edit the `config.py` file to customize your research:

```python
# Research mode: "niche", "channel", or "all"
MODE = "all"

# Your channel ID for personal analysis
CHANNEL_ID = "YOUR_CHANNEL_ID"

# Search settings
SEARCH_KEYWORDS = ["reddit", "AITA", "relationship"]
MAX_RESULTS = 100
MAX_CHANNELS = 30

# Output settings
OUTPUT_BASE_FILENAME = "youtube_research_results"
GENERATE_VISUALIZATIONS = False

# Advanced analysis
USE_GROQ_ANALYSIS = False
```

### Running the Research

After configuring the settings, simply run:

```bash
python run_research.py
```

## Output Files

The tool generates several files:
- `{output}.json`: Raw research data
- `{output}_basic_report.md`: Basic research findings
- `{output}_enhanced_report.md`: Detailed research with growth analysis
- `{output}_groq_analysis.md`: AI-powered analysis (if Groq is enabled)
- `{output}_channel_report.md`: Analysis of your channel (if channel analysis is enabled)

## Example Configurations

### Niche Research Only

```python
MODE = "niche"
SEARCH_KEYWORDS = ["reddit", "confession", "relationship", "AITA"]
MAX_RESULTS = 150
```

### Channel Analysis Only

```python
MODE = "channel"
CHANNEL_ID = "YOUR_CHANNEL_ID"
```

### Comprehensive Analysis with Visualizations

```python
MODE = "all"
CHANNEL_ID = "YOUR_CHANNEL_ID"
SEARCH_KEYWORDS = ["reddit", "confession", "relationship"]
GENERATE_VISUALIZATIONS = True
USE_GROQ_ANALYSIS = True
```