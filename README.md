# Shorts Reel Generator

A powerful tool for automatically generating short-form vertical videos from Reddit memes.

## Features

- **User-friendly GUI**: Configure all settings through an intuitive graphical interface
- **AI-powered**: Uses AI to analyze memes and generate natural-sounding captions
- **Text-to-Speech**: High-quality voice narration with customizable settings
- **Multi-subreddit Support**: Generate videos from any public subreddit
- **Queue System**: Set up multiple videos for batch processing
- **Preview System**: Preview memes and captions before generating videos
- **SSML Editor**: Edit speech markup for perfect voiceovers
- **Background Music**: Automatically adds background music to videos

## Requirements

- Python 3.8+
- Groq API key for AI services

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YourUsername/Shorts-Reel-Generator.git
cd Shorts-Reel-Generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# On Windows
set GROQ_API_KEY=your_api_key_here

# On Linux/Mac
export GROQ_API_KEY=your_api_key_here
```

## Usage

### GUI Mode

Run the GUI application:

```bash
python run_gui.py
```

The interface allows you to:
- Select subreddits for video generation
- Configure TTS voice settings
- Preview memes and captions
- Queue multiple videos for generation
- Customize video appearance and behavior

### Command-Line Mode

For advanced users, you can also use the command-line interface:

```bash
python -m redditmeme2video.redditmeme2video
```

## Configuration

Most settings can be configured through the GUI, but you can also:

- Save and load configuration presets
- Customize output quality and style
- Modify voice and animation settings
- Set subreddit filters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.