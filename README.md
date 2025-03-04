# Shorts Reel Generator

A tool for creating short-form videos from Reddit memes with AI-generated captions and TTS narration.

## Features

- Browse and select popular memes from Reddit
- Automatically generate captions with AI image analysis
- Edit captions with SSML support for precise voice control
- Generate videos with synchronized captions and narration
- Add background music
- Queue multiple video generation jobs

## Setup

### Prerequisites

- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) installed and in your PATH

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Shorts-Reel-Generator.git
   cd Shorts-Reel-Generator
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API key:
   ```
   # On Windows
   set GROQ_API_KEY=your_api_key_here
   
   # On Linux/Mac
   export GROQ_API_KEY=your_api_key_here
   ```

## Usage

### Starting the GUI

Run the GUI application:

```
python -m redditmeme2video.gui
```

### Workflow

1. **Configure Settings**
   - Select subreddits to pull memes from
   - Adjust voice settings
   - Set minimum upvotes threshold
   - Choose output location

2. **Analyze Memes**
   - Click "Analyze Memes" to fetch and select memes
   - Choose which memes to include in your video

3. **Edit Captions**
   - Edit the automatically generated captions
   - Use SSML tags for voice control
   - Preview how the voice will sound

4. **Generate Video**
   - Click "Generate Video" to create your short-form video
   - Monitor progress in the queue panel
   - View logs for detailed generation information

5. **Review Results**
   - When complete, open the video directly from the queue
   - Copy generated title, description, and tags for uploading

## SSML Caption Editing

The application includes a full-featured SSML editor for controlling voice characteristics:

- Add pauses with `<break time="500ms"/>`
- Emphasize words with `<emphasis level="strong">text</emphasis>`
- Control speaking rate with `<prosody rate="slow">text</prosody>`
- Adjust pitch with `<prosody pitch="high">text</prosody>`

## Troubleshooting

- **Missing Voice or TTS Issues**: Make sure to have an internet connection for Edge TTS
- **Generation Errors**: Check the logs for detailed error information
- **API Key Issues**: Verify your GROQ API key is correctly set as an environment variable

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.