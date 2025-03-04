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

### GUI Interface

The GUI is divided into three main panels:

1. **Configuration Panel** (Left) - Configure all settings for video generation
2. **Preview Panel** (Right) - Preview and edit selected memes and captions
3. **Queue Panel** (Bottom) - Manage multiple video generation jobs

### Detailed Workflow

1. **Configure Settings**
   - Select subreddits to pull memes from (add multiple for batch generation)
   - Adjust voice settings (voice, rate, volume, pitch)
   - Set minimum upvotes threshold
   - Choose output location
   - Configure background music and other options

2. **Analyze Memes**
   - Click "Analyze Memes" to fetch memes from the selected subreddit
   - Use the selection dialog to choose which memes to include
   - Reorder memes by dragging and dropping in the selection dialog
   - Click "OK" to proceed with selected memes

3. **Edit Captions**
   - Navigate between selected memes using the "Previous" and "Next" buttons
   - Edit auto-generated captions directly in the text area
   - Click "Save Caption" to save your edits
   - Use "Edit SSML" for advanced voice control options
   - Click "Preview Voice" to hear how the captions will sound

4. **Generate Video**
   - Click "Generate Video" to add a generation job to the queue
   - Monitor progress in the Queue Panel
   - Multiple videos can be queued (one per subreddit)
   - Set concurrent processing tasks (1-4) in the Queue Panel

5. **Review Results**
   - When complete, click "Open" to view the generated video
   - Click "View Metadata" to see title, description, and tags
   - Use "Copy Title", "Copy Description", etc. buttons to copy content for uploading

### Configuration Panel Options

- **Subreddit Selection**: Add multiple subreddits for batch processing
- **Basic Settings**:
  - Memes per video: Number of memes to include in each video (1-10)
  - Minimum upvotes: Filter out memes with fewer upvotes
  - Post type: Choose from hot, top, new, rising posts
  - Auto mode: Enable to skip manual selection
  - Background music: Toggle background music on/off

- **TTS Voice Settings**:
  - Voice: Select from multiple available voices
  - Rate: Adjust speaking speed
  - Volume: Adjust voice volume
  - Pitch: Fine-tune voice pitch
  - Test Voice: Preview current voice settings

- **Appearance Settings**:
  - Dark mode: Toggle dark/light theme
  - Animation level: Adjust UI animation effects

- **Output Settings**:
  - Output directory: Select where videos are saved
  - Auto upload: Enable automatic YouTube uploads (requires setup)

### Preview Panel Features

- **Image Preview**: View selected meme images
- **Caption Editor**: Edit captions with syntax highlighting
- **SSML Editor**: Advanced editing with Speech Synthesis Markup Language
- **Voice Preview**: Listen to how captions will sound
- **Metadata Display**: View post title, author, and URL

### Queue Panel Features

- **Job Management**:
  - Monitor multiple video generation jobs
  - View real-time progress and status
  - Cancel queued jobs
  - Retry failed jobs

- **Log Viewer**: See detailed generation logs for each job
- **Metadata Viewer**: Access generated video metadata
- **Concurrent Tasks**: Configure how many videos can be processed simultaneously

### SSML Caption Editing

The application includes a full-featured SSML editor for controlling voice characteristics:

- Add pauses with `<break time="500ms"/>`
- Emphasize words with `<emphasis level="strong">text</emphasis>`
- Control speaking rate with `<prosody rate="slow">text</prosody>`
- Adjust pitch with `<prosody pitch="high">text</prosody>`

Access the SSML editor by clicking "Edit SSML" in the Preview Panel.

## Troubleshooting

- **Missing Voice or TTS Issues**: Make sure to have an internet connection for Edge TTS
- **Generation Errors**: Check the logs in the Queue Panel for detailed error information
- **API Key Issues**: Verify your GROQ API key is correctly set as an environment variable
- **UI Responsiveness**: If the UI becomes unresponsive during heavy processing, try reducing the "Concurrent Tasks" setting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.