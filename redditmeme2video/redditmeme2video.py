"""
Reddit Meme to Video Generator

This module fetches memes from Reddit, extracts their captions using AI,
generates audio narration, and combines them into short-form videos with background.
"""

import json
import logging
import os
import random
import shutil
import sys
import threading
import time
import concurrent.futures
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import edge_tts
import numpy as np
import requests
from groq import Groq
from moviepy import *
from PIL import Image

# Add rfm module to path
sys.path.append("rfm")
from rfm import get_music_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("redditmeme2video")

# Constants
TARGET_WIDTH, TARGET_HEIGHT = 1080, 1920
FPS = 30
DEFAULT_MIN_UPVOTES = 3000
DEFAULT_VOICE = "en-AU-WilliamNeural"
DEFAULT_RATE = "+20%"
DEFAULT_PITCH = "+20Hz"

# AI prompt templates
GENERATE_CAPTION_PROMPT = """Analyze the provided image and identify the captions that are part of the meme's intended dialogue or message. You should not have to describe the meme format/image as this is a OCR tool. Only the relevant text on the image has to be processed. Ignore any watermarks, credits, or unrelated text. Do not duplicate any captions. If there is no text, return an empty list of reading_order. Determine the natural reading order as a human would perceive it. Then, output the captions in JSON format with an ordered list, as follows:
        {
  "reading_order": [
    "First caption",
    "Second caption",
  ]
}
Ensure the order reflects logical reading patterns based on spatial positioning and dialogue structure."""

GENERATE_TITLE_PROMPT = """
Given the following image:
    
Generate a social media package that includes:
- Title: A funny interesting, maybe exaggerated or controversial, title for the meme (make sure you understand the meme). Add tags at the end. Make sure it is not longer than 80 characters.

Return the result in JSON format:
{
    "title": "...",
}
"""

@dataclass
class Config:
    """Configuration settings for the video generator."""
    subreddits: List[str]
    min_upvotes: int = DEFAULT_MIN_UPVOTES
    auto_mode: bool = False
    upload: bool = False
    voice: str = DEFAULT_VOICE
    rate: str = DEFAULT_RATE
    pitch: str = DEFAULT_PITCH
    output_dir: str = "redditmeme2video/output"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            subreddits=os.environ.get("SUBREDDITS", "memes,HistoryMemes").split(","),
            min_upvotes=int(os.environ.get("MIN_UPVOTES", DEFAULT_MIN_UPVOTES)),
            auto_mode=os.environ.get("AUTO_MODE", "0") == "1",
            upload=os.environ.get("UPLOAD", "0") == "1",
            voice=os.environ.get("TTS_VOICE", DEFAULT_VOICE),
            rate=os.environ.get("TTS_RATE", DEFAULT_RATE),
            pitch=os.environ.get("TTS_PITCH", DEFAULT_PITCH),
            output_dir=os.environ.get("OUTPUT_DIR", "redditmeme2video/output"),
        )


class AIClient:
    """Client for AI-based text and image analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the AI client with API key."""
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required either as argument or environment variable")
        
        self.client = Groq(api_key=self.api_key)
        self._cache = {}  # Simple in-memory cache
    
    @lru_cache(maxsize=32)
    def get_text_completion(self, prompt: str, model: str = "qwen-2.5-32b") -> str:
        """Get text completion from AI model with caching."""
        logger.info("Sending text completion request to Groq API")
        
        # Check cache
        cache_key = f"{model}:{prompt}"
        if cache_key in self._cache:
            logger.info("Using cached response")
            return self._cache[cache_key]
            
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
            )
            logger.info("Received response from Groq API")
            result = chat_completion.choices[0].message.content
            
            # Cache result
            self._cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Error in API call: {str(e)}")
            
            if "Please try again in " in str(e):
                wait_info = str(e).split("Please try again in ")[1].split(". ")[0].split("m")
                wait_time = float(wait_info[0]) * 60 + float(wait_info[1].split("s")[0])
                logger.info(f"Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
                return self.get_text_completion(prompt, model)
            raise
    
    def get_image_analysis(self, prompt: str, image_url: str, 
                         model: str = "llama-3.2-11b-vision-preview") -> Dict[str, Any]:
        """Analyze image using AI vision model with caching."""
        # Check cache
        cache_key = f"{model}:{prompt}:{image_url}"
        if cache_key in self._cache:
            logger.info("Using cached image analysis")
            return self._cache[cache_key]
            
        logger.info("Sending image analysis request to Groq API")
        try:
            image_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt + " You have to return the result in json format.",
                            },
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                model=model,
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            logger.info("Received response from Groq API")
            result = json.loads(image_completion.choices[0].message.content)
            
            # Cache result
            self._cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Error in API call: {str(e)}")
            
            if "Please try again in " in str(e):
                wait_info = str(e).split("Please try again in ")[1].split(". ")[0].split("m")
                wait_time = float(wait_info[0]) * 60 + float(wait_info[1].split("s")[0])
                logger.info(f"Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
                return self.get_image_analysis(prompt, image_url, model)
            raise


class RedditClient:
    """Client for fetching posts from Reddit."""
    
    _session = None
    
    @classmethod
    def get_session(cls) -> requests.Session:
        """Get or create a reusable HTTP session."""
        if cls._session is None:
            cls._session = requests.Session()
            cls._session.headers.update({
                "User-Agent": "Chrome/133.0.6943.98",
            })
        return cls._session
    
    @staticmethod
    def get_posts(subreddit: str = "memes", post_type: str = "hot", time_frame: str = "day") -> Dict[str, Any]:
        """Fetch posts from Reddit using a persistent session."""
        url = f"https://www.reddit.com/r/{subreddit}/{post_type}/.json"
        
        if post_type == "top":
            url += f"?t={time_frame}"
        
        logger.info(f"Fetching posts from r/{subreddit} ({post_type}, {time_frame})")
        session = RedditClient.get_session()
        response = session.get(url)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def filter_meme_urls(posts: Dict[str, Any], min_upvotes: int) -> List[Tuple[str, str]]:
        """Extract meme URLs and authors from Reddit posts."""
        meme_urls = []
        for post in posts["data"]["children"]:
            url = post["data"].get("url_overridden_by_dest")
            if url and url.endswith((".jpeg", ".png")) and post["data"]["ups"] > min_upvotes:
                meme_urls.append((url, post["data"]["author"]))
        
        random.shuffle(meme_urls)
        logger.info(f"Found {len(meme_urls)} valid meme URLs")
        return meme_urls


class MediaProcessor:
    """Process media files for video creation."""
    
    def __init__(self, config: Config):
        """Initialize media processor."""
        self.config = config
        self.image_dir = Path("redditmeme2video/images")
        self.audio_dir = Path("redditmeme2video/audio")
        self.image_dir.mkdir(exist_ok=True, parents=True)
        self.audio_dir.mkdir(exist_ok=True, parents=True)
        self._session = requests.Session()  # Reuse session for downloads
        self._download_lock = threading.Lock()  # Lock for thread-safe downloads
        self._audio_lock = threading.Lock()  # Lock for thread-safe audio generation
        
    def download_image(self, url: str, idx: int) -> Optional[str]:
        """Download image from URL and save locally."""
        filename = self.image_dir / f"meme_{idx}.png"
        
        # If file exists, skip download
        if filename.exists():
            logger.info(f"Using cached image for {idx}")
            return str(filename)
            
        logger.info(f"Downloading image {idx} from {url}")
        with self._download_lock:  # Thread safety for downloading
            response = self._session.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                return str(filename)
        logger.error(f"Failed to download image: {response.status_code}")
        return None
    
    def generate_audio(self, text: str, idx: int) -> str:
        """Generate audio from text using edge_tts."""
        output_file = self.audio_dir / f"audio_{idx}.mp3"
        
        # If file exists, skip generation
        if output_file.exists():
            logger.info(f"Using cached audio for {idx}")
            return str(output_file)
            
        logger.info(f"Generating audio for text: {text[:30]}...")
        with self._audio_lock:  # Thread safety for audio generation
            tts = edge_tts.Communicate(
                text, self.config.voice,
                rate=self.config.rate,
                pitch=self.config.pitch
            )
            tts.save_sync(str(output_file))
        
        return str(output_file)
    
    def clean_temp_files(self):
        """Clean up temporary files created during processing."""
        logger.info("Cleaning up temporary files")
        try:
            for folder in ['rfm', 'rfm/music', 'redditmeme2video/audio', 'redditmeme2video/images']:
                if os.path.exists(folder):
                    for file_path in Path(folder).glob('*.mp3'):
                        try:
                            file_path.unlink()
                        except PermissionError:
                            logger.warning(f"Could not delete {file_path}, file in use")
                    for file_path in Path(folder).glob('*.png'):
                        try:
                            file_path.unlink()
                        except PermissionError:
                            logger.warning(f"Could not delete {file_path}, file in use")
        except Exception as e:
            logger.error(f"Error cleaning temporary files: {e}")


class VideoGenerator:
    """Generate videos from memes."""
    
    def __init__(self, config: Config, ai_client: AIClient, media_processor: MediaProcessor):
        """Initialize video generator."""
        self.config = config
        self.ai_client = ai_client
        self.media_processor = media_processor
        self.threads = []
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
    
    def collect_captions(self, subreddit: str, amount: int = 3, 
                       post_type: str = "hot", retry: bool = False) -> Tuple[List[Tuple[str, str]], List[List[str]]]:
        """Collect meme URLs and captions for subreddit."""
        try:
            posts = RedditClient.get_posts(
                subreddit=subreddit, 
                post_type=post_type, 
                time_frame="week" if retry else "day"
            )
            meme_urls = RedditClient.filter_meme_urls(posts, self.config.min_upvotes)
            
            if len(meme_urls) < amount:
                logger.warning(f"Not enough memes for {subreddit}")
                if retry:
                    raise ValueError(f"Could not find enough memes for {subreddit}")
                logger.info("Retrying with weekly posts")
                return self.collect_captions(subreddit, amount, "top", True)
                
            # Let user select memes for this video if not in auto mode
            while True:
                selected_meme_urls = random.sample(meme_urls, amount)
                print(f"\nCaptions will be generated for the following URLs in r/{subreddit}:")
                for url, author in selected_meme_urls:
                    print(f"    - {url}")
                
                if self.config.auto_mode or input("Proceed with these memes? (y/n): ").lower() == "y":
                    meme_urls = selected_meme_urls
                    break
            
            # Process captions in parallel using thread pool
            def process_meme(idx_url):
                idx, (meme_url, author) = idx_url
                image_reply = self.ai_client.get_image_analysis(GENERATE_CAPTION_PROMPT, meme_url)
                captions = image_reply["reading_order"]
                
                # Allow user to check captions if not in auto mode
                if not self.config.auto_mode:
                    print(f"\nImage URL: {meme_url}")
                    print("Received captions:")
                    for i, caption in enumerate(captions):
                        print(f"    {i}: {caption}")
                        
                    while True:
                        entry = input("For this image, enter index to remove or 'e' to edit (press Enter if ok): ")
                        if entry == "":
                            break
                        elif entry == "e":
                            _index = int(input("Enter the index to edit: "))
                            _caption = input("Enter the new caption: ")
                            captions[_index] = _caption
                        else:
                            try:
                                captions.pop(int(entry))
                            except (IndexError, ValueError):
                                print("Invalid index to remove")
                
                return idx, captions
            
            # Use ThreadPoolExecutor for parallel processing
            if self.config.auto_mode:  # Only parallelize in auto mode
                futures = []
                for idx, meme_data in enumerate(meme_urls):
                    futures.append(self.thread_pool.submit(process_meme, (idx, meme_data)))
                
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
                results.sort(key=lambda x: x[0])  # Sort by original index
                captions_list = [captions for _, captions in results]
            else:
                # Process serially if user interaction is needed
                captions_list = []
                for idx, meme_data in enumerate(meme_urls):
                    _, captions = process_meme((idx, meme_data))
                    captions_list.append(captions)
                
            return meme_urls, captions_list
            
        except Exception as e:
            logger.error(f"Error collecting captions: {e}")
            raise
            
    def generate_clip(self, image_url: str, idx: int, captions: Optional[List[str]] = None) -> VideoFileClip:
        """Generate a video clip from a meme image with captions."""
        logger.info(f"Generating clip {idx} from {image_url}")
        
        # Get captions if not provided
        if captions is None:
            image_reply = self.ai_client.get_image_analysis(GENERATE_CAPTION_PROMPT, image_url)
            captions = image_reply["reading_order"]
            
        # Process captions: trim punctuation
        for i, caption in enumerate(captions):
            caption = caption.strip()
            if caption and caption[-1] in [".", ",", "!", "?"]:
                caption = caption[:-1]
            captions[i] = caption
            
        # Generate audio from captions
        audio_path = self.media_processor.generate_audio(". ".join(captions), idx)
        
        # Download and process image
        meme_path = self.media_processor.download_image(image_url, idx)
        img = Image.open(meme_path)
        
        # Calculate resize ratio for fitting in target dimensions
        img_ratio = min(TARGET_WIDTH * 0.95 / img.width, TARGET_HEIGHT * 0.95 / img.height)
        new_size = (int(img.width * img_ratio), int(img.height * img_ratio))
        img = img.resize(new_size, Image.LANCZOS)
        
        # Create audio clip
        audio_clip = AudioFileClip(audio_path)
        
        # Create meme clip with proper centering
        meme_bg = Image.new("RGBA", (TARGET_WIDTH, TARGET_HEIGHT), (0, 0, 0, 0))
        paste_x = (TARGET_WIDTH - new_size[0]) // 2
        paste_y = (TARGET_HEIGHT - new_size[1]) // 2
        meme_bg.paste(img, (paste_x, paste_y))
        
        # Create video clip with effects
        meme_clip = ImageClip(np.array(meme_bg)).with_duration(audio_clip.duration)
        
        # Apply different effects based on position in sequence
        if idx == 0:
            final_clip = CompositeVideoClip([meme_clip.with_effects([vfx.FadeOut(0.05)])]).with_audio(audio_clip)
        else:
            final_clip = CompositeVideoClip([meme_clip.with_effects([vfx.FadeIn(0.05), vfx.FadeOut(0.05)])]).with_audio(audio_clip)
            
        return final_clip

    def generate_clips_in_parallel(self, meme_urls, pre_captions):
        """Generate clips in parallel using thread pool."""
        futures = []
        for idx, ((meme_url, _), captions) in enumerate(zip(meme_urls, pre_captions)):
            futures.append(
                self.thread_pool.submit(self.generate_clip, meme_url, idx, captions)
            )
        
        # Wait for all to complete
        clips = []
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            clip = future.result()
            # Add crossfade for last clip
            if idx == len(meme_urls) - 1:
                clip = clip.with_effects([vfx.CrossFadeOut(0.5)])
            clips.append((idx, clip))
        
        # Sort by original index
        clips.sort(key=lambda x: x[0])
        return [clip for _, clip in clips]
        
    def generate_video(self, subreddit: str, amount: int = 3, post_type: str = "hot",
                      retry: bool = False, add_comments: bool = False,
                      output_location: Optional[str] = None,
                      pre_meme_urls: Optional[List[Tuple[str, str]]] = None, 
                      pre_captions: Optional[List[List[str]]] = None) -> Dict[str, Dict[str, str]]:
        """Generate a complete video from Reddit memes."""
        logger.info(f"Generating video for r/{subreddit} with {amount} memes")
        
        # Use a thread-local directory to avoid conflicts
        thread_id = threading.get_ident()
        output_location = output_location or f"{self.config.output_dir}/{subreddit}_{thread_id}"
        os.makedirs(output_location, exist_ok=True)
        
        # Initialize metadata
        comment = "Credits to"
        description = f"Compilation of {amount} {subreddit} memes. Enjoy!"
        
        # Get memes and captions if not provided
        if pre_meme_urls is None or pre_captions is None:
            pre_meme_urls, pre_captions = self.collect_captions(subreddit, amount, post_type, retry)
            
        # Generate title using AI analysis of first meme
        title_data = self.ai_client.get_image_analysis(GENERATE_TITLE_PROMPT, pre_meme_urls[0][0])
        title = title_data["title"] + f' | r/{subreddit}'
        logger.info(f"Generated title: {title}")
        
        # Generate individual clips for each meme - now in parallel when possible
        if self.config.auto_mode:
            clips = self.generate_clips_in_parallel(pre_meme_urls, pre_captions)
        else:
            # Generate clips serially if not in auto mode (for better user experience)
            clips = []
            for idx, ((meme_url, _), captions) in enumerate(zip(pre_meme_urls, pre_captions)):
                subclip = self.generate_clip(meme_url, idx, captions=captions)
                if idx == len(pre_meme_urls) - 1:
                    subclip = subclip.with_effects([vfx.CrossFadeOut(0.5)])
                clips.append(subclip)
        
        # Build credits
        for meme_url, author in pre_meme_urls:
            comment += f" u/{author},"
        comment = comment[:-1] + " for the memes!"
        
        # Add outro
        # outro = VideoFileClip("redditmeme2video/outro.mp4").with_fps(FPS)
        # clips.append(outro)
        
        # Concatenate all clips
        final_video = concatenate_videoclips(clips)
        
        # Add background gameplay video
        gameplay_dir = "redditmeme2video/gameplay"
        gameplay_files = [f for f in os.listdir(gameplay_dir) if f.endswith((".mp4", ".webm"))]
        if not gameplay_files:
            raise FileNotFoundError("No gameplay videos found in directory")
            
        # Select random background video
        background_path = os.path.join(gameplay_dir, random.choice(gameplay_files))
        background_source = VideoFileClip(background_path).with_fps(FPS)
        logger.info(f"Using background video: {background_path}")
        
        # Extract segment of background video
        start = random.randrange(0, int(background_source.duration - final_video.duration))
        bg_clip = background_source.subclipped(start, start + final_video.duration)
        
        # Resize and crop background to match target dimensions - optimize by doing these in one step
        bg_clip = bg_clip.resized(height=TARGET_HEIGHT).cropped(
            width=TARGET_WIDTH,
            height=TARGET_HEIGHT,
            x_center=bg_clip.resized(height=TARGET_HEIGHT).size[0] // 2,
            y_center=TARGET_HEIGHT // 2,
        )
        
        # Get background music
        bg_audio_path = get_music_path()
        bg_audio_clip = (
            AudioFileClip(bg_audio_path)
            .subclipped(0, final_video.duration)
            .with_volume_scaled(0.05)
        )
        
        # Create transparent layer for audio
        transparent_background = (
            ColorClip((TARGET_WIDTH, TARGET_HEIGHT), color=(0, 0, 0, 0))
            .with_duration(final_video.duration)
            .with_audio(bg_audio_clip)
        )
        
        # Composite final video with background and audio
        final_video = CompositeVideoClip(
            [
                bg_clip.with_effects([vfx.MultiplyColor(1)]),
                final_video,
                transparent_background,
            ]
        )
        
        # Write video to file
        output_path = f"{output_location}/final_video.mp4"
        
        # Use threading for writing the video file
        write_thread = threading.Thread(
            target=self._write_video_to_file,
            args=(final_video, output_path, output_location),
            daemon=False
        )
        self.threads.append(write_thread)
        write_thread.start()
        
        # Return metadata
        if add_comments:
            return {
                output_path: {
                    "title": title,
                    "description": description,
                    "pinned_comment": comment,
                }
            }
        return {
            output_path: {
                "title": title,
                "description": description,
            }
        }
    
    def _write_video_to_file(self, video_clip: VideoFileClip, output_path: str, output_location: str) -> None:
        """Write video to file and clean up afterward."""
        logger.info(f"Writing video to {output_path}")
        try:
            video_clip.write_videofile(
                output_path,
                codec="libx264",
                fps=FPS,
                threads=os.cpu_count() or 4,  # Use all available CPU cores for encoding
                preset="fast",  # Faster encoding with slight quality drop
                temp_audiofile_path=output_location,
            )
            logger.info("Video written successfully")
        except Exception as e:
            logger.error(f"Error writing video: {e}")
        finally:
            # Clean up temporary files after writing is complete
            self.media_processor.clean_temp_files()
    
    def wait_for_completion(self):
        """Wait for all video generation threads to complete."""
        logger.info(f"Waiting for {len(self.threads)} video generation threads to complete")
        for thread in self.threads:
            thread.join()
        logger.info("All video generation complete")
        
        # Clean up thread pool
        self.thread_pool.shutdown()


def main():
    """Main entry point."""
    # Load configuration
    config = Config(
        subreddits=["HistoryMemes"],
        min_upvotes=1000,
        auto_mode=False
    )
    
    # Initialize components
    ai_client = AIClient()
    media_processor = MediaProcessor(config)
    video_generator = VideoGenerator(config, ai_client, media_processor)
    
    # Pre-generate captions for all videos
    precomputed = {}
    for idx, subreddit in enumerate(config.subreddits):
        logger.info(f"--- Caption collection for r/{subreddit} ---")
        meme_urls, pre_captions = video_generator.collect_captions(subreddit, amount=3, post_type="top")
        precomputed[idx] = (meme_urls, pre_captions)
        
    # Generate videos with precomputed captions
    for idx, subreddit in enumerate(config.subreddits):
        meme_urls, pre_captions = precomputed[idx]
        video_data = video_generator.generate_video(
            subreddit, 
            amount=3, 
            post_type="top", 
            add_comments=True,
            output_location=f"{config.output_dir}/{idx}",
            pre_meme_urls=meme_urls, 
            pre_captions=pre_captions
        )
        
        # Save metadata
        os.makedirs(f"{config.output_dir}/{idx}", exist_ok=True)
        with open(f"{config.output_dir}/{idx}/data.json", "w") as f:
            json.dump(video_data, f)
            
        logger.info(f"Video metadata: {video_data}")
    
    # Wait for all videos to complete writing
    video_generator.wait_for_completion()


if __name__ == "__main__":
    main()


