"""
Reddit Meme to Video Generator GUI

A graphical user interface for the redditmeme2video tool, providing easy configuration 
and video generation features.
"""

import os
import sys
import json
import time
import random
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import webbrowser
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from PIL import Image, ImageTk
import requests
import asyncio
from io import BytesIO
import edge_tts
import concurrent.futures
from functools import lru_cache

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from redditmeme2video module - keep only what we need
from redditmeme2video import (
    AIClient,
    Config,
    MediaProcessor,
    RedditClient,
    strip_ssml_tags,
)

# Import VideoGenerator but rename it to avoid conflicts with our GUI-specific version
from redditmeme2video import VideoGenerator as CoreVideoGenerator

# Import SSML editor
from ssml_editor import edit_ssml_captions

# Default configuration values
DEFAULT_CONFIG = {
    "subreddits": ["memes", "dankmemes", "HistoryMemes", "wholesomememes"],
    "min_upvotes": 3000,
    "auto_mode": False,
    "upload": False,
    "use_elevenlabs": True,  # Default to ElevenLabs
    "elevenlabs_voice_id": "UgBBYS2sOqTuMpoF3BR0",  # Default to Mark voice
    "elevenlabs_model": "eleven_flash_v2_5",
    "elevenlabs_stability": 0.34,
    "elevenlabs_similarity_boost": 0.75,
    "elevenlabs_style": 0.34,
    "elevenlabs_speaker_boost": True,
    "elevenlabs_speed": 1.05,
    "edge_tts_voice": "en-AU-WilliamNeural",
    "edge_tts_rate": "+15%",
    "edge_tts_volume": "+5%",
    "edge_tts_pitch": "+30Hz",
    "output_dir": "redditmeme2video/output",
    "dark_mode": True,
    "animation_level": "high",
    "use_background_music": True,
    "memes_per_video": 3,
    "post_type": "hot",
}

# Available TTS voices (popular options)
TTS_VOICES = [
    "en-AU-WilliamNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-GB-RyanNeural",
    "en-CA-ClaraNeural",
    "en-IE-ConnorNeural",
]

# Popular ElevenLabs voices
ELEVENLABS_VOICES = [
    ("UgBBYS2sOqTuMpoF3BR0", "Mark"),
    ("rMccYqkTp7l0Bj2Xxz3W", "Daniel"),
    ("X1DzLZQVKrXSJOiIrDrN", "James"),
    ("C7r8d6DTHYHXwTstJ9Vh", "Charlotte"),
    ("oWAxZDx7w5VEj9dYyzcH", "Grace"),
]

# Popular subreddit suggestions
POPULAR_SUBREDDITS = [
    "memes",
    "dankmemes",
    "HistoryMemes",
    "wholesomememes",
    "ProgrammerHumor",
    "me_irl",
    "funny",
    "MemeEconomy",
    "perfectlycutscreams",
    "hmm",
    "HolUp",
    "starterpacks",
    "technicallythetruth",
    "ShitPostCrusaders",
    "marvelmemes",
    "PrequelMemes",
    "lotrmemes",
]

# Global thread pool for image loading to avoid creating excess threads
IMAGE_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Create a cache for downloaded images to avoid redownloading
# This uses the URL as key and stores the PIL Image
@lru_cache(maxsize=100)
def get_image_from_url(url):
    """Download an image from URL and cache it"""
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        # Check file size before processing to avoid memory issues
        content_length = len(response.content)
        if content_length > 10_000_000:  # 10MB limit
            raise ValueError(f"Image too large ({content_length / 1_000_000:.1f}MB)")
        
        img = Image.open(BytesIO(response.content))
        
        # Validate image dimensions to avoid memory issues
        if img.width * img.height > 25_000_000:  # ~25 megapixels limit
            raise ValueError(f"Image dimensions too large: {img.width}x{img.height}")
            
        return img
    else:
        raise ValueError(f"Failed to download image: Status {response.status_code}")


# GUI-optimized video generator that avoids CLI interactions
class GUIVideoGenerator:
    """GUI-specific video generator that prevents CLI interactions"""

    def __init__(
        self, config: Config, ai_client: AIClient, media_processor: MediaProcessor
    ):
        """Initialize the GUI-optimized video generator"""
        self.config = config
        self.ai_client = ai_client
        self.media_processor = media_processor

        # Create the core video generator but we'll override its interactive methods
        self.core_generator = CoreVideoGenerator(config, ai_client, media_processor)

        # Thread pool for parallel operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count() or 4
        )
        self.threads = []

    def collect_memes(
        self, subreddit: str, amount: int = 3, post_type: str = "hot"
    ) -> List[Tuple[str, str, str]]:
        """
        Collect memes without generating captions to save API calls
        """
        try:
            # Get posts
            posts = RedditClient.get_posts(
                subreddit=subreddit,
                post_type=post_type,
                time_frame="week",  # Default to week for more selection
            )
            meme_urls = RedditClient.filter_meme_urls(posts, self.config.min_upvotes)

            if len(meme_urls) < amount:
                # Try with top posts of the week for more options
                posts = RedditClient.get_posts(
                    subreddit=subreddit, post_type="top", time_frame="week"
                )
                meme_urls = RedditClient.filter_meme_urls(
                    posts, self.config.min_upvotes
                )

                if len(meme_urls) < amount:
                    # Last resort: lower minimum upvotes
                    reduced_upvotes = max(500, self.config.min_upvotes // 2)
                    meme_urls = RedditClient.filter_meme_urls(posts, reduced_upvotes)

                    if len(meme_urls) < amount:
                        raise ValueError(
                            f"Not enough memes found for r/{subreddit} even with reduced criteria"
                        )

            # Get up to 10 memes or all available, whichever is more
            max_display = max(10, amount * 2)
            selected_meme_urls = random.sample(
                meme_urls, min(max_display, len(meme_urls))
            )

            return selected_meme_urls

        except Exception as e:
            raise RuntimeError(f"Error collecting memes: {str(e)}") from e

    def generate_captions(self, memes: List[Tuple[str, str, str]]) -> List[List[str]]:
        """
        Generate captions for selected memes - this is the API-heavy part
        """
        try:
            # Process memes in parallel for faster analysis
            def process_meme(idx_url):
                idx, (meme_url, author, title) = idx_url
                from redditmeme2video import GENERATE_CAPTION_PROMPT

                formatted_prompt = GENERATE_CAPTION_PROMPT.format(post_title=title)
                image_reply = self.ai_client.get_image_analysis(
                    formatted_prompt, meme_url
                )
                captions = image_reply["reading_order"]
                return idx, captions

            # Process in parallel with thread pool
            futures = []
            for idx, meme_data in enumerate(memes):
                futures.append(self.thread_pool.submit(process_meme, (idx, meme_data)))

            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
            results.sort(key=lambda x: x[0])  # Sort by original index
            captions_list = [captions for _, captions in results]

            return captions_list

        except Exception as e:
            raise RuntimeError(f"Error generating captions: {str(e)}") from e

    def collect_captions(
        self,
        subreddit: str,
        amount: int = 3,
        post_type: str = "hot",
        auto_mode: bool = None,
    ) -> Tuple[List[Tuple[str, str, str]], List[List[str]]]:
        """
        Legacy method that combines collect_memes and generate_captions for backward compatibility
        """
        # Force auto mode to ensure no CLI prompts
        is_auto_mode = self.config.auto_mode if auto_mode is None else auto_mode

        # First collect memes
        selected_meme_urls = self.collect_memes(subreddit, amount, post_type)

        # Then generate captions
        captions_list = self.generate_captions(selected_meme_urls)

        return selected_meme_urls, captions_list

    def generate_video(
        self,
        subreddit: str,
        amount: int = 3,
        post_type: str = "hot",
        output_location: str = None,
        pre_meme_urls: Optional[List[Tuple[str, str, str]]] = None,
        pre_captions: Optional[List[List[str]]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        GUI-optimized video generation that avoids CLI interactions
        """
        # Generate a unique output location if not provided
        if output_location is None:
            thread_id = threading.get_ident()
            output_location = f"{self.config.output_dir}/{subreddit}_{thread_id}"

        os.makedirs(output_location, exist_ok=True)

        # Get memes and captions if not provided
        if pre_meme_urls is None or pre_captions is None:
            # Force auto mode to ensure no CLI prompts
            pre_meme_urls, pre_captions = self.collect_captions(
                subreddit, amount, post_type, auto_mode=True
            )

        # Use the core generator's implementation but with our data
        metadata = self.core_generator.generate_video(
            subreddit=subreddit,
            amount=amount,
            post_type=post_type,
            add_comments=True,
            output_location=output_location,
            pre_meme_urls=pre_meme_urls,
            pre_captions=pre_captions,
        )

        return metadata

    def wait_for_completion(self):
        """Wait for all video generation threads to complete."""
        # Forward to the core generator
        self.core_generator.wait_for_completion()

        # Also shut down our own thread pool
        self.thread_pool.shutdown()


# The rest of your GUI code remains the same
# ...existing code...


class ScrollableFrame(ttk.Frame):
    """A scrollable frame widget"""

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Configure canvas for mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class VideoQueueItem:
    """Represents a video in the generation queue"""

    def __init__(
        self,
        subreddit: str,
        memes_count: int,
        post_type: str,
        config: Dict[str, Any],
        output_dir: str,
    ):
        self.subreddit = subreddit
        self.memes_count = memes_count
        self.post_type = post_type
        self.config = config
        self.output_dir = output_dir
        self.status = "queued"  # queued, processing, completed, failed
        self.progress = 0
        self.output_path = None
        self.metadata = {}
        self.meme_urls = None
        self.captions = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "subreddit": self.subreddit,
            "memes_count": self.memes_count,
            "post_type": self.post_type,
            "config": self.config,
            "output_dir": self.output_dir,
            "status": self.status,
            "progress": self.progress,
            "output_path": self.output_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoQueueItem":
        """Create from dictionary after deserialization"""
        item = cls(
            data["subreddit"],
            data["memes_count"],
            data["post_type"],
            data["config"],
            data["output_dir"],
        )
        item.status = data["status"]
        item.progress = data["progress"]
        item.output_path = data["output_path"]
        item.metadata = data["metadata"]
        return item


class MemePreviewPanel(ttk.Frame):
    """Panel for previewing memes and captions"""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

        # Current meme data
        self.current_meme_url = None
        self.current_caption = None
        self.current_title = None
        self.current_author = None

        # Initialize TTS for preview
        self.tts_voice = DEFAULT_CONFIG["edge_tts_voice"]
        self.tts_rate = DEFAULT_CONFIG["edge_tts_rate"]
        self.tts_volume = DEFAULT_CONFIG["edge_tts_volume"]
        self.tts_pitch = DEFAULT_CONFIG["edge_tts_pitch"]

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        """Create all UI widgets for the preview panel"""
        # Title bar
        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.title_label = ttk.Label(
            self.title_frame, text="Meme Preview", font=("TkDefaultFont", 14, "bold")
        )
        self.title_label.pack(side=tk.LEFT)

        # Navigation and caption editing buttons at the top for better visibility
        self.nav_frame = ttk.Frame(self)
        self.nav_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Navigation panel (left side)
        nav_controls = ttk.Frame(self.nav_frame)
        nav_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.prev_btn = ttk.Button(
            nav_controls,
            text="← Previous",
            command=self._handle_prev_meme,  # Modified to save caption before navigation
            state=tk.DISABLED,
            width=10
        )
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.meme_counter = ttk.Label(
            nav_controls, text="0/0", width=8
        )
        self.meme_counter.pack(side=tk.LEFT, padx=5)

        self.next_btn = ttk.Button(
            nav_controls, 
            text="Next →",
            command=self._handle_next_meme,  # Modified to save caption before navigation
            state=tk.DISABLED,
            width=10
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)

        # Caption editing buttons (right side)
        caption_controls = ttk.Frame(self.nav_frame)
        caption_controls.pack(side=tk.RIGHT, fill=tk.X)

        self.save_caption_btn = ttk.Button(
            caption_controls, text="💾 Save Caption", command=self.save_caption, width=15
        )
        self.save_caption_btn.pack(side=tk.LEFT, padx=5)
        
        self.edit_ssml_btn = ttk.Button(
            caption_controls, text="✏️ Edit SSML", command=self.open_ssml_editor, width=15
        )
        self.edit_ssml_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_caption_btn = ttk.Button(
            caption_controls, text="🔊 Preview Voice", command=self.play_caption, width=15
        )
        self.play_caption_btn.pack(side=tk.LEFT, padx=5)

        # Create a PanedWindow to allow resizing between image and caption areas
        self.content_paned = ttk.PanedWindow(self, orient=tk.VERTICAL)
        self.content_paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Preview image area
        self.image_frame = ttk.LabelFrame(self.content_paned, text="Image")
        self.content_paned.add(self.image_frame, weight=2)  # Give image more initial weight

        self.canvas = tk.Canvas(self.image_frame, bg="#f0f0f0", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Placeholder image
        placeholder_text = "No meme selected for preview"
        self.canvas.create_text(200, 150, text=placeholder_text, fill="gray")

        # Caption preview area - MODIFIED to make captions resizable
        self.caption_frame = ttk.LabelFrame(self.content_paned, text="Caption Editor - Edit Text Below")
        self.content_paned.add(self.caption_frame, weight=1)  # Give caption area initial weight

        # Create a container frame for the text widget and scrollbar
        self.caption_container = ttk.Frame(self.caption_frame)
        self.caption_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.caption_container.columnconfigure(0, weight=1)
        self.caption_container.rowconfigure(0, weight=1)

        # Create caption text widget with scrollbar attached directly to the container
        self.caption_text = scrolledtext.ScrolledText(
            self.caption_container,
            wrap=tk.WORD,  # Word wrapping
            undo=True      # Enable undo/redo
        )
        self.caption_text.grid(row=0, column=0, sticky="nsew")  # Make it fill the grid cell completely
        self.caption_text.insert(tk.END, "No captions to display")
        
        # Caption hint text
        hint_label = ttk.Label(
            self.caption_frame, 
            text="Edit captions here and click 'Save Caption' when done",
            font=("TkDefaultFont", 8), 
            foreground="grey"
        )
        hint_label.pack(side=tk.BOTTOM, anchor=tk.W, padx=5, pady=(0, 5))
        
        # Metadata area
        self.meta_frame = ttk.LabelFrame(self, text="Metadata")
        self.meta_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.meta_grid = ttk.Frame(self.meta_frame)
        self.meta_grid.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.meta_grid, text="Title:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.title_value = ttk.Label(self.meta_grid, text="N/A")
        self.title_value.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(self.meta_grid, text="Author:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.author_value = ttk.Label(self.meta_grid, text="N/A")
        self.author_value.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(self.meta_grid, text="URL:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.url_value = ttk.Label(self.meta_grid, text="N/A", cursor="hand2")
        self.url_value.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        self.url_value.bind("<Button-1>", self.open_meme_url)

        # Action buttons
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.open_reddit_btn = ttk.Button(
            self.button_frame, text="Open on Reddit", command=self.open_on_reddit
        )
        self.open_reddit_btn.pack(side=tk.LEFT, padx=5)

        # Set initial sash position after all widgets are created
        self.after(100, lambda: self.content_paned.sashpos(0, 400))

    def _handle_prev_meme(self):
        """Save caption before navigating to previous meme"""
        # Save current caption first without showing message
        self.save_caption(show_message=False)
        # Then navigate
        if hasattr(self.parent, '_prev_meme'):
            self.parent._prev_meme()

    def _handle_next_meme(self):
        """Save caption before navigating to next meme"""
        # Save current caption first without showing message
        self.save_caption(show_message=False)
        # Then navigate
        if hasattr(self.parent, '_next_meme'):
            self.parent._next_meme()

    def set_meme(self, meme_url: str, captions: List[str], title: str, author: str):
        """Set the current meme for preview"""
        self.current_meme_url = meme_url
        self.current_caption = captions
        self.current_title = title
        self.current_author = author

        # Update the UI with the new meme
        self.update_preview()

    def update_preview(self):
        """Update the preview with current meme data"""
        if not self.current_meme_url:
            return

        # Update metadata
        self.title_value.config(text=self.current_title or "N/A")
        self.author_value.config(
            text=f"u/{self.current_author}" if self.current_author else "N/A"
        )
        self.url_value.config(text=self.current_meme_url)

        # Update captions - MODIFIED: now keeping captions editable
        self.caption_text.config(state=tk.NORMAL)
        self.caption_text.delete(1.0, tk.END)

        if self.current_caption:
            formatted_captions = []
            for i, caption in enumerate(self.current_caption):
                # Strip SSML tags for display
                plain_text = strip_ssml_tags(caption)
                formatted_captions.append(f"{i+1}. {plain_text}")

            self.caption_text.insert(tk.END, "\n\n".join(formatted_captions))
        else:
            self.caption_text.insert(tk.END, "No captions to display")
        
        # Make it clearer that captions are editable
        self.caption_text.config(background="#f8f8f8")  # Light gray background to indicate editability
        
        # Add a hint at the top of the caption area
        if not hasattr(self, "edit_hint_added"):
            self.caption_text.insert("1.0", "--- EDIT CAPTIONS HERE (Save when done) ---\n\n")
            self.edit_hint_added = True
        
        # Download and display the image using the global thread pool
        IMAGE_THREAD_POOL.submit(self._load_image)
    
    def _load_image(self):
        """Load image in a background thread using the cached image function"""
        try:
            if not self.current_meme_url:
                return
                
            # Use the cached image loader
            image = get_image_from_url(self.current_meme_url)

            # Calculate dimensions to fit in canvas while maintaining aspect ratio
            canvas_width = self.canvas.winfo_width() or 400
            canvas_height = self.canvas.winfo_height() or 300

            # Initial scale to fit either width or height
            scale = min(canvas_width / image.width, canvas_height / image.height) * 0.9
            new_width = int(image.width * scale)
            new_height = int(image.height * scale)

            # Resize image
            image = image.resize((new_width, new_height), Image.LANCZOS)

            # Convert to PhotoImage for display
            photo_image = ImageTk.PhotoImage(image)

            # Update UI in main thread
            self.after(
                0, lambda: self._update_image(photo_image, new_width, new_height)
            )
        except Exception as e:
            # Update UI in main thread to show error
            error_msg = str(e)
            self.after(0, lambda: self._show_image_error(error_msg))

    def save_caption(self, show_message=True):
        """Save edited caption text back to the data model"""
        if not self.current_caption:
            if show_message:
                messagebox.showinfo("No Caption", "No captions available to edit.")
            return
            
        try:
            # Get the edited text
            edited_text = self.caption_text.get("1.0", tk.END).strip()
            
            # Handle the case where the edit hint is present
            if "--- EDIT CAPTIONS HERE" in edited_text:
                # Remove the hint line
                edited_text = "\n".join([line for line in edited_text.split("\n") 
                                       if not line.startswith("---")])
            
            # Split into sections by line/paragraph
            sections = [s.strip() for s in edited_text.split("\n\n") if s.strip()]
            
            # Parse section numbers (if present) and extract clean text
            clean_sections = []
            for section in sections:
                # Remove section numbers like "1. " if present
                if section and section[0].isdigit() and ". " in section[:4]:
                    clean_text = section[section.index(". ") + 2:]
                else:
                    clean_text = section
                clean_sections.append(clean_text)
                
            # Convert to SSML format
            new_captions = []
            for text in clean_sections:
                new_captions.append(f"<speak>{text}</speak>")
            
            # Update the current caption locally
            self.current_caption = new_captions
            
            # Update the captions in parent (MainApplication)
            if hasattr(self.parent, 'update_captions'):
                self.parent.update_captions(new_captions)
                if show_message:  # Only show message if requested
                    messagebox.showinfo("Success", "Captions saved successfully")
            elif show_message:  # Only show message if requested
                messagebox.showinfo("Info", "Captions edited but not saved to video generation")
        except Exception as e:
            if show_message:  # Only show message if requested
                messagebox.showerror("Error", f"Failed to save captions: {str(e)}")

    def open_ssml_editor(self):
        """Open captions in SSML editor"""
        if not self.current_caption:
            messagebox.showinfo("No Captions", "No captions available to edit.")
            return
            
        try:
            # Get current TTS settings from parent application if available
            voice = self.tts_voice
            rate = self.tts_rate
            volume = self.tts_volume
            pitch = self.tts_pitch
            
            # Update settings from parent config panel if available
            if hasattr(self.parent, 'config_panel'):
                voice = self.parent.config_panel.voice_var.get()
                rate = self.parent.config_panel.rate_var.get()
                volume = self.parent.config_panel.volume_var.get()
                pitch = self.parent.config_panel.pitch_var.get()
            
            # Use SSML editor and pass current captions
            edited_captions = edit_ssml_captions(
                self.current_caption,
                voice_id=voice,
                rate=rate,
                volume=volume,
                pitch=pitch
            )
            
            # Update the current caption locally
            self.current_caption = edited_captions
            
            # Update the captions in parent
            if hasattr(self.parent, 'update_captions'):
                self.parent.update_captions(edited_captions)
                
            # Refresh display to show edited captions
            self.update_preview()
            
            messagebox.showinfo("Success", "SSML captions updated successfully")
        except Exception as e:
            messagebox.showerror("SSML Editor Error", f"Error editing captions: {str(e)}")

    def _update_image(self, photo_image, width, height):
        """Update the canvas with the loaded image (called from main thread)"""
        # Clear current canvas contents
        self.canvas.delete("all")

        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width() or 400
        canvas_height = self.canvas.winfo_height() or 300

        # Calculate center position
        x = canvas_width // 2
        y = canvas_height // 2

        # Create image on canvas
        self.canvas.create_image(x, y, image=photo_image, anchor=tk.CENTER)
        self.image_ref = photo_image  # Keep reference to prevent garbage collection

    def _show_image_error(self, error_msg):
        """Show error message when image loading fails"""
        self.canvas.delete("all")
        self.canvas.create_text(
            200, 150, text=f"Error loading image:\n{error_msg}", fill="red"
        )

    def play_caption(self):
        """Play the caption text with TTS"""
        if not self.current_caption:
            messagebox.showinfo("No Caption", "No caption available to play.")
            return

        # Disable button during playback
        self.play_caption_btn.config(state=tk.DISABLED)
        self.play_caption_btn.config(text="Playing...")

        # Get the full caption text without SSML
        caption_text = ""
        for caption in self.current_caption:
            caption_text += strip_ssml_tags(caption) + ". "

        # Start playback in a background thread
        threading.Thread(
            target=self._play_audio, args=(caption_text,), daemon=True
        ).start()

    def _play_audio(self, text):
        """Play audio in a background thread"""
        try:
            # Create temp file for audio
            import tempfile

            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_file.close()

            # Generate audio file with edge_tts
            async def generate_audio():
                communicate = edge_tts.Communicate(
                    text,
                    self.tts_voice,
                    rate=self.tts_rate,
                    volume=self.tts_volume,
                    pitch=self.tts_pitch,
                )
                await communicate.save(temp_file.name)

            # Run async code
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(generate_audio())

            # Play the audio
            import pygame

            pygame.mixer.init()
            pygame.mixer.music.load(temp_file.name)
            pygame.mixer.music.play()

            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            # Cleanup
            pygame.mixer.quit()
            os.unlink(temp_file.name)

            # Update UI in main thread
            self.after(
                0, lambda: self._playback_complete()
            )

        except Exception as e:
            # Show error in main thread
            self.after(
                0,
                lambda: messagebox.showerror(
                    "Playback Error", f"Error playing caption: {str(e)}"
                ),
            )
            self.after(0, lambda: self._playback_complete())

    def _playback_complete(self):
        """Reset button after playback completes"""
        self.play_caption_btn.config(state=tk.NORMAL)
        self.play_caption_btn.config(text="Play Caption")

    def open_meme_url(self, event=None):
        """Open meme URL in browser"""
        if self.current_meme_url:
            webbrowser.open(self.current_meme_url)

    def open_on_reddit(self):
        """Try to open the Reddit post in a browser"""
        if not self.current_author or not self.current_title:
            messagebox.showinfo("Not Available", "Reddit post link not available.")
            return

        # Search for the post on Reddit
        search_url = f"https://www.reddit.com/search/?q={self.current_title}"
        webbrowser.open(search_url)

    def set_tts_settings(self, voice, rate, volume, pitch):
        """Update TTS settings for caption preview"""
        self.tts_voice = voice
        self.tts_rate = rate
        self.tts_volume = volume
        self.tts_pitch = pitch


class ConfigPanel(ScrollableFrame):
    """Configuration panel for video generation settings"""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

        # Current configuration
        self.config = DEFAULT_CONFIG.copy()

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        """Create all UI widgets for the config panel"""
        main_frame = self.scrollable_frame

        # Title
        self.title_label = ttk.Label(
            main_frame,
            text="Configuration Settings",
            font=("TkDefaultFont", 14, "bold"),
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

        # Subreddit Selection
        ttk.Label(main_frame, text="Subreddit:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )

        self.subreddit_frame = ttk.Frame(main_frame)
        self.subreddit_frame.grid(row=1, column=1, sticky="we", padx=5, pady=5)

        # Combobox with autocomplete for subreddit selection
        self.subreddit_var = tk.StringVar()
        self.subreddit_combo = ttk.Combobox(
            self.subreddit_frame,
            textvariable=self.subreddit_var,
            values=POPULAR_SUBREDDITS,
        )
        self.subreddit_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Add button for multi-subreddit list
        self.add_subreddit_btn = ttk.Button(
            self.subreddit_frame, text="+", width=3, command=self.add_subreddit
        )
        self.add_subreddit_btn.pack(side=tk.LEFT, padx=5)

        # Subreddit list display (for batch mode)
        self.subreddit_list_frame = ttk.Frame(main_frame)
        self.subreddit_list_frame.grid(
            row=2, column=0, columnspan=2, sticky="we", padx=5, pady=5
        )

        self.subreddit_listbox = tk.Listbox(self.subreddit_list_frame, height=5)
        self.subreddit_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.subreddit_scrollbar = ttk.Scrollbar(
            self.subreddit_list_frame,
            orient="vertical",
            command=self.subreddit_listbox.yview,
        )
        self.subreddit_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.subreddit_listbox.config(yscrollcommand=self.subreddit_scrollbar.set)

        # Buttons for list manipulation
        self.list_btn_frame = ttk.Frame(main_frame)
        self.list_btn_frame.grid(
            row=3, column=0, columnspan=2, sticky="we", padx=5, pady=5
        )

        self.remove_subreddit_btn = ttk.Button(
            self.list_btn_frame, text="Remove Selected", command=self.remove_subreddit
        )
        self.remove_subreddit_btn.pack(side=tk.LEFT, padx=5)

        self.clear_subreddits_btn = ttk.Button(
            self.list_btn_frame, text="Clear All", command=self.clear_subreddits
        )
        self.clear_subreddits_btn.pack(side=tk.LEFT, padx=5)

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=4, column=0, columnspan=2, sticky="we", pady=10
        )

        # Basic Settings
        ttk.Label(
            main_frame, text="Basic Settings", font=("TkDefaultFont", 12, "bold")
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        # Memes per video
        ttk.Label(main_frame, text="Memes per video:").grid(
            row=6, column=0, sticky="w", padx=5, pady=5
        )

        self.memes_var = tk.IntVar(value=DEFAULT_CONFIG["memes_per_video"])
        self.memes_spinbox = ttk.Spinbox(
            main_frame, from_=1, to=10, textvariable=self.memes_var, width=5
        )
        self.memes_spinbox.grid(row=6, column=1, sticky="w", padx=5, pady=5)

        # Minimum upvotes
        ttk.Label(main_frame, text="Minimum upvotes:").grid(
            row=7, column=0, sticky="w", padx=5, pady=5
        )

        self.upvotes_var = tk.IntVar(value=DEFAULT_CONFIG["min_upvotes"])
        self.upvotes_spinbox = ttk.Spinbox(
            main_frame,
            from_=100,
            to=100000,
            increment=100,
            textvariable=self.upvotes_var,
            width=10,
        )
        self.upvotes_spinbox.grid(row=7, column=1, sticky="w", padx=5, pady=5)

        # Post type
        ttk.Label(main_frame, text="Post type:").grid(
            row=8, column=0, sticky="w", padx=5, pady=5
        )

        self.post_type_var = tk.StringVar(value=DEFAULT_CONFIG["post_type"])
        self.post_type_combo = ttk.Combobox(
            main_frame,
            textvariable=self.post_type_var,
            values=["hot", "top", "new", "rising"],
            state="readonly",
            width=10,
        )
        self.post_type_combo.grid(row=8, column=1, sticky="w", padx=5, pady=5)

        # Auto mode
        self.auto_mode_var = tk.BooleanVar(value=DEFAULT_CONFIG["auto_mode"])
        self.auto_mode_check = ttk.Checkbutton(
            main_frame,
            text="Auto mode (no manual selection)",
            variable=self.auto_mode_var,
        )
        self.auto_mode_check.grid(
            row=9, column=0, columnspan=2, sticky="w", padx=5, pady=5
        )

        # Background music
        self.bg_music_var = tk.BooleanVar(value=DEFAULT_CONFIG["use_background_music"])
        self.bg_music_check = ttk.Checkbutton(
            main_frame, text="Include background music", variable=self.bg_music_var
        )
        self.bg_music_check.grid(
            row=10, column=0, columnspan=2, sticky="w", padx=5, pady=5
        )

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=11, column=0, columnspan=2, sticky="we", pady=10
        )

        # Voice Settings
        ttk.Label(
            main_frame, text="Voice Settings", font=("TkDefaultFont", 12, "bold")
        ).grid(row=12, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        # TTS Engine Selection
        ttk.Label(main_frame, text="TTS Engine:").grid(
            row=13, column=0, sticky="w", padx=5, pady=5
        )

        self.tts_engine_var = tk.StringVar(value="elevenlabs" if DEFAULT_CONFIG["use_elevenlabs"] else "edge")
        self.tts_engine_combo = ttk.Combobox(
            main_frame,
            textvariable=self.tts_engine_var,
            values=["elevenlabs", "edge"],
            state="readonly"
        )
        self.tts_engine_combo.grid(row=13, column=1, sticky="we", padx=5, pady=5)
        self.tts_engine_combo.bind('<<ComboboxSelected>>', self._on_engine_change)

        # ElevenLabs Settings Frame
        self.elevenlabs_frame = ttk.LabelFrame(main_frame, text="ElevenLabs Settings")
        self.elevenlabs_frame.grid(row=14, column=0, columnspan=2, sticky="we", padx=5, pady=5)

        # ElevenLabs Voice selection
        ttk.Label(self.elevenlabs_frame, text="Voice:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        
        self.elevenlabs_voice_var = tk.StringVar(value=DEFAULT_CONFIG["elevenlabs_voice_id"])
        self.elevenlabs_voice_combo = ttk.Combobox(
            self.elevenlabs_frame,
            textvariable=self.elevenlabs_voice_var,
            values=[f"{name} ({id})" for id, name in ELEVENLABS_VOICES],
            state="readonly"
        )
        self.elevenlabs_voice_combo.grid(row=0, column=1, sticky="we", padx=5, pady=5)

        # Stability slider
        ttk.Label(self.elevenlabs_frame, text="Stability:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        self.stability_var = tk.DoubleVar(value=DEFAULT_CONFIG["elevenlabs_stability"])
        self.stability_scale = ttk.Scale(
            self.elevenlabs_frame,
            from_=0,
            to=1,
            variable=self.stability_var,
            orient="horizontal"
        )
        self.stability_scale.grid(row=1, column=1, sticky="we", padx=5, pady=5)

        # Similarity Boost slider
        ttk.Label(self.elevenlabs_frame, text="Similarity:").grid(
            row=2, column=0, sticky="w", padx=5, pady=5
        )
        self.similarity_var = tk.DoubleVar(value=DEFAULT_CONFIG["elevenlabs_similarity_boost"])
        self.similarity_scale = ttk.Scale(
            self.elevenlabs_frame,
            from_=0,
            to=1,
            variable=self.similarity_var,
            orient="horizontal"
        )
        self.similarity_scale.grid(row=2, column=1, sticky="we", padx=5, pady=5)

        # Style slider
        ttk.Label(self.elevenlabs_frame, text="Style:").grid(
            row=3, column=0, sticky="w", padx=5, pady=5
        )
        self.style_var = tk.DoubleVar(value=DEFAULT_CONFIG["elevenlabs_style"])
        self.style_scale = ttk.Scale(
            self.elevenlabs_frame,
            from_=0,
            to=1,
            variable=self.style_var,
            orient="horizontal"
        )
        self.style_scale.grid(row=3, column=1, sticky="we", padx=5, pady=5)

        # Speed slider
        ttk.Label(self.elevenlabs_frame, text="Speed:").grid(
            row=4, column=0, sticky="w", padx=5, pady=5
        )
        self.speed_var = tk.DoubleVar(value=DEFAULT_CONFIG["elevenlabs_speed"])
        self.speed_scale = ttk.Scale(
            self.elevenlabs_frame,
            from_=0.5,
            to=2.0,
            variable=self.speed_var,
            orient="horizontal"
        )
        self.speed_scale.grid(row=4, column=1, sticky="we", padx=5, pady=5)

        # Speaker Boost checkbox
        self.speaker_boost_var = tk.BooleanVar(value=DEFAULT_CONFIG["elevenlabs_speaker_boost"])
        self.speaker_boost_check = ttk.Checkbutton(
            self.elevenlabs_frame,
            text="Speaker Boost",
            variable=self.speaker_boost_var
        )
        self.speaker_boost_check.grid(row=5, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        # Edge TTS Settings Frame (existing settings)
        self.edge_frame = ttk.LabelFrame(main_frame, text="Edge TTS Settings (Fallback)")
        self.edge_frame.grid(row=15, column=0, columnspan=2, sticky="we", padx=5, pady=5)

        # Move existing Edge TTS settings into edge_frame
        ttk.Label(self.edge_frame, text="Voice:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )

        self.voice_var = tk.StringVar(value=DEFAULT_CONFIG["edge_tts_voice"])
        self.voice_combo = ttk.Combobox(
            self.edge_frame, textvariable=self.voice_var, values=TTS_VOICES, state="readonly"
        )
        self.voice_combo.grid(row=0, column=1, sticky="we", padx=5, pady=5)

        # Rate
        ttk.Label(self.edge_frame, text="Rate:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )

        self.rate_var = tk.StringVar(value=DEFAULT_CONFIG["edge_tts_rate"])
        self.rate_combo = ttk.Combobox(
            self.edge_frame,
            textvariable=self.rate_var,
            values=["-20%", "-10%", "+0%", "+10%", "+15%", "+20%", "+30%"],
        )
        self.rate_combo.grid(row=1, column=1, sticky="we", padx=5, pady=5)

        # Volume
        ttk.Label(self.edge_frame, text="Volume:").grid(
            row=2, column=0, sticky="w", padx=5, pady=5
        )

        self.volume_var = tk.StringVar(value=DEFAULT_CONFIG["edge_tts_volume"])
        self.volume_combo = ttk.Combobox(
            self.edge_frame,
            textvariable=self.volume_var,
            values=["-20%", "-10%", "+0%", "+5%", "+10%", "+20%"],
        )
        self.volume_combo.grid(row=2, column=1, sticky="we", padx=5, pady=5)

        # Pitch
        ttk.Label(self.edge_frame, text="Pitch:").grid(
            row=3, column=0, sticky="w", padx=5, pady=5
        )

        self.pitch_var = tk.StringVar(value=DEFAULT_CONFIG["edge_tts_pitch"])
        self.pitch_combo = ttk.Combobox(
            self.edge_frame,
            textvariable=self.pitch_var,
            values=["-20Hz", "-10Hz", "+0Hz", "+10Hz", "+20Hz", "+30Hz"],
        )
        self.pitch_combo.grid(row=3, column=1, sticky="we", padx=5, pady=5)

        # Test voice button
        self.test_voice_btn = ttk.Button(
            self.edge_frame, text="Test Voice", command=self.test_voice
        )
        self.test_voice_btn.grid(
            row=4, column=0, columnspan=2, sticky="w", padx=5, pady=5
        )

        # Initial visibility
        self._on_engine_change(None)

        # Continue with rest of create_widgets
        # ...existing code...

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=18, column=0, columnspan=2, sticky="we", pady=10
        )

        # Appearance Settings
        ttk.Label(
            main_frame, text="Appearance Settings", font=("TkDefaultFont", 12, "bold")
        ).grid(row=19, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        # Dark mode
        self.dark_mode_var = tk.BooleanVar(value=DEFAULT_CONFIG["dark_mode"])
        self.dark_mode_check = ttk.Checkbutton(
            main_frame, text="Dark mode theme", variable=self.dark_mode_var
        )
        self.dark_mode_check.grid(
            row=20, column=0, columnspan=2, sticky="w", padx=5, pady=5
        )

        # Animation level
        ttk.Label(main_frame, text="Animation:").grid(
            row=21, column=0, sticky="w", padx=5, pady=5
        )

        self.animation_var = tk.StringVar(value=DEFAULT_CONFIG["animation_level"])
        self.animation_combo = ttk.Combobox(
            main_frame,
            textvariable=self.animation_var,
            values=["low", "medium", "high"],
            state="readonly",
            width=10,
        )
        self.animation_combo.grid(row=21, column=1, sticky="w", padx=5, pady=5)

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=22, column=0, columnspan=2, sticky="we", pady=10
        )

        # Output Settings
        ttk.Label(
            main_frame, text="Output Settings", font=("TkDefaultFont", 12, "bold")
        ).grid(row=23, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        # Output directory
        ttk.Label(main_frame, text="Output directory:").grid(
            row=24, column=0, sticky="w", padx=5, pady=5
        )

        self.output_dir_frame = ttk.Frame(main_frame)
        self.output_dir_frame.grid(row=24, column=1, sticky="we", padx=5, pady=5)

        self.output_dir_var = tk.StringVar(value=DEFAULT_CONFIG["output_dir"])
        self.output_dir_entry = ttk.Entry(
            self.output_dir_frame, textvariable=self.output_dir_var
        )
        self.output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.browse_btn = ttk.Button(
            self.output_dir_frame, text="Browse", command=self.browse_output_dir
        )
        self.browse_btn.pack(side=tk.RIGHT, padx=5)

        # Auto upload to YouTube
        self.upload_var = tk.BooleanVar(value=DEFAULT_CONFIG["upload"])
        self.upload_check = ttk.Checkbutton(
            main_frame,
            text="Auto upload to YouTube (requires setup)",
            variable=self.upload_var,
        )
        self.upload_check.grid(
            row=25, column=0, columnspan=2, sticky="w", padx=5, pady=5
        )

        # Configuration buttons
        self.config_btn_frame = ttk.Frame(main_frame)
        self.config_btn_frame.grid(
            row=26, column=0, columnspan=2, sticky="we", padx=5, pady=10
        )

        self.save_config_btn = ttk.Button(
            self.config_btn_frame, text="Save Config", command=self.save_config
        )
        self.save_config_btn.pack(side=tk.LEFT, padx=5)

        self.load_config_btn = ttk.Button(
            self.config_btn_frame, text="Load Config", command=self.load_config
        )
        self.load_config_btn.pack(side=tk.LEFT, padx=5)

        self.reset_config_btn = ttk.Button(
            self.config_btn_frame, text="Reset to Default", command=self.reset_config
        )
        self.reset_config_btn.pack(side=tk.RIGHT, padx=5)

        # Load subreddits from default config
        for subreddit in self.config["subreddits"]:
            self.subreddit_listbox.insert(tk.END, subreddit)

    def _on_engine_change(self, event):
        """Handle TTS engine selection change"""
        if self.tts_engine_var.get() == "elevenlabs":
            self.elevenlabs_frame.grid()
            self.edge_frame.grid_remove()
        else:
            self.elevenlabs_frame.grid_remove()
            self.edge_frame.grid()

    def add_subreddit(self):
        """Add a subreddit to the list"""
        subreddit = self.subreddit_var.get().strip()
        if not subreddit:
            return

        # Avoid duplicates
        existing = self.subreddit_listbox.get(0, tk.END)
        if subreddit in existing:
            return

        self.subreddit_listbox.insert(tk.END, subreddit)
        self.subreddit_var.set("")  # Clear entry

    def remove_subreddit(self):
        """Remove selected subreddit from the list"""
        selected = self.subreddit_listbox.curselection()
        if not selected:
            return

        for idx in reversed(selected):  # Remove from bottom to top
            self.subreddit_listbox.delete(idx)

    def clear_subreddits(self):
        """Clear all subreddits from the list"""
        self.subreddit_listbox.delete(0, tk.END)

    def test_voice(self):
        """Test the current voice settings with sample text"""
        voice = self.voice_var.get()
        rate = self.rate_var.get()
        volume = self.volume_var.get()
        pitch = self.pitch_var.get()

        # Sample text for testing
        sample_text = (
            "This is a test of the text-to-speech voice settings. How does it sound?"
        )

        self.test_voice_btn.config(state=tk.DISABLED, text="Generating...")

        # Start generation in a background thread
        threading.Thread(
            target=self._generate_test_audio,
            args=(sample_text, voice, rate, volume, pitch),
            daemon=True,
        ).start()

    def _generate_test_audio(self, text, voice, rate, volume, pitch):
        """Generate and play test audio in a background thread"""
        try:
            # Create temp file for audio
            import tempfile
            import pygame

            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_file.close()

            # Generate audio file with edge_tts
            async def generate_audio():
                communicate = edge_tts.Communicate(
                    text, voice, rate=rate, volume=volume, pitch=pitch
                )
                await communicate.save(temp_file.name)

            # Run async code
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(generate_audio())

            # Play the audio
            pygame.mixer.init()
            pygame.mixer.music.load(temp_file.name)
            pygame.mixer.music.play()

            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            # Cleanup
            pygame.mixer.quit()
            os.unlink(temp_file.name)

            # Update UI in main thread
            self.parent.after(
                0,
                lambda: self.test_voice_btn.config(state=tk.NORMAL, text="Test Voice"),
            )

        except Exception as e:
            # Store error message before passing to lambda
            error_message = str(e)
            
            # Show error in main thread with the pre-captured error message
            self.parent.after(
                0,
                lambda error=error_message: messagebox.showerror(
                    "Voice Test Error", f"Error testing voice: {error}"
                ),
            )
            self.parent.after(
                0,
                lambda: self.test_voice_btn.config(state=tk.NORMAL, text="Test Voice"),
            )

    def browse_output_dir(self):
        """Open directory browser for output directory selection"""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)

    def save_config(self):
        """Save current configuration to a file"""
        # Get current configuration with layout
        config = self.parent.save_layout()

        # Ask for file location
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(__file__)),
        )

        if not filepath:
            return

        try:
            # Write to file
            with open(filepath, "w") as f:
                json.dump(config, f, indent=2)

            messagebox.showinfo("Success", f"Configuration saved to {filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

    def load_config(self):
        """Load configuration from a file"""
        # Ask for file location
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(__file__)),
        )

        if not filepath:
            return

        try:
            # Read from file
            with open(filepath, "r") as f:
                config = json.load(f)

            # Apply configuration
            self.apply_config(config)
            
            # Apply layout settings
            self.parent.load_layout(config)
            
            messagebox.showinfo("Success", f"Configuration loaded from {filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")

    def reset_config(self):
        """Reset configuration to default values"""
        if messagebox.askyesno("Confirm", "Reset all settings to default values?"):
            self.apply_config(DEFAULT_CONFIG.copy())

    def apply_config(self, config):
        """Apply configuration values to the UI"""
        # Store configuration
        self.config = config.copy()

        # Update UI elements
        # Clear and repopulate subreddit list
        self.subreddit_listbox.delete(0, tk.END)
        for subreddit in config.get("subreddits", []):
            self.subreddit_listbox.insert(tk.END, subreddit)

        # Update other fields
        self.memes_var.set(
            config.get("memes_per_video", DEFAULT_CONFIG["memes_per_video"])
        )
        self.upvotes_var.set(config.get("min_upvotes", DEFAULT_CONFIG["min_upvotes"]))
        self.post_type_var.set(config.get("post_type", DEFAULT_CONFIG["post_type"]))
        self.auto_mode_var.set(config.get("auto_mode", DEFAULT_CONFIG["auto_mode"]))
        self.bg_music_var.set(
            config.get("use_background_music", DEFAULT_CONFIG["use_background_music"])
        )
        self.voice_var.set(
            config.get("edge_tts_voice", DEFAULT_CONFIG["edge_tts_voice"])
        )
        self.rate_var.set(config.get("edge_tts_rate", DEFAULT_CONFIG["edge_tts_rate"]))
        self.volume_var.set(
            config.get("edge_tts_volume", DEFAULT_CONFIG["edge_tts_volume"])
        )
        self.pitch_var.set(
            config.get("edge_tts_pitch", DEFAULT_CONFIG["edge_tts_pitch"])
        )
        self.dark_mode_var.set(config.get("dark_mode", DEFAULT_CONFIG["dark_mode"]))
        self.animation_var.set(
            config.get("animation_level", DEFAULT_CONFIG["animation_level"])
        )
        self.output_dir_var.set(config.get("output_dir", DEFAULT_CONFIG["output_dir"]))
        self.upload_var.set(config.get("upload", DEFAULT_CONFIG["upload"]))

        # Apply TTS-specific settings
        self.tts_engine_var.set("elevenlabs" if config.get("use_elevenlabs", True) else "edge")
        self.elevenlabs_voice_var.set(config.get("elevenlabs_voice_id", DEFAULT_CONFIG["elevenlabs_voice_id"]))
        self.stability_var.set(config.get("elevenlabs_stability", DEFAULT_CONFIG["elevenlabs_stability"]))
        self.similarity_var.set(config.get("elevenlabs_similarity_boost", DEFAULT_CONFIG["elevenlabs_similarity_boost"]))
        self.style_var.set(config.get("elevenlabs_style", DEFAULT_CONFIG["elevenlabs_style"]))
        self.speaker_boost_var.set(config.get("elevenlabs_speaker_boost", DEFAULT_CONFIG["elevenlabs_speaker_boost"]))
        self.speed_var.set(config.get("elevenlabs_speed", DEFAULT_CONFIG["elevenlabs_speed"]))
        
        # Update visibility
        self._on_engine_change(None)

    def update_config(self):
        """Update configuration from UI values"""
        # Get subreddits from listbox
        subreddits = list(self.subreddit_listbox.get(0, tk.END))

        # Build configuration object
        self.config = {
            "subreddits": subreddits,
            "memes_per_video": self.memes_var.get(),
            "min_upvotes": self.upvotes_var.get(),
            "post_type": self.post_type_var.get(),
            "auto_mode": self.auto_mode_var.get(),
            "use_background_music": self.bg_music_var.get(),
            "edge_tts_voice": self.voice_var.get(),
            "edge_tts_rate": self.rate_var.get(),
            "edge_tts_volume": self.volume_var.get(),
            "edge_tts_pitch": self.pitch_var.get(),
            "dark_mode": self.dark_mode_var.get(),
            "animation_level": self.animation_var.get(),
            "output_dir": self.output_dir_var.get(),
            "upload": self.upload_var.get(),
            "use_elevenlabs": self.tts_engine_var.get() == "elevenlabs",
            "elevenlabs_voice_id": self.elevenlabs_voice_var.get().split("(")[-1].strip(")"),
            "elevenlabs_model": "eleven_flash_v2_5",
            "elevenlabs_stability": self.stability_var.get(),
            "elevenlabs_similarity_boost": self.similarity_var.get(),
            "elevenlabs_style": self.style_var.get(),
            "elevenlabs_speaker_boost": self.speaker_boost_var.get(),
            "elevenlabs_speed": self.speed_var.get(),
        }

        return self.config

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration"""
        return self.update_config()


class QueuePanel(ttk.Frame):
    """Panel for managing video generation queue"""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

        # Queue of VideoQueueItem objects
        self.queue = []
        self.active_tasks = 0
        self.max_concurrent_tasks = 1  # Default to 1, can be adjusted
        
        # Log buffer for each queue item
        self.log_buffers = {}  # item_id -> log string
        self.currently_viewing_log = None  # Currently displayed log item_id

        # Create UI elements
        self.create_widgets()

        # Check queue periodically
        self.after(1000, self.check_queue)

    def create_widgets(self):
        """Create all UI widgets for the queue panel"""
        # Title bar
        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.title_label = ttk.Label(
            self.title_frame,
            text="Video Generation Queue",
            font=("TkDefaultFont", 14, "bold"),
        )
        self.title_label.pack(side=tk.LEFT)

        # Create a PanedWindow for queue list and log viewer
        self.main_paned = ttk.PanedWindow(self, orient=tk.VERTICAL)
        self.main_paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Queue list frame
        self.queue_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.queue_frame, weight=1)

        # Column headers
        self.header_frame = ttk.Frame(self.queue_frame)
        self.header_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))

        ttk.Label(self.header_frame, text="Subreddit", width=15).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(self.header_frame, text="Status", width=10).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(self.header_frame, text="Progress", width=20).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(self.header_frame, text="Actions", width=25).pack(
            side=tk.LEFT, padx=(0, 10)
        )

        # Queue items will be added here
        self.queue_list_frame = ScrollableFrame(self.queue_frame)
        self.queue_list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Log viewer frame
        self.log_frame = ttk.LabelFrame(self.main_paned, text="Process Log")
        self.main_paned.add(self.log_frame, weight=1)

        # Log text widget with scrollbar
        self.log_text = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)  # Read-only initially
        
        # Metadata display (for video title, description, tags)
        self.metadata_frame = ttk.LabelFrame(self.main_paned, text="Generated Metadata")
        self.main_paned.add(self.metadata_frame, weight=1)
        
        # Create notebook for metadata tabs
        self.metadata_notebook = ttk.Notebook(self.metadata_frame)
        self.metadata_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Title/Description tab
        self.title_desc_frame = ttk.Frame(self.metadata_notebook)
        self.metadata_notebook.add(self.title_desc_frame, text="Title & Description")
        
        # Add title and description fields
        title_frame = ttk.Frame(self.title_desc_frame)
        title_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(title_frame, text="Title:").pack(side=tk.LEFT)
        self.title_entry = ttk.Entry(title_frame, width=50)
        self.title_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        desc_frame = ttk.Frame(self.title_desc_frame)
        desc_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Label(desc_frame, text="Description:").pack(anchor=tk.W)
        self.desc_text = scrolledtext.ScrolledText(desc_frame, height=5)
        self.desc_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Comments tab
        self.comments_frame = ttk.Frame(self.metadata_notebook)
        self.metadata_notebook.add(self.comments_frame, text="Comments")
        
        self.comments_text = scrolledtext.ScrolledText(self.comments_frame)
        self.comments_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tags tab
        self.tags_frame = ttk.Frame(self.metadata_notebook)
        self.metadata_notebook.add(self.tags_frame, text="Tags")
        
        self.tags_text = scrolledtext.ScrolledText(self.tags_frame)
        self.tags_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Metadata copy buttons
        metadata_btn_frame = ttk.Frame(self.metadata_frame)
        metadata_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.copy_title_btn = ttk.Button(metadata_btn_frame, text="Copy Title", 
                                        command=lambda: self.copy_to_clipboard(self.title_entry.get()))
        self.copy_title_btn.pack(side=tk.LEFT, padx=2)
        
        self.copy_desc_btn = ttk.Button(metadata_btn_frame, text="Copy Description", 
                                       command=lambda: self.copy_to_clipboard(self.desc_text.get(1.0, tk.END)))
        self.copy_desc_btn.pack(side=tk.LEFT, padx=2)
        
        self.copy_comments_btn = ttk.Button(metadata_btn_frame, text="Copy Comments",
                                          command=lambda: self.copy_to_clipboard(self.comments_text.get(1.0, tk.END)))
        self.copy_comments_btn.pack(side=tk.LEFT, padx=2)
        
        self.copy_tags_btn = ttk.Button(metadata_btn_frame, text="Copy Tags",
                                      command=lambda: self.copy_to_clipboard(self.tags_text.get(1.0, tk.END)))
        self.copy_tags_btn.pack(side=tk.LEFT, padx=2)

        # Control panel
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.clear_completed_btn = ttk.Button(
            self.control_frame, text="Clear Completed", command=self.clear_completed
        )
        self.clear_completed_btn.pack(side=tk.LEFT, padx=5)

        self.clear_all_btn = ttk.Button(
            self.control_frame, text="Clear All", command=self.clear_all
        )
        self.clear_all_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_log_btn = ttk.Button(
            self.control_frame, text="Clear Log", command=self.clear_log
        )
        self.clear_log_btn.pack(side=tk.LEFT, padx=5)

        # Concurrent tasks setting
        ttk.Label(self.control_frame, text="Concurrent Tasks:").pack(
            side=tk.LEFT, padx=(15, 5)
        )

        self.concurrent_var = tk.IntVar(value=self.max_concurrent_tasks)
        self.concurrent_spinbox = ttk.Spinbox(
            self.control_frame,
            from_=1,
            to=4,
            textvariable=self.concurrent_var,
            width=5,
            command=self.update_concurrent_tasks,
        )
        self.concurrent_spinbox.pack(side=tk.LEFT)
        
        # Set initial sash positions after a short delay
        self.after(100, lambda: self.main_paned.sashpos(0, 200))
        self.after(100, lambda: self.main_paned.sashpos(1, 400))

    def copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Copied", "Content copied to clipboard")

    def clear_log(self):
        """Clear the log display"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_queue_display(self):
        """Update the queue display with current items"""
        # Clear existing items
        for widget in self.queue_list_frame.scrollable_frame.winfo_children():
            widget.destroy()

        # Add each queue item to the display
        for idx, item in enumerate(self.queue):
            # Generate a unique ID for the item for log tracking
            if not hasattr(item, 'item_id'):
                item.item_id = f"item_{id(item)}"
                # Initialize log buffer if needed
                if item.item_id not in self.log_buffers:
                    self.log_buffers[item.item_id] = ""
            
            item_frame = ttk.Frame(self.queue_list_frame.scrollable_frame)
            item_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

            # Subreddit name
            subreddit_label = ttk.Label(
                item_frame, text=f"r/{item.subreddit}", width=15
            )
            subreddit_label.pack(side=tk.LEFT, padx=(0, 10))

            # Status with more detail
            status_text = item.status.title()
            if item.status == "processing" and hasattr(item, 'current_step'):
                status_text = f"{status_text}: {item.current_step}"
            
            status_label = ttk.Label(item_frame, text=status_text, width=10)
            status_label.pack(side=tk.LEFT, padx=(0, 10))

            # Progress with percentage text
            progress_frame = ttk.Frame(item_frame, width=150)
            progress_frame.pack(side=tk.LEFT, padx=(0, 10))
            
            if item.status == "processing":
                progress_var = tk.IntVar(value=item.progress)
                progress_bar = ttk.Progressbar(
                    progress_frame, variable=progress_var, length=100, mode="determinate"
                )
                progress_bar.pack(side=tk.LEFT)
                
                # Add percentage text
                percent_label = ttk.Label(progress_frame, text=f"{item.progress}%", width=5)
                percent_label.pack(side=tk.LEFT, padx=5)
                
                # Store progress var and label reference
                item_frame.progress_var = progress_var
                item_frame.percent_label = percent_label
            else:
                placeholder = ttk.Frame(progress_frame, width=140)
                placeholder.pack(side=tk.LEFT)

            # Action buttons based on status
            actions_frame = ttk.Frame(item_frame)
            actions_frame.pack(side=tk.LEFT, padx=(0, 10))

            # View log button for all items
            ttk.Button(
                actions_frame,
                text="View Log",
                command=lambda i=item.item_id: self.view_item_log(i),
                width=8
            ).pack(side=tk.LEFT, padx=2)
            
            # View metadata button for completed items
            if item.status == "completed" and item.metadata:
                ttk.Button(
                    actions_frame,
                    text="View Metadata",
                    command=lambda m=item.metadata: self.view_metadata(m),
                    width=12
                ).pack(side=tk.LEFT, padx=2)

            if item.status == "queued":
                # Cancel button for queued items
                ttk.Button(
                    actions_frame,
                    text="Cancel",
                    command=lambda i=idx: self.cancel_item(i),
                    width=8
                ).pack(side=tk.LEFT, padx=2)
            elif item.status == "processing":
                # Can't cancel while processing, but show a details button
                ttk.Label(actions_frame, text="Processing...").pack(
                    side=tk.LEFT, padx=2
                )
            elif item.status == "completed":
                # Open video button for completed items
                ttk.Button(
                    actions_frame,
                    text="Open",
                    command=lambda p=item.output_path: self.open_video(p),
                    width=8
                ).pack(side=tk.LEFT, padx=2)
                # Open folder button
                ttk.Button(
                    actions_frame,
                    text="Folder",
                    command=lambda p=item.output_path: self.open_folder(p),
                    width=8
                ).pack(side=tk.LEFT, padx=2)
            elif item.status == "failed":
                # Retry button for failed items
                ttk.Button(
                    actions_frame,
                    text="Retry",
                    command=lambda i=idx: self.retry_item(i),
                    width=8
                ).pack(side=tk.LEFT, padx=2)

    def view_item_log(self, item_id):
        """Display the log for a specific queue item"""
        if item_id in self.log_buffers:
            # Update log display
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, self.log_buffers[item_id])
            self.log_text.config(state=tk.DISABLED)
            
            # Scroll to the end
            self.log_text.see(tk.END)
            
            # Remember which log we're viewing
            self.currently_viewing_log = item_id

    def view_metadata(self, metadata):
        """Display video metadata in the metadata tabs"""
        # Clear previous content
        self.title_entry.delete(0, tk.END)
        self.desc_text.delete(1.0, tk.END)
        self.comments_text.delete(1.0, tk.END)
        self.tags_text.delete(1.0, tk.END)
        
        if not metadata:
            return
            
        # Extract metadata from the first key (there should only be one)
        video_path = list(metadata.keys())[0] if metadata else None
        if not video_path or video_path not in metadata:
            return
            
        video_metadata = metadata[video_path]
        
        # Set title and description if available
        if 'title' in video_metadata:
            self.title_entry.insert(0, video_metadata['title'])
            
        if 'description' in video_metadata:
            self.desc_text.insert(1.0, video_metadata['description'])
            
        # Set comments if available
        if 'comments' in video_metadata:
            comments = video_metadata['comments']
            if isinstance(comments, list):
                self.comments_text.insert(1.0, '\n\n'.join(comments))
            elif isinstance(comments, str):
                self.comments_text.insert(1.0, comments)
        
        # Set tags if available
        if 'tags' in video_metadata:
            tags = video_metadata['tags']
            if isinstance(tags, list):
                self.tags_text.insert(1.0, ', '.join(tags))
            elif isinstance(tags, str):
                self.tags_text.insert(1.0, tags)
        
        # Switch to the first tab
        self.metadata_notebook.select(0)

    def add_to_queue(self, queue_item):
        """Add a new item to the queue"""
        self.queue.append(queue_item)
        self.update_queue_display()
        self.check_queue()  # Check if we can start processing

    # ...existing code...

    def _generate_video_thread(self, item):
        """Worker thread for video generation - modified to use GUI-optimized generator and add logging"""
        try:
            # Extract only the parameters that Config accepts
            # Create a filtered config dictionary with only the parameters Config accepts
            config_dict = {
                "subreddits": item.config.get("subreddits", ["memes"]),
                "min_upvotes": item.config.get("min_upvotes", 3000),
                "auto_mode": item.config.get("auto_mode", False),
                "edge_tts_voice": item.config.get("edge_tts_voice", "en-AU-WilliamNeural"),
                "edge_tts_rate": item.config.get("edge_tts_rate", "+15%"),
                "edge_tts_volume": item.config.get("edge_tts_volume", "+5%"),
                "edge_tts_pitch": item.config.get("edge_tts_pitch", "+30Hz"),
                "upload": item.config.get("upload", False)
            }
            
            # Add to log
            self._add_to_log(item.item_id, f"Starting video generation for r/{item.subreddit}")
            self._add_to_log(item.item_id, f"Configuration: {json.dumps(config_dict, indent=2)}")
            
            # Initialize components with filtered config
            config = Config(**config_dict)
            ai_client = AIClient()
            media_processor = MediaProcessor(config)

            # Use our GUI-optimized video generator with logging capability
            video_generator = GUIVideoGenerator(config, ai_client, media_processor)
            
            # Add logging callbacks
            def log_callback(message):
                self._add_to_log(item.item_id, message)
                
                # Extract progress information if available
                if "Progress:" in message:
                    try:
                        # Update the current step
                        step_match = re.search(r"Step: ([^,]+),", message)
                        if step_match:
                            item.current_step = step_match.group(1)
                            
                        # Update progress percentage if available
                        percent_match = re.search(r"(\d+)%", message)
                        if percent_match:
                            item.progress = int(percent_match.group(1))
                            # Update UI
                            self.parent.after(0, self.update_queue_display)
                    except:
                        pass  # Ignore parsing errors
            
            # Attach logger to video generator
            video_generator.logger = log_callback
            
            # Log meme information
            if item.meme_urls:
                self._add_to_log(item.item_id, f"Using {len(item.meme_urls)} pre-selected memes")
            else:
                self._add_to_log(item.item_id, f"Will fetch {item.memes_count} memes from r/{item.subreddit}")
            
            # Update progress periodically
            self.start_progress_updates(item)
            
            # Update current step
            item.current_step = "Initializing"
            self.parent.after(0, self.update_queue_display)

            # Generate video - this won't produce CLI prompts
            metadata = video_generator.generate_video(
                subreddit=item.subreddit,
                amount=item.memes_count,
                post_type=item.post_type,
                output_location=item.output_dir,
                pre_meme_urls=item.meme_urls,  # May be None
                pre_captions=item.captions,  # May be None
            )
            
            self._add_to_log(item.item_id, "Video generation complete, finalizing...")
            item.current_step = "Finalizing"
            self.parent.after(0, self.update_queue_display)

            # Wait for all processing to complete
            video_generator.wait_for_completion()

            # Video is complete, update item
            output_path = list(metadata.keys())[0] if metadata else None
            item.output_path = output_path
            item.metadata = metadata
            item.status = "completed"
            item.progress = 100
            item.current_step = "Completed"
            
            # Log completion
            self._add_to_log(item.item_id, f"Video successfully generated: {output_path}")
            
            # If this is the current log being viewed, update it
            if self.currently_viewing_log == item.item_id:
                self.view_item_log(item.item_id)
                
            # Also show metadata if available
            if metadata:
                self.view_metadata(metadata)

            # Update UI
            self.parent.after(0, self.update_queue_display)
        except Exception as e:
            # Store error message before passing to lambda
            error_message = str(e)
            item.status = "failed"

            # Update UI with the pre-captured error message
            self.parent.after(0, self.update_queue_display)
            self.parent.after(
                0,
                lambda msg=error_message: messagebox.showerror(
                    "Generation Error",
                    f"Error generating video for r/{item.subreddit}: {msg}",
                ),
            )

    def start_progress_updates(self, item):
        """Start periodic updates of progress percentage"""
        # This is just a simulation since we don't have real-time progress tracking
        # In a real implementation, we would use callbacks from the video generator

        def update_progress():
            if item.status != "processing":
                return  # Stop updating if no longer processing

            # Increment progress by 5-15% each time
            item.progress += random.randint(5, 15)
            if item.progress > 95:
                item.progress = 95  # Cap at 95% until actually complete

            # Update UI
            self.parent.after(0, self.update_progress_ui)

            # Schedule next update if still processing
            if item.status == "processing":
                delay = random.randint(2000, 5000)  # 2-5 seconds
                self.parent.after(delay, update_progress)

        # Start first update
        self.parent.after(1000, update_progress)

    def update_progress_ui(self):
        """Update progress bars in the UI"""
        for idx, item in enumerate(self.queue):
            # Find the frame for this item
            try:
                item_frame = self.queue_list_frame.scrollable_frame.winfo_children()[
                    idx
                ]
                if hasattr(item_frame, "progress_var") and item.status == "processing":
                    item_frame.progress_var.set(item.progress)
                    item_frame.percent_label.config(text=f"{item.progress}%")
            except (IndexError, AttributeError):
                pass

    def _add_to_log(self, item_id, message):
        """Add a message to the log buffer for a specific item"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        if item_id in self.log_buffers:
            self.log_buffers[item_id] += log_message
        else:
            self.log_buffers[item_id] = log_message

        # If this is the current log being viewed, update it
        if self.currently_viewing_log == item_id:
            self.view_item_log(item_id)

    def cancel_item(self, idx):
        """Cancel a queued item"""
        if 0 <= idx < len(self.queue) and self.queue[idx].status == "queued":
            del self.queue[idx]
            self.update_queue_display()

    def retry_item(self, idx):
        """Retry a failed item"""
        if 0 <= idx < len(self.queue) and self.queue[idx].status == "failed":
            self.queue[idx].status = "queued"
            self.queue[idx].progress = 0
            self.update_queue_display()
            self.check_queue()  # Check if we can start processing

    def open_video(self, path):
        """Open the generated video"""
        if path and os.path.exists(path):
            os.startfile(path)
        else:
            messagebox.showwarning("Warning", "Video file not found")

    def open_folder(self, path):
        """Open the folder containing the video"""
        if path:
            folder = os.path.dirname(path)
            if os.path.exists(folder):
                os.startfile(folder)
            else:
                messagebox.showwarning("Warning", "Folder not found")

    def clear_completed(self):
        """Clear completed items from the queue"""
        self.queue = [
            item for item in self.queue if item.status not in ("completed", "failed")
        ]
        self.update_queue_display()

    def clear_all(self):
        """Clear all items from the queue"""
        if messagebox.askyesno(
            "Confirm", "Clear entire queue? Processing items cannot be stopped."
        ):
            self.queue = [item for item in self.queue if item.status == "processing"]
            self.update_queue_display()

    def check_queue(self):
        """Check queue for items that need processing"""
        # Count active tasks
        self.active_tasks = sum(1 for item in self.queue if item.status == "processing")

        # If we can run more tasks, find the next queued item
        if self.active_tasks < self.max_concurrent_tasks:
            for item in self.queue:
                if (item.status == "queued" and 
                    self.active_tasks < self.max_concurrent_tasks):
                    self.process_item(item)
                    self.active_tasks += 1  # Increment active task count
                    break

        # Schedule next check
        self.after(1000, self.check_queue)

    def process_item(self, item):
        """Process a queue item (generate video)"""
        item.status = "processing"
        self.update_queue_display()

        # Start processing in a background thread
        thread = threading.Thread(
            target=self._generate_video_thread, args=(item,), daemon=True
        )
        thread.start()

    def update_concurrent_tasks(self, *args):
        """Update the maximum number of concurrent tasks"""
        try:
            value = int(self.concurrent_var.get())
            if 1 <= value <= 4:  # Validate the range
                self.max_concurrent_tasks = value
                # Check queue immediately to start new tasks if needed
                self.check_queue()
        except (ValueError, tk.TclError):
            # Reset to previous value if invalid
            self.concurrent_var.set(self.max_concurrent_tasks)


# Add a meme selection dialog class for picking memes
class MemeSelectionDialog(tk.Toplevel):
    """Dialog for selecting which memes to include in the final video"""

    def __init__(self, parent, meme_urls, captions=None, required_amount=3):
        super().__init__(parent)
        self.parent = parent
        self.meme_urls = meme_urls
        self.captions = (
            captions if captions else [[] for _ in meme_urls]
        )  # Empty captions if none provided
        self.required_amount = required_amount
        self.result = None
        self.selected_indices = []

        # Configure dialog
        self.title("Meme Selection")
        self.geometry("900x700")
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.transient(parent)
        self.grab_set()

        # Create UI elements
        self.create_widgets()

        # Populate meme list
        self.populate_memes()

        # Wait for user interaction
        self.wait_window()
        
    # Add the missing method
    def on_cancel(self):
        """Handle dialog cancellation"""
        self.result = None
        self.destroy()

    def create_widgets(self):
        """Create UI widgets for the selection dialog"""
        # Main container
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Instructions
        instructions = ttk.Label(
            main_frame,
            text=f"Select {self.required_amount} memes to include in your video. You can drag to reorder them.",
            font=("TkDefaultFont", 10, "bold"),
        )
        instructions.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        # Selection count
        self.count_label = ttk.Label(
            main_frame, text=f"Selected: 0/{self.required_amount}"
        )
        self.count_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        # Create treeview for meme list with reordering capability
        self.tree_frame = ttk.Frame(main_frame)
        self.tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))

        self.tree = ttk.Treeview(
            self.tree_frame, columns=("title", "author", "selected"), show="headings"
        )
        self.tree.heading("title", text="Title")
        self.tree.heading("author", text="Author")
        self.tree.heading("selected", text="Selected")

        self.tree.column("title", width=400)
        self.tree.column("author", width=150)
        self.tree.column("selected", width=80)

        # Add scrollbar
        tree_scrollbar = ttk.Scrollbar(
            self.tree_frame, orient="vertical", command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=tree_scrollbar.set)

        # Pack tree and scrollbar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind events
        self.tree.bind("<Double-1>", self.on_double_click)
        self.tree.bind("<ButtonPress-1>", self.on_tree_click)
        self.tree.bind("<B1-Motion>", self.on_tree_drag)
        self.tree.bind("<ButtonRelease-1>", self.on_tree_release)

        # Preview frame for the selected meme
        self.preview_frame = ttk.LabelFrame(
            main_frame, text="Meme Preview", padding="10"
        )
        self.preview_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10), ipady=5)

        # Preview canvas
        self.preview_canvas = tk.Canvas(
            self.preview_frame, width=300, height=200, bg="white"
        )
        self.preview_canvas.pack(side=tk.TOP, pady=10)

        # Caption editor - only show if captions are provided
        if any(self.captions):
            caption_frame = ttk.Frame(self.preview_frame)
            caption_frame.pack(side=tk.TOP, fill=tk.X, expand=True)

            ttk.Label(caption_frame, text="Caption:").pack(side=tk.TOP, anchor=tk.W)

            # Increased height from 4 to 6
            self.caption_editor = scrolledtext.ScrolledText(caption_frame, height=6)
            self.caption_editor.pack(side=tk.TOP, fill=tk.X, expand=True)

            # Edit and update buttons
            button_frame = ttk.Frame(caption_frame)
            button_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))

            self.edit_btn = ttk.Button(
                button_frame,
                text="Edit with SSML Editor",
                command=self.open_ssml_editor,
            )
            self.edit_btn.pack(side=tk.LEFT)

            self.update_btn = ttk.Button(
                button_frame, text="Update Caption", command=self.update_caption
            )
            self.update_btn.pack(side=tk.LEFT, padx=(5, 0))

        # Processing status indicator
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.pack(side=tk.TOP, fill=tk.X)

        # Buttons at the bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        self.select_all_btn = ttk.Button(
            button_frame,
            text=f"Select First {self.required_amount}",
            command=self.select_first_n,
        )
        self.select_all_btn.pack(side=tk.LEFT)

        self.cancel_btn = ttk.Button(
            button_frame, text="Cancel", command=self.on_cancel
        )
        self.cancel_btn.pack(side=tk.RIGHT, padx=(5, 0))

        self.ok_btn = ttk.Button(button_frame, text="OK", command=self.on_ok)
        self.ok_btn.pack(side=tk.RIGHT)

        # Variables for reordering
        self.drag_source = None
        self.drag_image = None

    def populate_memes(self):
        """Populate the listbox with available memes"""
        for i, (url, author, title) in enumerate(self.meme_urls):
            self.tree.insert(
                "", tk.END, iid=str(i), values=(title, f"u/{author}", "No")
            )

    def on_tree_click(self, event):
        """Handle initial click for drag and drop"""
        region = self.tree.identify_region(event.x, event.y)
        if (region != "cell"):
            return

        # Get clicked item
        item = self.tree.identify_row(event.y)
        if not item:
            return

        # Save for drag operation
        self.drag_source = item

        # Preview the clicked item
        self.preview_meme(int(item))

    def on_tree_drag(self, event):
        """Handle drag motion for reordering"""
        if not self.drag_source:
            return

        # Get target item
        target = self.tree.identify_row(event.y)
        if not target or target == self.drag_source:
            return

        # Move item
        self.tree.move(self.drag_source, "", self.tree.index(target))

    def on_tree_release(self, event):
        """Handle release to complete drag operation"""
        # Reset drag variables
        self.drag_source = None

        # Update selection indices if needed
        self.update_selection_order()

    def update_selection_order(self):
        """Update the selection indices after reordering"""
        # Rebuild selection list based on current order
        new_selection = []
        for i in range(len(self.meme_urls)):
            item = self.tree.item(str(i))
            if item["values"][2] == "Yes":
                new_selection.append(i)

        self.selected_indices = new_selection
        self.update_count_label()

    def on_double_click(self, event):
        """Toggle selection on double click"""
        # Get clicked item
        item = self.tree.identify_row(event.y)
        if not item:
            return

        idx = int(item)

        # Toggle selection
        if idx in self.selected_indices:
            self.selected_indices.remove(idx)
            self.tree.item(
                item,
                values=(
                    self.tree.item(item)["values"][0],
                    self.tree.item(item)["values"][1],
                    "No",
                ),
            )
        else:
            # Only add if we haven't reached the limit
            if len(self.selected_indices) < self.required_amount:
                self.selected_indices.append(idx)
                self.tree.item(
                    item,
                    values=(
                        self.tree.item(item)["values"][0],
                        self.tree.item(item)["values"][1],
                        "Yes",
                    ),
                )

        # Update preview and counter
        self.update_count_label()

    def preview_meme(self, idx):
        """Show a preview of the selected meme"""
        if 0 <= idx < len(self.meme_urls):
            meme_url, _, _ = self.meme_urls[idx]

            # Start loading the image using the global thread pool
            IMAGE_THREAD_POOL.submit(self.load_preview_image, meme_url)

            # Update caption editor if it exists and captions are available
            if (
                hasattr(self, "caption_editor")
                and idx < len(self.captions)
                and self.captions[idx]
            ):
                caption_group = self.captions[idx]
                plain_text = "\n".join(
                    [strip_ssml_tags(caption) for caption in caption_group]
                )
                self.caption_editor.delete(1.0, tk.END)
                self.caption_editor.insert(tk.END, plain_text)

            # Store current index for later use
            self.current_preview_index = idx

    def load_preview_image(self, url):
        """Load and display the preview image using the cached function"""
        try:
            # Use cached image function
            image = get_image_from_url(url)

            # Calculate resize dimensions
            canvas_width = self.preview_canvas.winfo_width() or 300
            canvas_height = self.preview_canvas.winfo_height() or 200

            # Resize image to fit preview
            scale = (
                min(canvas_width / image.width, canvas_height / image.height) * 0.9
            )
            new_width = int(image.width * scale)
            new_height = int(image.height * scale)
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(resized_image)

            # Update UI in main thread
            self.after(
                0, lambda: self.update_preview_image(photo, new_width, new_height)
            )

        except Exception as e:
            # Show error message
            self.after(
                0,
                lambda: self.preview_canvas.create_text(
                    150, 100, text=f"Error loading image:\n{str(e)}", fill="red"
                ),
            )

    def update_preview_image(self, photo, width, height):
        """Update the preview canvas with the loaded image"""
        # Clear canvas
        self.preview_canvas.delete("all")

        # Get canvas dimensions
        canvas_width = self.preview_canvas.winfo_width() or 300
        canvas_height = self.preview_canvas.winfo_height() or 200

        # Calculate center position
        x = canvas_width // 2
        y = canvas_height // 2

        # Display image
        self.preview_canvas.create_image(x, y, image=photo, anchor=tk.CENTER)
        self.photo_ref = photo  # Keep reference to prevent garbage collection

    def update_count_label(self):
        """Update the selection count label"""
        self.count_label.config(
            text=f"Selected: {len(self.selected_indices)}/{self.required_amount}"
        )

        # Enable/disable OK button based on selection count
        if len(self.selected_indices) == self.required_amount:
            self.ok_btn.config(state=tk.NORMAL)
        else:
            self.ok_btn.config(state=tk.DISABLED)

    def select_first_n(self):
        """Select the first n memes automatically"""
        # Clear existing selection
        self.selected_indices = []

        # Select first n items
        for i in range(min(self.required_amount, len(self.meme_urls))):
            self.selected_indices.append(i)
            self.tree.item(
                str(i),
                values=(
                    self.tree.item(str(i))["values"][0],
                    self.tree.item(str(i))["values"][1],
                    "Yes",
                ),
            )

        # Update counter
        self.update_count_label()

    def open_ssml_editor(self):
        """Open the SSML editor for the current caption"""
        if not hasattr(self, "current_preview_index") or not hasattr(
            self, "caption_editor"
        ):
            return

        idx = self.current_preview_index
        if 0 <= idx < len(self.captions) and self.captions[idx]:
            caption_group = self.captions[idx]

            # Use SSML editor for a more comprehensive editing experience
            try:
                edited_captions = edit_ssml_captions(
                    caption_group,
                    voice_id=self.parent.config_panel.voice_var.get(),
                    rate=self.parent.config_panel.rate_var.get(),
                    volume=self.parent.config_panel.volume_var.get(),
                    pitch=self.parent.config_panel.pitch_var.get(),
                )

                # Update the captions
                self.captions[idx] = edited_captions

                # Update preview
                plain_text = "\n".join(
                    [strip_ssml_tags(caption) for caption in edited_captions]
                )
                self.caption_editor.delete(1.0, tk.END)
                self.caption_editor.insert(tk.END, plain_text)

            except Exception as e:
                messagebox.showerror(
                    "SSML Editor Error", f"Error editing captions: {str(e)}"
                )

    def update_caption(self):
        """Update caption based on text editor content"""
        if not hasattr(self, "current_preview_index") or not hasattr(
            self, "caption_editor"
        ):
            return

        idx = self.current_preview_index
        if 0 <= idx < len(self.captions):
            # Get edited text
            edited_text = self.caption_editor.get(1.0, tk.END).strip()

            # Split into lines for multiple captions
            lines = [line.strip() for line in edited_text.split("\n") if line.strip()]

            if not lines:
                messagebox.showerror("Error", "Caption cannot be empty")
                return

            # Convert to SSML format
            ssml_captions = []
            for line in lines:
                ssml_captions.append(f"<speak>{line}</speak>")

            # Update captions
            self.captions[idx] = ssml_captions
            messagebox.showinfo("Success", "Caption updated successfully")

    def update_status(self, message, is_error=False):
        """Update status message"""
        self.status_label.config(
            text=message, foreground="red" if is_error else "black"
        )
        self.update_idletasks()  # Force update

    def on_ok(self):
        """Handle OK button click"""
        if len(self.selected_indices) != self.required_amount:
            messagebox.showerror(
                "Selection Error",
                f"Please select exactly {self.required_amount} memes.",
            )
            return

        # If we have captions already, just return the selection
        if all(self.captions[i] for i in self.selected_indices):
            # Get selected memes and captions in the right order
            selected_memes = []
            selected_captions = []

            # Get items in display order
            items = self.tree.get_children()
            for item in items:
                idx = int(item)
                if idx in self.selected_indices:
                    selected_memes.append(self.meme_urls[idx])
                    selected_captions.append(self.captions[idx])

            # Set result and close
            self.result = (selected_memes, selected_captions)
            self.destroy()
        else:
            # We need to generate captions for selected memes
            self._generate_captions_for_selected()

    def _generate_captions_for_selected(self):
        """Generate captions for selected memes using Groq API"""
        # Disable buttons during processing
        self.ok_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.DISABLED)
        self.select_all_btn.config(state=tk.DISABLED)

        # Get selected memes
        selected_memes = [self.meme_urls[i] for i in self.selected_indices]

        # Show status
        self.update_status("Generating captions, please wait...")

        # Start caption generation in background thread
        threading.Thread(
            target=self._caption_thread, args=(selected_memes,), daemon=True
        ).start()

    def _caption_thread(self, selected_memes):
        """Generate captions in a background thread"""
        try:
            # Get the current subreddit from the parent's config panel, or use a default
            current_subreddits = []
            if hasattr(self.parent, 'config_panel') and self.parent.config_panel.subreddit_listbox.get(0):
                current_subreddits = list(self.parent.config_panel.subreddit_listbox.get(0, tk.END))
            
            if not current_subreddits:
                # Use the first meme's author as a fallback subreddit name, or just use "memes"
                if selected_memes and len(selected_memes[0]) > 1 and selected_memes[0][1]:
                    current_subreddits = [selected_memes[0][1]]  # Use author as subreddit
                else:
                    current_subreddits = ["memes"]  # Default fallback
            
            # Fix: Only pass the recognized parameters to Config
            # Store TTS parameters separately for later use
            config = Config(
                subreddits=current_subreddits,
                min_upvotes=self.parent.config_panel.upvotes_var.get(),
                auto_mode=False  # Always use manual mode for selection
            )
            
            # Store TTS parameters for later use with edge_tts
            tts_voice = self.parent.config_panel.voice_var.get()
            tts_rate = self.parent.config_panel.rate_var.get()
            tts_volume = self.parent.config_panel.volume_var.get()
            tts_pitch = self.parent.config_panel.pitch_var.get()

            ai_client = AIClient()
            media_processor = MediaProcessor(config)
            video_gen = GUIVideoGenerator(config, ai_client, media_processor)

            # Generate captions
            captions = video_gen.generate_captions(selected_memes)

            # Update result and close dialog
            def finish():
                self.result = (selected_memes, captions)
                self.destroy()

            self.after(0, finish)

        except Exception as e:
            # Store error message before passing to lambda
            error_message = str(e)
            
            def show_error(msg):
                self.update_status(f"Error: {msg}", True)
                self.ok_btn.config(state=tk.NORMAL)
                self.cancel_btn.config(state=tk.NORMAL)
                self.select_all_btn.config(state=tk.NORMAL)
            
            self.after(0, lambda: show_error(error_message))
# ...existing code...


class MainApplication(tk.Tk):
    """Main application window"""

    def __init__(self):
        super().__init__()

        # Start in full screen
        self.state('zoomed')  # For Windows
        # For Linux/Mac, use:
        # self.attributes('-zoomed', True)

        self.title("Reddit Meme to Video Generator")
        self.geometry("1200x800")
        self.minsize(900, 600)

        # Configure the grid with resizable weights - simplified for better stability
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Create main frame to hold all content
        main_frame = ttk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=3)  # Top section gets more space
        main_frame.rowconfigure(1, weight=1)  # Bottom section gets less space

        # Create a vertical paned window for the main layout
        self.main_vertical_paned = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        self.main_vertical_paned.grid(row=0, column=0, rowspan=2, sticky="nsew")

        # Top section frame
        top_section = ttk.Frame(self.main_vertical_paned)
        self.main_vertical_paned.add(top_section, weight=3)
        
        # Bottom section frame for queue panel
        bottom_section = ttk.Frame(self.main_vertical_paned)
        self.main_vertical_paned.add(bottom_section, weight=1)

        # Create horizontal paned window for the top section
        self.top_horizontal_paned = ttk.PanedWindow(top_section, orient=tk.HORIZONTAL)
        self.top_horizontal_paned.pack(fill=tk.BOTH, expand=True)

        # Left side - Config panel
        config_frame = ttk.Frame(self.top_horizontal_paned)
        self.top_horizontal_paned.add(config_frame, weight=1)
        
        # Right side - Preview panel 
        preview_frame = ttk.Frame(self.top_horizontal_paned)
        self.top_horizontal_paned.add(preview_frame, weight=2)

        # Create the inner panels
        self.config_panel = ConfigPanel(config_frame)
        self.config_panel.pack(fill=tk.BOTH, expand=True)

        self.preview_panel = MemePreviewPanel(preview_frame)
        self.preview_panel.pack(fill=tk.BOTH, expand=True)

        # Bottom queue panel
        self.queue_panel = QueuePanel(bottom_section)
        self.queue_panel.pack(fill=tk.BOTH, expand=True)

        # Create main menu
        self.create_menu()

        # Add action buttons to config panel
        self.action_frame = ttk.Frame(self.config_panel)
        self.action_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.analyze_btn = ttk.Button(
            self.action_frame, text="Analyze Memes", command=self.analyze_memes
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        self.generate_btn = ttk.Button(
            self.action_frame,
            text="Generate Video",
            command=self.start_video_generation,
        )
        self.generate_btn.pack(side=tk.LEFT, padx=5)

        # State variables
        self.analyzed_memes = None
        self.analyzed_captions = None
        self.current_meme_index = 0
        
        # Set initial sash positions after the window is fully created
        self.after(100, self.set_initial_sash_positions)
        
        # Bind resize event to maintain panel visibility
        self.bind("<Configure>", self.on_window_resize)

        # Track layout configuration
        self.layout_config = {
            'window_state': 'zoomed',
            'window_geometry': '',
            'top_horizontal_sash': 0.3,  # 30% for config panel
            'main_vertical_sash': 0.7,   # 70% for top section
        }

        # Load any saved layout
        self.load_layout()
    
    def set_initial_sash_positions(self):
        """Set initial sash positions for the paned windows"""
        # Get window width and height
        width = self.winfo_width()
        height = self.winfo_height()
        
        # Set horizontal paned window sash position (30% for config, 70% for preview)
        self.top_horizontal_paned.sashpos(0, int(width * 0.3))
        
        # Set vertical paned window sash position (70% for top section, 30% for queue)
        self.main_vertical_paned.sashpos(0, int(height * 0.7))
    
    def on_window_resize(self, event):
        """Maintain minimum sizes for panels when resizing"""
        # Only respond to the main window's resize events
        if event.widget == self:
            # Give a short delay to allow the window to finish resizing
            self.after(100, self.adjust_sash_positions)
    
    def adjust_sash_positions(self):
        """Adjust sash positions to maintain minimum sizes after resize"""
        width = self.winfo_width()
        height = self.winfo_height()
        
        # Ensure config panel has at least 25% width
        min_config_width = int(width * 0.25)
        current_config_width = self.top_horizontal_paned.sashpos(0)
        if current_config_width < min_config_width:
            self.top_horizontal_paned.sashpos(0, min_config_width)
        
        # Ensure top section has at least 60% height
        min_top_height = int(height * 0.6)
        current_top_height = self.main_vertical_paned.sashpos(0)
        if current_top_height < min_top_height:
            self.main_vertical_paned.sashpos(0, min_top_height)
            
    def update_captions(self, new_captions):
        """Update captions for the current meme"""
        if (self.analyzed_memes is None or 
            self.current_meme_index < 0 or 
            self.current_meme_index >= len(self.analyzed_captions)):
            return
            
        self.analyzed_captions[self.current_meme_index] = new_captions
        print(f"Updated captions for meme {self.current_meme_index+1}/{len(self.analyzed_memes)}")

    def create_menu(self):
        menu_bar = tk.Menu(self)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New Video", command=self.clear_state)
        file_menu.add_command(
            label="Save Configuration", command=self.config_panel.save_config
        )
        file_menu.add_command(
            label="Load Configuration", command=self.config_panel.load_config
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menu_bar, tearoff=0)
        tools_menu.add_command(label="Check API Key", command=self.check_api_key)
        tools_menu.add_command(label="Clear Cache", command=self.clear_cache)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.open_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menu_bar)

    def clear_state(self):
        """Clear the current state (analyzed memes, etc.)"""
        self.analyzed_memes = None
        self.analyzed_captions = None
        self.current_meme_index = 0

        # Clear preview
        self.preview_panel.set_meme(None, None, None, None)

    def check_api_key(self):
        """Check if the API key is valid"""
        groq_api_key = os.environ.get("GROQ_API_KEY")

        if not groq_api_key:
            messagebox.showerror(
                "API Key Error",
                "GROQ_API_KEY not found in environment variables. Please set it before continuing.",
            )
            return

        # Start a test in background thread
        self.analyze_btn.config(state=tk.DISABLED)
        threading.Thread(
            target=self._test_api_key, args=(groq_api_key,), daemon=True
        ).start()

    def _test_api_key(self, api_key):
        """Test if the API key works in a background thread"""
        try:
            # Try to initialize the client
            ai_client = AIClient(api_key)

            # Make a small test request
            response = ai_client.get_text_completion("Hello, are you working?")

            # Show success message
            self.after(
                0,
                lambda: messagebox.showinfo(
                    "Success", "API key is valid and working correctly."
                ),
            )
        except Exception as e:
            # Store error message before passing to lambda
            error_message = str(e)
            
            self.after(
                0,
                lambda msg=error_message: messagebox.showerror(
                    "API Key Error", f"Error testing API key: {msg}"
                ),
            )
        finally:
            # Re-enable button
            self.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))

    def clear_cache(self):
        """Clear cached files"""
        try:
            # Initialize media processor with default config
            media_processor = MediaProcessor(Config(subreddits=["memes"]))
            media_processor.clean_temp_files()
            messagebox.showinfo("Success", "Cache cleared successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear cache: {str(e)}")

    def open_documentation(self):
        """Open documentation"""
        webbrowser.open("https://github.com/YourUsername/Shorts-Reel-Generator")

    def show_about(self):
        """Show about dialog"""
        about_text = """
        Reddit Meme to Video Generator
        
        A tool for creating short-form videos from Reddit memes
        with AI-generated captions and TTS narration.
        
        Version: 1.0.0
        """
        messagebox.showinfo("About", about_text)

    def analyze_memes(self):
        """Analyze memes from selected subreddit"""
        # Get current config
        config = self.config_panel.get_config()

        # Check if subreddits are selected
        if not config["subreddits"]:
            messagebox.showerror("Error", "Please add at least one subreddit")
            return

        # Get the first subreddit for analysis
        subreddit = config["subreddits"][0]

        # Show processing dialog
        self.analyze_btn.config(state=tk.DISABLED, text="Analyzing...")

        # Start analysis in a background thread
        threading.Thread(
            target=self._analyze_thread,
            args=(
                subreddit,
                config["memes_per_video"],
                config["post_type"],
                config["min_upvotes"],
            ),
            daemon=True,
        ).start()

    def _analyze_thread(self, subreddit, amount, post_type, min_upvotes):
        """Run meme analysis in a background thread with selection capability"""
        try:
            # Initialize components
            config = Config(
                subreddits=[subreddit],
                min_upvotes=min_upvotes,
                auto_mode=False,  # Always use manual mode for GUI
                edge_tts_voice=self.config_panel.voice_var.get(),
                edge_tts_rate=self.config_panel.rate_var.get(),
                edge_tts_volume=self.config_panel.volume_var.get(),
                edge_tts_pitch=self.config_panel.pitch_var.get(),
            )

            ai_client = AIClient()
            media_processor = MediaProcessor(config)

            # Use our GUI-optimized generator to fetch memes WITHOUT captions to save API calls
            video_generator = GUIVideoGenerator(config, ai_client, media_processor)

            # Fetch memes only - no captions yet!
            meme_urls = video_generator.collect_memes(subreddit, amount, post_type)

            # Show selection dialog in the main thread
            def show_selection_dialog():
                # Note: We're passing None for captions since we haven't generated them yet
                dialog = MemeSelectionDialog(self, meme_urls, None, amount)
                if dialog.result:
                    # Use selected memes and captions
                    self.analyzed_memes, self.analyzed_captions = dialog.result
                    self.current_meme_index = 0

                    # Show first meme in preview
                    self._show_meme(0)

                    # Update navigation buttons
                    self._update_nav_buttons()

                    # Make sure the navigation buttons are properly connected
                    self._reconnect_navigation_buttons()
                    
                    # Complete analysis
                    self._analysis_complete()
                else:
                    # User cancelled, reset button
                    self.analyze_btn.config(state=tk.NORMAL, text="Analyze Memes")

            # Run selection dialog in main thread
            self.after(0, show_selection_dialog)

        except Exception as e:
            # Store error message before passing to lambda
            error_message = str(e)
            
            self.after(
                0,
                lambda msg=error_message: messagebox.showerror(
                    "Analysis Error", f"Error analyzing memes: {msg}"
                ),
            )
            self.after(
                0,
                lambda: self.analyze_btn.config(state=tk.NORMAL, text="Analyze Memes"),
            )
            
    def _reconnect_navigation_buttons(self):
        """Ensure navigation buttons are properly connected to their methods"""
        # Explicitly reconnect navigation buttons to their methods
        if hasattr(self.preview_panel, "prev_btn"):
            # Use preview panel's handler methods instead of direct navigation
            # These handlers save captions before navigating
            self.preview_panel.prev_btn.config(command=self.preview_panel._handle_prev_meme)
            
        if hasattr(self.preview_panel, "next_btn"):
            # Use preview panel's handler methods instead of direct navigation
            self.preview_panel.next_btn.config(command=self.preview_panel._handle_next_meme)
        
        # Make sure the preview panel can access the parent methods
        self.preview_panel.parent = self
            
        # Debug print to verify if navigation is enabled
        print(f"Navigation reconnected. Meme count: {len(self.analyzed_memes) if self.analyzed_memes else 0}")
        print(f"Current index: {self.current_meme_index}")

    def _setup_navigation_ui(self, count):
        """Create and configure navigation buttons - REMOVED as it's now built into the preview panel"""
        # Method no longer needed as navigation buttons are part of the preview panel
        pass

    def _analysis_complete(self):
        """Handle completion of meme analysis"""
        self.analyze_btn.config(state=tk.NORMAL, text="Analyze Memes")

    def _show_meme(self, index):
        """Show a specific meme in the preview panel"""
        if not self.analyzed_memes or index < 0 or index >= len(self.analyzed_memes):
            return

        # Get meme and caption info
        meme_url, author, title = self.analyzed_memes[index]
        captions = self.analyzed_captions[index]

        # Update preview panel
        self.preview_panel.set_meme(meme_url, captions, title, author)
        self.preview_panel.set_tts_settings(
            self.config_panel.voice_var.get(),
            self.config_panel.rate_var.get(),
            self.config_panel.volume_var.get(),
            self.config_panel.pitch_var.get(),
        )

        # Store current index
        self.current_meme_index = index

        # Update meme counter directly in preview panel
        if self.analyzed_memes:
            self.preview_panel.meme_counter.config(
                text=f"{index+1}/{len(self.analyzed_memes)}"
            )
        
        # Update navigation buttons
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        """Update navigation button states in preview panel"""
        if not hasattr(self, "analyzed_memes") or not self.analyzed_memes:
            # Disable both buttons if no memes
            self.preview_panel.prev_btn.config(state=tk.DISABLED)
            self.preview_panel.next_btn.config(state=tk.DISABLED)
            return

        # Enable/disable previous button
        if self.current_meme_index <= 0:
            self.preview_panel.prev_btn.config(state=tk.DISABLED)
        else:
            self.preview_panel.prev_btn.config(state=tk.NORMAL)

        # Enable/disable next button
        if self.current_meme_index >= len(self.analyzed_memes) - 1:
            self.preview_panel.next_btn.config(state=tk.DISABLED)
        else:
            self.preview_panel.next_btn.config(state=tk.NORMAL)

    def start_video_generation(self):
        """Start generating a video with current settings"""
        # Get current config
        config_dict = self.config_panel.get_config()

        # Check if subreddits are selected
        if not config_dict["subreddits"]:
            messagebox.showerror("Error", "Please add at least one subreddit")
            return

        # Check if captions have been edited
        if self.analyzed_memes and self.analyzed_captions:
            # Fix: Save current meme's captions before proceeding
            if hasattr(self.preview_panel, "save_caption"):
                self.preview_panel.save_caption()
                
            # Remind to save captions if they might have been edited
            if messagebox.askyesno(
                "Caption Check", 
                "Make sure you've saved any caption edits using the 'Save Caption Changes' button.\n\nContinue with video generation?"
            ) is False:
                return

        # Ask for confirmation before generating
        if not messagebox.askyesno(
            "Confirm", "Start video generation with current settings?"
        ):
            return

        # If we've already analyzed memes, ask if we want to use them
        use_analyzed = False
        if self.analyzed_memes and self.analyzed_captions:
            if messagebox.askyesno(
                "Use Analyzed", "Use the already analyzed memes for video generation?"
            ):
                use_analyzed = True

        # Create output directory if it doesn't exist
        output_dir = config_dict["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # For each subreddit, create a queue item
        for subreddit in config_dict["subreddits"]:
            # Generate unique output folder with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            unique_dir = os.path.join(output_dir, f"{subreddit}_{timestamp}")

            # Create queue item
            queue_item = VideoQueueItem(
                subreddit=subreddit,
                memes_count=config_dict["memes_per_video"],
                post_type=config_dict["post_type"],
                config=config_dict,
                output_dir=unique_dir,
            )

            # If using analyzed memes and it's the first subreddit, add them to the queue item
            if use_analyzed and subreddit == config_dict["subreddits"][0]:
                queue_item.meme_urls = self.analyzed_memes
                queue_item.captions = self.analyzed_captions

            # Add item to queue
            self.queue_panel.add_to_queue(queue_item)

        # Show confirmation
        messagebox.showinfo(
            "Success",
            f"Added {len(config_dict['subreddits'])} video(s) to the generation queue",
        )

    def _prev_meme(self):
        """Show previous meme in preview"""
        if self.analyzed_memes and self.current_meme_index > 0:
            self.current_meme_index -= 1
            self._show_meme(self.current_meme_index)

    def _next_meme(self):
        """Show next meme in preview"""
        if (self.analyzed_memes and 
            self.current_meme_index < len(self.analyzed_memes) - 1):
            self.current_meme_index += 1
            self._show_meme(self.current_meme_index)

    def save_layout(self):
        """Save current layout configuration"""
        try:
            # Get current window state and geometry
            self.layout_config['window_state'] = self.state()
            self.layout_config['window_geometry'] = self.geometry()

            # Get sash positions as ratios of total size
            width = self.winfo_width()
            height = self.winfo_height()
            
            if width > 0 and height > 0:  # Prevent division by zero
                top_sash = self.top_horizontal_paned.sashpos(0)
                main_sash = self.main_vertical_paned.sashpos(0)
                
                self.layout_config['top_horizontal_sash'] = top_sash / width
                self.layout_config['main_vertical_sash'] = main_sash / height

            # Update the main configuration with layout information
            config = self.config_panel.get_config()
            config['layout'] = self.layout_config
            
            return config
        except Exception as e:
            print(f"Error saving layout: {e}")
            return self.config_panel.get_config()

    def load_layout(self, config=None):
        """Load layout configuration"""
        try:
            if config and 'layout' in config:
                self.layout_config = config['layout']
            
            # Apply window state
            if self.layout_config.get('window_state'):
                self.state(self.layout_config['window_state'])
            
            # Apply window geometry if not starting in zoomed state
            if self.layout_config.get('window_geometry') and self.layout_config['window_state'] != 'zoomed':
                self.geometry(self.layout_config['window_geometry'])
            
            # Schedule sash position updates after window is drawn
            self.after(100, self.apply_sash_positions)
        except Exception as e:
            print(f"Error loading layout: {e}")

    def apply_sash_positions(self):
        """Apply saved sash positions"""
        try:
            width = self.winfo_width()
            height = self.winfo_height()
            
            # Apply horizontal sash position
            if 'top_horizontal_sash' in self.layout_config:
                sash_x = int(width * self.layout_config['top_horizontal_sash'])
                self.top_horizontal_paned.sashpos(0, sash_x)
            
            # Apply vertical sash position
            if 'main_vertical_sash' in self.layout_config:
                sash_y = int(height * self.layout_config['main_vertical_sash'])
                self.main_vertical_paned.sashpos(0, sash_y)
        except Exception as e:
            print(f"Error applying sash positions: {e}")


def main():
    """Main entry point for the GUI application"""
    # Check for required environment variables
    required_vars = ["GROQ_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    if missing_vars:
        # Show GUI warning
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showerror(
            "Missing Environment Variables",
            f"The following environment variables are required:\n\n"
            + "\n".join(missing_vars)
            + "\n\nPlease set them before running the application.",
        )
        root.destroy()
        sys.exit(1)

    # Start the application
    app = MainApplication()
    app.mainloop()


if __name__ == "__main__":
    main()