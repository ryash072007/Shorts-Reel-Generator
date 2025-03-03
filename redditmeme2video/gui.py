"""
Reddit Meme to Video Generator GUI

A graphical user interface for the redditmeme2video tool, providing easy configuration 
and video generation features.
"""

import os
import sys
import json
import time
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from redditmeme2video module
from redditmeme2video import (
    AIClient, 
    Config, 
    MediaProcessor, 
    RedditClient, 
    VideoGenerator,
    strip_ssml_tags
)

# Default configuration values
DEFAULT_CONFIG = {
    "subreddits": ["memes", "dankmemes", "HistoryMemes", "wholesomememes"],
    "min_upvotes": 3000,
    "auto_mode": False,
    "upload": False,
    "edge_tts_voice": "en-AU-WilliamNeural",
    "edge_tts_rate": "+15%",
    "edge_tts_volume": "+5%",
    "edge_tts_pitch": "+30Hz",
    "output_dir": "redditmeme2video/output",
    "dark_mode": True,
    "animation_level": "high",
    "use_background_music": True,
    "memes_per_video": 3,
    "post_type": "hot"
}

# Available TTS voices (popular options)
TTS_VOICES = [
    "en-AU-WilliamNeural", 
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-GB-RyanNeural",
    "en-CA-ClaraNeural",
    "en-IE-ConnorNeural"
]

# Popular subreddit suggestions
POPULAR_SUBREDDITS = [
    "memes", "dankmemes", "HistoryMemes", "wholesomememes", "ProgrammerHumor", 
    "me_irl", "funny", "MemeEconomy", "perfectlycutscreams", "hmm", 
    "HolUp", "starterpacks", "technicallythetruth", "ShitPostCrusaders",
    "marvelmemes", "PrequelMemes", "lotrmemes"
]

class ScrollableFrame(ttk.Frame):
    """A scrollable frame widget"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Configure canvas for mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")


class VideoQueueItem:
    """Represents a video in the generation queue"""
    def __init__(self, subreddit: str, memes_count: int, post_type: str, 
                 config: Dict[str, Any], output_dir: str):
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
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoQueueItem":
        """Create from dictionary after deserialization"""
        item = cls(
            data["subreddit"],
            data["memes_count"],
            data["post_type"],
            data["config"],
            data["output_dir"]
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
        
        self.title_label = ttk.Label(self.title_frame, text="Meme Preview", font=("TkDefaultFont", 14, "bold"))
        self.title_label.pack(side=tk.LEFT)
        
        # Preview image area
        self.image_frame = ttk.LabelFrame(self, text="Image")
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.image_frame, bg="#f0f0f0", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Placeholder image
        placeholder_text = "No meme selected for preview"
        self.canvas.create_text(200, 150, text=placeholder_text, fill="gray")
        
        # Caption preview area
        self.caption_frame = ttk.LabelFrame(self, text="Captions")
        self.caption_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.caption_text = scrolledtext.ScrolledText(self.caption_frame, height=5, wrap=tk.WORD)
        self.caption_text.pack(fill=tk.X, expand=True, padx=5, pady=5)
        self.caption_text.insert(tk.END, "No captions to display")
        self.caption_text.config(state=tk.DISABLED)
        
        # Metadata area
        self.meta_frame = ttk.LabelFrame(self, text="Metadata")
        self.meta_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.meta_grid = ttk.Frame(self.meta_frame)
        self.meta_grid.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.meta_grid, text="Title:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.title_value = ttk.Label(self.meta_grid, text="N/A")
        self.title_value.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(self.meta_grid, text="Author:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.author_value = ttk.Label(self.meta_grid, text="N/A")
        self.author_value.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(self.meta_grid, text="URL:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.url_value = ttk.Label(self.meta_grid, text="N/A", cursor="hand2")
        self.url_value.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        self.url_value.bind("<Button-1>", self.open_meme_url)
        
        # Action buttons
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.play_caption_btn = ttk.Button(self.button_frame, text="Play Caption", command=self.play_caption)
        self.play_caption_btn.pack(side=tk.LEFT, padx=5)
        
        self.open_reddit_btn = ttk.Button(self.button_frame, text="Open on Reddit", command=self.open_on_reddit)
        self.open_reddit_btn.pack(side=tk.LEFT, padx=5)
        
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
        self.author_value.config(text=f"u/{self.current_author}" if self.current_author else "N/A")
        self.url_value.config(text=self.current_meme_url)
        
        # Update captions
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
            
        self.caption_text.config(state=tk.DISABLED)
        
        # Download and display the image
        threading.Thread(target=self._load_image, daemon=True).start()
        
    def _load_image(self):
        """Load image in a background thread"""
        try:
            response = requests.get(self.current_meme_url)
            if response.status_code == 200:
                # Load the image from response data
                image = Image.open(BytesIO(response.content))
                
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
                self.after(0, lambda: self._update_image(photo_image, new_width, new_height))
        except Exception as e:
            # Update UI in main thread to show error
            self.after(0, lambda: self._show_image_error(str(e)))
            
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
        self.canvas.create_text(200, 150, text=f"Error loading image:\n{error_msg}", fill="red")
        
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
        threading.Thread(target=self._play_audio, args=(caption_text,), daemon=True).start()
        
    def _play_audio(self, text):
        """Play audio in a background thread"""
        try:
            # Create temp file for audio
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_file.close()
            
            # Generate audio file with edge_tts
            async def generate_audio():
                communicate = edge_tts.Communicate(
                    text,
                    self.tts_voice,
                    rate=self.tts_rate,
                    volume=self.tts_volume,
                    pitch=self.tts_pitch
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
            self.after(0, lambda: self._playback_complete())
            
        except Exception as e:
            # Show error in main thread
            self.after(0, lambda: messagebox.showerror("Playback Error", f"Error playing caption: {str(e)}"))
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
        self.title_label = ttk.Label(main_frame, text="Configuration Settings", font=("TkDefaultFont", 14, "bold"))
        self.title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")
        
        # Subreddit Selection
        ttk.Label(main_frame, text="Subreddit:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        self.subreddit_frame = ttk.Frame(main_frame)
        self.subreddit_frame.grid(row=1, column=1, sticky="we", padx=5, pady=5)
        
        # Combobox with autocomplete for subreddit selection
        self.subreddit_var = tk.StringVar()
        self.subreddit_combo = ttk.Combobox(self.subreddit_frame, 
                                            textvariable=self.subreddit_var,
                                            values=POPULAR_SUBREDDITS)
        self.subreddit_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add button for multi-subreddit list
        self.add_subreddit_btn = ttk.Button(self.subreddit_frame, text="+", width=3, 
                                           command=self.add_subreddit)
        self.add_subreddit_btn.pack(side=tk.LEFT, padx=5)
        
        # Subreddit list display (for batch mode)
        self.subreddit_list_frame = ttk.Frame(main_frame)
        self.subreddit_list_frame.grid(row=2, column=0, columnspan=2, sticky="we", padx=5, pady=5)
        
        self.subreddit_listbox = tk.Listbox(self.subreddit_list_frame, height=5)
        self.subreddit_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.subreddit_scrollbar = ttk.Scrollbar(self.subreddit_list_frame, 
                                                orient="vertical", 
                                                command=self.subreddit_listbox.yview)
        self.subreddit_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.subreddit_listbox.config(yscrollcommand=self.subreddit_scrollbar.set)
        
        # Buttons for list manipulation
        self.list_btn_frame = ttk.Frame(main_frame)
        self.list_btn_frame.grid(row=3, column=0, columnspan=2, sticky="we", padx=5, pady=5)
        
        self.remove_subreddit_btn = ttk.Button(self.list_btn_frame, text="Remove Selected", 
                                              command=self.remove_subreddit)
        self.remove_subreddit_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_subreddits_btn = ttk.Button(self.list_btn_frame, text="Clear All", 
                                              command=self.clear_subreddits)
        self.clear_subreddits_btn.pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(row=4, column=0, columnspan=2, 
                                                           sticky="we", pady=10)
        
        # Basic Settings
        ttk.Label(main_frame, text="Basic Settings", font=("TkDefaultFont", 12, "bold")).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=5, pady=5
        )
        
        # Memes per video
        ttk.Label(main_frame, text="Memes per video:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        
        self.memes_var = tk.IntVar(value=DEFAULT_CONFIG["memes_per_video"])
        self.memes_spinbox = ttk.Spinbox(main_frame, from_=1, to=10, textvariable=self.memes_var, 
                                        width=5)
        self.memes_spinbox.grid(row=6, column=1, sticky="w", padx=5, pady=5)
        
        # Minimum upvotes
        ttk.Label(main_frame, text="Minimum upvotes:").grid(row=7, column=0, sticky="w", padx=5, pady=5)
        
        self.upvotes_var = tk.IntVar(value=DEFAULT_CONFIG["min_upvotes"])
        self.upvotes_spinbox = ttk.Spinbox(main_frame, from_=100, to=100000, 
                                          increment=100, textvariable=self.upvotes_var,
                                          width=10)
        self.upvotes_spinbox.grid(row=7, column=1, sticky="w", padx=5, pady=5)
        
        # Post type
        ttk.Label(main_frame, text="Post type:").grid(row=8, column=0, sticky="w", padx=5, pady=5)
        
        self.post_type_var = tk.StringVar(value=DEFAULT_CONFIG["post_type"])
        self.post_type_combo = ttk.Combobox(main_frame, textvariable=self.post_type_var, 
                                           values=["hot", "top", "new", "rising"], 
                                           state="readonly", width=10)
        self.post_type_combo.grid(row=8, column=1, sticky="w", padx=5, pady=5)
        
        # Auto mode
        self.auto_mode_var = tk.BooleanVar(value=DEFAULT_CONFIG["auto_mode"])
        self.auto_mode_check = ttk.Checkbutton(main_frame, text="Auto mode (no manual selection)", 
                                              variable=self.auto_mode_var)
        self.auto_mode_check.grid(row=9, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Background music
        self.bg_music_var = tk.BooleanVar(value=DEFAULT_CONFIG["use_background_music"])
        self.bg_music_check = ttk.Checkbutton(main_frame, text="Include background music", 
                                             variable=self.bg_music_var)
        self.bg_music_check.grid(row=10, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(row=11, column=0, columnspan=2, 
                                                           sticky="we", pady=10)
        
        # Voice Settings
        ttk.Label(main_frame, text="TTS Voice Settings", font=("TkDefaultFont", 12, "bold")).grid(
            row=12, column=0, columnspan=2, sticky="w", padx=5, pady=5
        )
        
        # Voice selection
        ttk.Label(main_frame, text="Voice:").grid(row=13, column=0, sticky="w", padx=5, pady=5)
        
        self.voice_var = tk.StringVar(value=DEFAULT_CONFIG["edge_tts_voice"])
        self.voice_combo = ttk.Combobox(main_frame, textvariable=self.voice_var, 
                                       values=TTS_VOICES, state="readonly")
        self.voice_combo.grid(row=13, column=1, sticky="we", padx=5, pady=5)
        
        # Rate
        ttk.Label(main_frame, text="Rate:").grid(row=14, column=0, sticky="w", padx=5, pady=5)
        
        self.rate_var = tk.StringVar(value=DEFAULT_CONFIG["edge_tts_rate"])
        self.rate_combo = ttk.Combobox(main_frame, textvariable=self.rate_var, 
                                      values=["-20%", "-10%", "+0%", "+10%", "+15%", "+20%", "+30%"])
        self.rate_combo.grid(row=14, column=1, sticky="we", padx=5, pady=5)
        
        # Volume
        ttk.Label(main_frame, text="Volume:").grid(row=15, column=0, sticky="w", padx=5, pady=5)
        
        self.volume_var = tk.StringVar(value=DEFAULT_CONFIG["edge_tts_volume"])
        self.volume_combo = ttk.Combobox(main_frame, textvariable=self.volume_var, 
                                        values=["-20%", "-10%", "+0%", "+5%", "+10%", "+20%"])
        self.volume_combo.grid(row=15, column=1, sticky="we", padx=5, pady=5)
        
        # Pitch
        ttk.Label(main_frame, text="Pitch:").grid(row=16, column=0, sticky="w", padx=5, pady=5)
        
        self.pitch_var = tk.StringVar(value=DEFAULT_CONFIG["edge_tts_pitch"])
        self.pitch_combo = ttk.Combobox(main_frame, textvariable=self.pitch_var, 
                                       values=["-20Hz", "-10Hz", "+0Hz", "+10Hz", "+20Hz", "+30Hz"])
        self.pitch_combo.grid(row=16, column=1, sticky="we", padx=5, pady=5)
        
        # Test voice button
        self.test_voice_btn = ttk.Button(main_frame, text="Test Voice", command=self.test_voice)
        self.test_voice_btn.grid(row=17, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(row=18, column=0, columnspan=2, 
                                                           sticky="we", pady=10)
        
        # Appearance Settings
        ttk.Label(main_frame, text="Appearance Settings", font=("TkDefaultFont", 12, "bold")).grid(
            row=19, column=0, columnspan=2, sticky="w", padx=5, pady=5
        )
        
        # Dark mode
        self.dark_mode_var = tk.BooleanVar(value=DEFAULT_CONFIG["dark_mode"])
        self.dark_mode_check = ttk.Checkbutton(main_frame, text="Dark mode theme", 
                                              variable=self.dark_mode_var)
        self.dark_mode_check.grid(row=20, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Animation level
        ttk.Label(main_frame, text="Animation:").grid(row=21, column=0, sticky="w", padx=5, pady=5)
        
        self.animation_var = tk.StringVar(value=DEFAULT_CONFIG["animation_level"])
        self.animation_combo = ttk.Combobox(main_frame, textvariable=self.animation_var, 
                                           values=["low", "medium", "high"], 
                                           state="readonly", width=10)
        self.animation_combo.grid(row=21, column=1, sticky="w", padx=5, pady=5)
        
        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(row=22, column=0, columnspan=2, 
                                                           sticky="we", pady=10)
        
        # Output Settings
        ttk.Label(main_frame, text="Output Settings", font=("TkDefaultFont", 12, "bold")).grid(
            row=23, column=0, columnspan=2, sticky="w", padx=5, pady=5
        )
        
        # Output directory
        ttk.Label(main_frame, text="Output directory:").grid(row=24, column=0, sticky="w", padx=5, pady=5)
        
        self.output_dir_frame = ttk.Frame(main_frame)
        self.output_dir_frame.grid(row=24, column=1, sticky="we", padx=5, pady=5)
        
        self.output_dir_var = tk.StringVar(value=DEFAULT_CONFIG["output_dir"])
        self.output_dir_entry = ttk.Entry(self.output_dir_frame, textvariable=self.output_dir_var)
        self.output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.browse_btn = ttk.Button(self.output_dir_frame, text="Browse", command=self.browse_output_dir)
        self.browse_btn.pack(side=tk.RIGHT, padx=5)
        
        # Auto upload to YouTube
        self.upload_var = tk.BooleanVar(value=DEFAULT_CONFIG["upload"])
        self.upload_check = ttk.Checkbutton(main_frame, text="Auto upload to YouTube (requires setup)", 
                                           variable=self.upload_var)
        self.upload_check.grid(row=25, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Configuration buttons
        self.config_btn_frame = ttk.Frame(main_frame)
        self.config_btn_frame.grid(row=26, column=0, columnspan=2, sticky="we", padx=5, pady=10)
        
        self.save_config_btn = ttk.Button(self.config_btn_frame, text="Save Config", 
                                         command=self.save_config)
        self.save_config_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_config_btn = ttk.Button(self.config_btn_frame, text="Load Config", 
                                         command=self.load_config)
        self.load_config_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_config_btn = ttk.Button(self.config_btn_frame, text="Reset to Default", 
                                          command=self.reset_config)
        self.reset_config_btn.pack(side=tk.RIGHT, padx=5)
        
        # Load subreddits from default config
        for subreddit in self.config["subreddits"]:
            self.subreddit_listbox.insert(tk.END, subreddit)
            
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
        sample_text = "This is a test of the text-to-speech voice settings. How does it sound?"
        
        self.test_voice_btn.config(state=tk.DISABLED, text="Generating...")
        
        # Start generation in a background thread
        threading.Thread(target=self._generate_test_audio, args=(sample_text, voice, rate, volume, pitch), 
                        daemon=True).start()
        
    def _generate_test_audio(self, text, voice, rate, volume, pitch):
        """Generate and play test audio in a background thread"""
        try:
            # Create temp file for audio
            import tempfile
            import pygame
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_file.close()
            
            # Generate audio file with edge_tts
            async def generate_audio():
                communicate = edge_tts.Communicate(
                    text,
                    voice,
                    rate=rate,
                    volume=volume,
                    pitch=pitch
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
            self.parent.after(0, lambda: self.test_voice_btn.config(state=tk.NORMAL, text="Test Voice"))
            
        except Exception as e:
            # Show error in main thread
            self.parent.after(0, lambda: messagebox.showerror("Voice Test Error", 
                                                           f"Error testing voice: {str(e)}"))
            self.parent.after(0, lambda: self.test_voice_btn.config(state=tk.NORMAL, text="Test Voice"))
            
    def browse_output_dir(self):
        """Open directory browser for output directory selection"""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)
            
    def save_config(self):
        """Save current configuration to a file"""
        # Get current configuration
        self.update_config()
        
        # Ask for file location
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(__file__))
        )
        
        if not filepath:
            return
            
        try:
            # Write to file
            with open(filepath, "w") as f:
                json.dump(self.config, f, indent=2)
                
            messagebox.showinfo("Success", f"Configuration saved to {filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
        
    def load_config(self):
        """Load configuration from a file"""
        # Ask for file location
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(__file__))
        )
        
        if not filepath:
            return
            
        try:
            # Read from file
            with open(filepath, "r") as f:
                config = json.load(f)
                
            # Apply configuration
            self.apply_config(config)
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
        self.memes_var.set(config.get("memes_per_video", DEFAULT_CONFIG["memes_per_video"]))
        self.upvotes_var.set(config.get("min_upvotes", DEFAULT_CONFIG["min_upvotes"]))
        self.post_type_var.set(config.get("post_type", DEFAULT_CONFIG["post_type"]))
        self.auto_mode_var.set(config.get("auto_mode", DEFAULT_CONFIG["auto_mode"]))
        self.bg_music_var.set(config.get("use_background_music", DEFAULT_CONFIG["use_background_music"]))
        self.voice_var.set(config.get("edge_tts_voice", DEFAULT_CONFIG["edge_tts_voice"]))
        self.rate_var.set(config.get("edge_tts_rate", DEFAULT_CONFIG["edge_tts_rate"]))
        self.volume_var.set(config.get("edge_tts_volume", DEFAULT_CONFIG["edge_tts_volume"]))
        self.pitch_var.set(config.get("edge_tts_pitch", DEFAULT_CONFIG["edge_tts_pitch"]))
        self.dark_mode_var.set(config.get("dark_mode", DEFAULT_CONFIG["dark_mode"]))
        self.animation_var.set(config.get("animation_level", DEFAULT_CONFIG["animation_level"]))
        self.output_dir_var.set(config.get("output_dir", DEFAULT_CONFIG["output_dir"]))
        self.upload_var.set(config.get("upload", DEFAULT_CONFIG["upload"]))
        
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
            "upload": self.upload_var.get()
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
        
        # Create UI elements
        self.create_widgets()
        
        # Check queue periodically
        self.after(1000, self.check_queue)
        
    def create_widgets(self):
        """Create all UI widgets for the queue panel"""
        # Title bar
        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.title_label = ttk.Label(self.title_frame, text="Video Generation Queue", 
                                    font=("TkDefaultFont", 14, "bold"))
        self.title_label.pack(side=tk.LEFT)
        
        # Queue list
        self.queue_frame = ttk.Frame(self)
        self.queue_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Column headers
        self.header_frame = ttk.Frame(self.queue_frame)
        self.header_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Label(self.header_frame, text="Subreddit", width=15).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(self.header_frame, text="Status", width=10).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(self.header_frame, text="Progress", width=10).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(self.header_frame, text="Actions", width=15).pack(side=tk.LEFT, padx=(0, 10))
        
        # Queue items will be added here
        self.queue_list_frame = ScrollableFrame(self.queue_frame)
        self.queue_list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Control panel
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.clear_completed_btn = ttk.Button(self.control_frame, text="Clear Completed", 
                                            command=self.clear_completed)
        self.clear_completed_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_all_btn = ttk.Button(self.control_frame, text="Clear All", 
                                       command=self.clear_all)
        self.clear_all_btn.pack(side=tk.LEFT, padx=5)
        
        # Concurrent tasks setting
        ttk.Label(self.control_frame, text="Concurrent Tasks:").pack(side=tk.LEFT, padx=(15, 5))
        
        self.concurrent_var = tk.IntVar(value=self.max_concurrent_tasks)
        self.concurrent_spinbox = ttk.Spinbox(self.control_frame, from_=1, to=4, 
                                             textvariable=self.concurrent_var, 
                                             width=5, command=self.update_concurrent_tasks)
        self.concurrent_spinbox.pack(side=tk.LEFT)
        
    def update_concurrent_tasks(self):
        """Update the maximum number of concurrent tasks"""
        try:
            self.max_concurrent_tasks = int(self.concurrent_var.get())
            self.check_queue()  # Check queue immediately to start new tasks if needed
        except:
            pass
    
    def add_to_queue(self, queue_item):
        """Add a new item to the queue"""
        self.queue.append(queue_item)
        self.update_queue_display()
        self.check_queue()  # Check if we can start processing
        
    def update_queue_display(self):
        """Update the queue display with current items"""
        # Clear existing items
        for widget in self.queue_list_frame.scrollable_frame.winfo_children():
            widget.destroy()
            
        # Add each queue item to the display
        for idx, item in enumerate(self.queue):
            item_frame = ttk.Frame(self.queue_list_frame.scrollable_frame)
            item_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
            
            # Subreddit name
            subreddit_label = ttk.Label(item_frame, text=f"r/{item.subreddit}", width=15)
            subreddit_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # Status
            status_label = ttk.Label(item_frame, text=item.status.title(), width=10)
            status_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # Progress
            if item.status == "processing":
                progress_var = tk.IntVar(value=item.progress)
                progress_bar = ttk.Progressbar(item_frame, variable=progress_var, 
                                              length=100, mode="determinate")
                progress_bar.pack(side=tk.LEFT, padx=(0, 10))
                # Store progress var reference
                item_frame.progress_var = progress_var
            else:
                placeholder = ttk.Frame(item_frame, width=100)
                placeholder.pack(side=tk.LEFT, padx=(0, 10))
                
            # Action buttons based on status
            actions_frame = ttk.Frame(item_frame)
            actions_frame.pack(side=tk.LEFT, padx=(0, 10))
            
            if item.status == "queued":
                # Cancel button for queued items
                ttk.Button(actions_frame, text="Cancel", 
                          command=lambda i=idx: self.cancel_item(i)).pack(side=tk.LEFT, padx=2)
            elif item.status == "processing":
                # Can't cancel while processing
                ttk.Label(actions_frame, text="Processing...").pack(side=tk.LEFT, padx=2)
            elif item.status == "completed":
                # Open video button for completed items
                ttk.Button(actions_frame, text="Open", 
                         command=lambda p=item.output_path: self.open_video(p)).pack(side=tk.LEFT, padx=2)
                # Open folder button
                ttk.Button(actions_frame, text="Folder", 
                         command=lambda p=item.output_path: self.open_folder(p)).pack(side=tk.LEFT, padx=2)
            elif item.status == "failed":
                # Retry button for failed items
                ttk.Button(actions_frame, text="Retry", 
                         command=lambda i=idx: self.retry_item(i)).pack(side=tk.LEFT, padx=2)
            
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
            if os.name == 'nt':  # Windows
                os.startfile(path)
            else:  # macOS or Linux
                try:
                    import subprocess
                    subprocess.call(('xdg-open' if os.name == 'posix' else 'open', path))
                except:
                    messagebox.showinfo("Info", f"Video file is located at: {path}")
        else:
            messagebox.showwarning("Warning", "Video file not found")
            
    def open_folder(self, path):
        """Open the folder containing the video"""
        if path:
            folder = os.path.dirname(path)
            if os.path.exists(folder):
                if os.name == 'nt':  # Windows
                    os.startfile(folder)
                else:  # macOS or Linux
                    try:
                        import subprocess
                        subprocess.call(('xdg-open' if os.name == 'posix' else 'open', folder))
                    except:
                        messagebox.showinfo("Info", f"Folder path is: {folder}")
            else:
                messagebox.showwarning("Warning", "Folder not found")
    
    def clear_completed(self):
        """Clear completed items from the queue"""
        self.queue = [item for item in self.queue if item.status not in ("completed", "failed")]
        self.update_queue_display()
        
    def clear_all(self):
        """Clear all items from the queue"""
        if messagebox.askyesno("Confirm", "Clear entire queue? Processing items cannot be stopped."):
            self.queue = [item for item in self.queue if item.status == "processing"]
            self.update_queue_display()
            
    def check_queue(self):
        """Check queue for items that need processing"""
        # Count active tasks
        self.active_tasks = sum(1 for item in self.queue if item.status == "processing")
        
        # If we can run more tasks, find the next queued item
        if self.active_tasks < self.max_concurrent_tasks:
            for item in self.queue:
                if item.status == "queued":
                    self.process_item(item)
                    break
        
        # Schedule next check
        self.after(1000, self.check_queue)
        
    def process_item(self, item):
        """Process a queue item (generate video)"""
        item.status = "processing"
        self.update_queue_display()
        
        # Start processing in a background thread
        thread = threading.Thread(target=self._generate_video_thread, args=(item,), daemon=True)
        thread.start()
        
    def _generate_video_thread(self, item):
        """Worker thread for video generation"""
        try:
            # Initialize needed components
            config = Config(**item.config)
            ai_client = AIClient()
            media_processor = MediaProcessor(config)
            video_generator = VideoGenerator(config, ai_client, media_processor)
            
            # Update progress periodically
            self.start_progress_updates(item)
            
            # Generate video
            metadata = video_generator.generate_video(
                subreddit=item.subreddit,
                amount=item.memes_count,
                post_type=item.post_type,
                output_location=item.output_dir
            )
            
            # Wait for all processing to complete 
            video_generator.wait_for_completion()
            
            # Video is complete, update item
            output_path = list(metadata.keys())[0] if metadata else None
            item.output_path = output_path
            item.metadata = metadata
            item.status = "completed"
            item.progress = 100
            
            # Update UI
            self.parent.after(0, self.update_queue_display)
            
        except Exception as e:
            # Handle errors
            print(f"Error generating video: {str(e)}")
            item.status = "failed"
            
            # Update UI
            self.parent.after(0, self.update_queue_display)
            self.parent.after(0, lambda: messagebox.showerror("Generation Error", 
                                                           f"Error generating video for r/{item.subreddit}: {str(e)}"))
    
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
                item_frame = self.queue_list_frame.scrollable_frame.winfo_children()[idx]
                if hasattr(item_frame, 'progress_var') and item.status == "processing":
                    item_frame.progress_var.set(item.progress)
            except (IndexError, AttributeError):
                pass


class MainApplication(tk.Tk):
    """Main application window"""
    def __init__(self):
        super().__init__()
        
        self.title("Reddit Meme to Video Generator")
        self.geometry("1200x800")
        self.minsize(900, 600)
        
        # Configure the grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=1)
        
        # Create panels
        self.config_panel = ConfigPanel(self)
        self.config_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.preview_panel = MemePreviewPanel(self)
        self.preview_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.queue_panel = QueuePanel(self)
        self.queue_panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        # Create main menu
        self.create_menu()
        
        # Add action buttons below config panel
        self.action_frame = ttk.Frame(self.config_panel)
        self.action_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        self.analyze_btn = ttk.Button(self.action_frame, text="Analyze Memes",
                                    command=self.analyze_memes)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.generate_btn = ttk.Button(self.action_frame, text="Generate Video",
                                     command=self.start_video_generation)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        # State variables
        self.analyzed_memes = None
        self.analyzed_captions = None
        self.current_meme_index = 0
        
    def create_menu(self):
        """Create the application menu"""
        menu_bar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New Video", command=self.clear_state)
        file_menu.add_command(label="Save Configuration", command=self.config_panel.save_config)
        file_menu.add_command(label="Load Configuration", command=self.config_panel.load_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menu_bar, tearoff=0)
        tools_menu.add_command(label="Check API Key", command=self.check_api_key)
        tools_menu.add_command(label="Clear Cache", command=self.clear_cache)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
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
            messagebox.showerror("API Key Error", 
                               "GROQ_API_KEY not found in environment variables. Please set it before continuing.")
            return
            
        # Start a test in background thread
        self.analyze_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._test_api_key, args=(groq_api_key,), daemon=True).start()
        
    def _test_api_key(self, api_key):
        """Test if the API key works in a background thread"""
        try:
            # Try to initialize the client
            ai_client = AIClient(api_key)
            
            # Make a small test request
            response = ai_client.get_text_completion("Hello, are you working?")
            
            # Show success message
            self.after(0, lambda: messagebox.showinfo("Success", 
                                                    "API key is valid and working correctly."))
        except Exception as e:
            # Show error message
            self.after(0, lambda: messagebox.showerror("API Key Error", 
                                                     f"Error testing API key: {str(e)}"))
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
        threading.Thread(target=self._analyze_thread, 
                        args=(subreddit, config["memes_per_video"], 
                             config["post_type"], config["min_upvotes"]), 
                        daemon=True).start()
    
    def _analyze_thread(self, subreddit, amount, post_type, min_upvotes):
        """Run meme analysis in a background thread"""
        try:
            # Initialize components
            config = Config(
                subreddits=[subreddit],
                min_upvotes=min_upvotes,
                auto_mode=True,  # Use auto mode for preview
                edge_tts_voice=self.config_panel.voice_var.get(),
                edge_tts_rate=self.config_panel.rate_var.get(),
                edge_tts_volume=self.config_panel.volume_var.get(),
                edge_tts_pitch=self.config_panel.pitch_var.get()
            )
            
            ai_client = AIClient()
            media_processor = MediaProcessor(config)
            video_generator = VideoGenerator(config, ai_client, media_processor)
            
            # Fetch and analyze memes
            meme_urls, captions = video_generator.collect_captions(subreddit, amount, post_type)
            
            # Store analysis results 
            self.analyzed_memes = meme_urls
            self.analyzed_captions = captions
            self.current_meme_index = 0
            
            # Show first meme in preview
            self._show_meme(0)
            
            # Update UI in main thread
            self.after(0, lambda: self._analysis_complete(len(meme_urls)))
            
        except Exception as e:
            # Show error in main thread
            self.after(0, lambda: messagebox.showerror("Analysis Error", 
                                                     f"Error analyzing memes: {str(e)}"))
            self.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL, text="Analyze Memes"))
    
    def _analysis_complete(self, count):
        """Handle completion of meme analysis"""
        self.analyze_btn.config(state=tk.NORMAL, text="Analyze Memes")
        
        # Add navigation buttons to preview panel if multiple memes
        if count > 1:
            if not hasattr(self.preview_panel, 'nav_frame'):
                self.preview_panel.nav_frame = ttk.Frame(self.preview_panel)
                self.preview_panel.nav_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
                
                self.preview_panel.prev_btn = ttk.Button(self.preview_panel.nav_frame, 
                                                       text="Previous", 
                                                       command=self._prev_meme)
                self.preview_panel.prev_btn.pack(side=tk.LEFT, padx=5)
                
                self.preview_panel.next_btn = ttk.Button(self.preview_panel.nav_frame, 
                                                       text="Next", 
                                                       command=self._next_meme)
                self.preview_panel.next_btn.pack(side=tk.LEFT, padx=5)
                
                self.preview_panel.meme_counter = ttk.Label(self.preview_panel.nav_frame, 
                                                          text=f"1/{count}")
                self.preview_panel.meme_counter.pack(side=tk.LEFT, padx=10)
            else:
                self.preview_panel.meme_counter.config(text=f"1/{count}")
                
            # Update button states
            self._update_nav_buttons()
    
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
            self.config_panel.pitch_var.get()
        )
        
        # Update meme counter if it exists
        if hasattr(self.preview_panel, 'meme_counter'):
            self.preview_panel.meme_counter.config(
                text=f"{index+1}/{len(self.analyzed_memes)}"
            )
    
    def _prev_meme(self):
        """Show previous meme in preview"""
        if self.current_meme_index > 0:
            self.current_meme_index -= 1
            self._show_meme(self.current_meme_index)
            self._update_nav_buttons()
    
    def _next_meme(self):
        """Show next meme in preview"""
        if self.analyzed_memes and self.current_meme_index < len(self.analyzed_memes) - 1:
            self.current_meme_index += 1
            self._show_meme(self.current_meme_index)
            self._update_nav_buttons()
    
    def _update_nav_buttons(self):
        """Update navigation button states"""
        if not hasattr(self.preview_panel, 'prev_btn') or not hasattr(self.preview_panel, 'next_btn'):
            return
            
        # Enable/disable previous button
        if self.current_meme_index <= 0:
            self.preview_panel.prev_btn.config(state=tk.DISABLED)
        else:
            self.preview_panel.prev_btn.config(state=tk.NORMAL)
            
        # Enable/disable next button
        if not self.analyzed_memes or self.current_meme_index >= len(self.analyzed_memes) - 1:
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
            
        # Ask for confirmation before generating
        if not messagebox.askyesno("Confirm", "Start video generation with current settings?"):
            return
            
        # If we've already analyzed memes, ask if we want to use them
        use_analyzed = False
        if self.analyzed_memes and self.analyzed_captions:
            if messagebox.askyesno("Use Analyzed", 
                                "Use the already analyzed memes for video generation?"):
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
                output_dir=unique_dir
            )
            
            # If using analyzed memes and it's the first subreddit, add them to the queue item
            if use_analyzed and subreddit == config_dict["subreddits"][0]:
                queue_item.meme_urls = self.analyzed_memes
                queue_item.captions = self.analyzed_captions
                
            # Add item to queue
            self.queue_panel.add_to_queue(queue_item)
            
        # Show confirmation
        messagebox.showinfo("Success", 
                          f"Added {len(config_dict['subreddits'])} video(s) to the generation queue")


def main():
    """Main entry point for the GUI application"""
    # Check for required environment variables
    required_vars = ["GROQ_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        # Show GUI warning
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showerror("Missing Environment Variables",
                           f"The following environment variables are required:\n\n" +
                           "\n".join(missing_vars) + 
                           "\n\nPlease set them before running the application.")
        root.destroy()
        sys.exit(1)
        
    # Start the application
    app = MainApplication()
    app.mainloop()


if __name__ == "__main__":
    main()