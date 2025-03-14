"""
SSML Caption Editor GUI

This module provides a user-friendly interface for editing SSML captions.
It handles the complex SSML tags while giving users simple controls.
"""

import re
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import pygame  # Replace playsound with pygame
from typing import List, Callable, Optional
import edge_tts  # Add edge_tts for audio preview
import asyncio  # Add asyncio for edge_tts
import tempfile
import os
import time
from pathlib import Path
import hashlib  # Add this import at the top
# Remove ElevenLabs imports
# from elevenlabs import VoiceSettings
# from elevenlabs.client import ElevenLabs


class SSMLTag:
    """Representation of an SSML tag with its properties."""
    def __init__(self, name: str, attributes: dict = None, content: str = None):
        self.name = name
        self.attributes = attributes or {}
        self.content = content
        
    def to_string(self) -> str:
        """Convert tag to SSML string format."""
        if not self.name:
            return self.content or ""
            
        attr_str = " ".join([f'{k}="{v}"' for k, v in self.attributes.items()])
        attr_str = " " + attr_str if attr_str else ""
        
        if self.content is not None:
            return f"<{self.name}{attr_str}>{self.content}</{self.name}>"
        else:
            return f"<{self.name}{attr_str}/>"


class SSMLParser:
    """Parse and manipulate SSML text."""
    @staticmethod
    def extract_content(ssml: str) -> str:
        """Extract plain text content from SSML string."""
        # Remove all SSML tags to get plain text
        plain_text = re.sub(r'<[^>]+>', '', ssml)
        return plain_text
    
    @staticmethod
    def parse_ssml(ssml: str) -> List[SSMLTag]:
        """Parse SSML string into a list of tag objects."""
        # This is a simplified parser - a real one would be more complex
        tags = []
        
        # Clean up the SSML string
        ssml = ssml.strip()
        
        # Remove outer <speak> tags if present
        if ssml.startswith("<speak>") and ssml.endswith("</speak>"):
            ssml = ssml[7:-8].strip()
            
        # Find all prosody and break tags
        pattern = r'<(prosody|break|emphasis)([^>]*)>(.*?)</\1>|<(break)([^/]*)/>|([^<]+)'
        matches = re.findall(pattern, ssml, re.DOTALL)
        
        for match in matches:
            if match[0] or match[3]:  # Found tag
                tag_name = match[0] or match[3]
                attrs_str = match[1] or match[4]
                content = match[2] if match[0] else None
                
                # Parse attributes
                attrs = {}
                attr_matches = re.findall(r'(\w+)=["\']([^"\']+)["\']', attrs_str)
                for attr_name, attr_value in attr_matches:
                    attrs[attr_name] = attr_value
                
                tags.append(SSMLTag(tag_name, attrs, content))
            elif match[5]:  # Plain text
                tags.append(SSMLTag("", content=match[5]))
                
        return tags
    
    @staticmethod
    def generate_ssml(tags: List[SSMLTag]) -> str:
        """Generate SSML string from tag objects."""
        ssml = "".join(tag.to_string() for tag in tags)
        return f"<speak>{ssml}</speak>"


class SSMLEditorGUI:
    """GUI for editing SSML captions."""
    
    def __init__(self, master, captions: List[str], voice_id="en-AU-WilliamNeural", 
                 rate=None, volume=None, pitch=None):
        """Initialize the editor window."""
        self.master = master
        self.master.title("SSML Caption Editor")
        self.master.geometry("800x600")
        self.captions = captions.copy()
        self.current_index = 0
        self.audio_file = None
        self.play_thread = None
        self.voice_id = voice_id
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Store Edge TTS parameters
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        
        # Add tracking for previewed and edited captions
        self.previewed_captions = {}  # hash -> audio_path
        self.edited_captions = set()  # indices that were edited
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Initialize event loop for edge_tts
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        # Create GUI elements
        self._create_widgets()
        
        # Load initial caption
        if self.captions:
            self._load_caption(0)
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Caption navigation
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        self.caption_label = ttk.Label(nav_frame, text="Caption 1 of 1")
        self.caption_label.pack(side=tk.LEFT, padx=5)
        
        self.prev_btn = ttk.Button(nav_frame, text="Previous", command=self._prev_caption)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = ttk.Button(nav_frame, text="Next", command=self._next_caption)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        self.done_btn = ttk.Button(nav_frame, text="Save All", command=self._save_and_exit)
        self.done_btn.pack(side=tk.RIGHT, padx=5)
        
        # Editing area
        edit_frame = ttk.LabelFrame(main_frame, text="Edit SSML", padding="10")
        edit_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Plain text preview
        preview_frame = ttk.Frame(edit_frame)
        preview_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(preview_frame, text="Preview:").pack(side=tk.LEFT)
        self.preview_label = ttk.Label(preview_frame, text="", font=("TkDefaultFont", 10, "italic"))
        self.preview_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # SSML editing area
        self.ssml_text = scrolledtext.ScrolledText(edit_frame, height=10, wrap=tk.WORD)
        self.ssml_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # SSML helper buttons
        helpers_frame = ttk.Frame(edit_frame)
        helpers_frame.pack(fill=tk.X, pady=5)
        
        # Add SSML tag buttons
        self.add_prosody_btn = ttk.Button(helpers_frame, text="Add Prosody", 
                                         command=lambda: self._add_tag_dialog("prosody"))
        self.add_prosody_btn.pack(side=tk.LEFT, padx=2)
        
        self.add_break_btn = ttk.Button(helpers_frame, text="Add Break", 
                                       command=lambda: self._add_tag_dialog("break"))
        self.add_break_btn.pack(side=tk.LEFT, padx=2)
        
        self.add_emphasis_btn = ttk.Button(helpers_frame, text="Add Emphasis", 
                                          command=lambda: self._add_tag_dialog("emphasis"))
        self.add_emphasis_btn.pack(side=tk.LEFT, padx=2)
        
        # TTS Preview button
        self.preview_btn = ttk.Button(helpers_frame, text="Play Preview", command=self._play_preview)
        self.preview_btn.pack(side=tk.RIGHT, padx=2)
        
        # Update caption when text is edited
        self.ssml_text.bind("<KeyRelease>", self._update_preview)
    
    def _load_caption(self, index: int):
        """Load caption at the specified index."""
        if 0 <= index < len(self.captions):
            self.current_index = index
            self.caption_label.config(text=f"Caption {index+1} of {len(self.captions)}")
            
            # Fill editor with caption
            self.ssml_text.delete("1.0", tk.END)
            self.ssml_text.insert("1.0", self.captions[index])
            
            # Update preview
            self._update_preview()
            
            # Update nav button states
            self.prev_btn.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL if index < len(self.captions)-1 else tk.DISABLED)
    
    def _update_preview(self, event=None):
        """Update the plain text preview."""
        ssml = self.ssml_text.get("1.0", tk.END).strip()
        plain_text = SSMLParser.extract_content(ssml)
        
        # Truncate preview if too long
        if len(plain_text) > 50:
            plain_text = plain_text[:47] + "..."
            
        self.preview_label.config(text=plain_text)
        
        # Update the caption in the list and mark as edited
        if 0 <= self.current_index < len(self.captions):
            if self.captions[self.current_index] != ssml:
                self.edited_captions.add(self.current_index)
                self.captions[self.current_index] = ssml
    
    def _prev_caption(self):
        """Go to previous caption."""
        if self.current_index > 0:
            self._load_caption(self.current_index - 1)
    
    def _next_caption(self):
        """Go to next caption."""
        if self.current_index < len(self.captions) - 1:
            self._load_caption(self.current_index + 1)
    
    def _save_and_exit(self):
        """Save all captions and close the editor."""
        self._update_preview()
        self.master.destroy()
    
    def _add_tag_dialog(self, tag_type: str):
        """Show dialog for adding a specific SSML tag."""
        dialog = tk.Toplevel(self.master)
        dialog.title(f"Add {tag_type.title()} Tag")
        dialog.geometry("400x300")
        dialog.transient(self.master)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        if tag_type == "prosody":
            # Prosody options
            ttk.Label(main_frame, text="Rate:").grid(row=0, column=0, sticky=tk.W, pady=5)
            rate_var = tk.StringVar(value="medium")
            rate_combo = ttk.Combobox(main_frame, textvariable=rate_var,
                                    values=["x-slow", "slow", "medium", "fast", "x-fast", "+10%", "+20%", "-10%"])
            rate_combo.grid(row=0, column=1, sticky=tk.EW, pady=5)
            
            ttk.Label(main_frame, text="Pitch:").grid(row=1, column=0, sticky=tk.W, pady=5)
            pitch_var = tk.StringVar(value="medium")
            pitch_combo = ttk.Combobox(main_frame, textvariable=pitch_var,
                                     values=["x-low", "low", "medium", "high", "x-high", "+10%", "+20%", "-10%"])
            pitch_combo.grid(row=1, column=1, sticky=tk.EW, pady=5)
            
            ttk.Label(main_frame, text="Volume:").grid(row=2, column=0, sticky=tk.W, pady=5)
            volume_var = tk.StringVar(value="medium")
            volume_combo = ttk.Combobox(main_frame, textvariable=volume_var,
                                      values=["silent", "x-soft", "soft", "medium", "loud", "x-loud"])
            volume_combo.grid(row=2, column=1, sticky=tk.EW, pady=5)
            
            ttk.Label(main_frame, text="Content:").grid(row=3, column=0, sticky=tk.NW, pady=5)
            content_text = scrolledtext.ScrolledText(main_frame, height=5, width=30)
            content_text.grid(row=3, column=1, sticky=tk.EW, pady=5)
            
            def apply():
                attrs = {}
                if rate_var.get() != "medium":
                    attrs["rate"] = rate_var.get()
                if pitch_var.get() != "medium":
                    attrs["pitch"] = pitch_var.get()
                if volume_var.get() != "medium":
                    attrs["volume"] = volume_var.get()
                
                content = content_text.get("1.0", tk.END).strip()
                tag = SSMLTag("prosody", attrs, content)
                self._insert_tag(tag.to_string())
                dialog.destroy()
                
            ttk.Button(main_frame, text="Apply", command=apply).grid(row=4, column=1, sticky=tk.E, pady=10)
            
        elif tag_type == "break":
            # Break options
            ttk.Label(main_frame, text="Time:").grid(row=0, column=0, sticky=tk.W, pady=5)
            time_var = tk.StringVar(value="500ms")
            time_combo = ttk.Combobox(main_frame, textvariable=time_var,
                                    values=["100ms", "250ms", "500ms", "1s", "2s", "3s"])
            time_combo.grid(row=0, column=1, sticky=tk.EW, pady=5)
            
            def apply():
                tag = SSMLTag("break", {"time": time_var.get()})
                self._insert_tag(tag.to_string())
                dialog.destroy()
                
            ttk.Button(main_frame, text="Apply", command=apply).grid(row=1, column=1, sticky=tk.E, pady=10)
            
        elif tag_type == "emphasis":
            # Emphasis options
            ttk.Label(main_frame, text="Level:").grid(row=0, column=0, sticky=tk.W, pady=5)
            level_var = tk.StringVar(value="moderate")
            level_combo = ttk.Combobox(main_frame, textvariable=level_var,
                                     values=["strong", "moderate", "none", "reduced"])
            level_combo.grid(row=0, column=1, sticky=tk.EW, pady=5)
            
            ttk.Label(main_frame, text="Content:").grid(row=1, column=0, sticky=tk.NW, pady=5)
            content_text = scrolledtext.ScrolledText(main_frame, height=5, width=30)
            content_text.grid(row=1, column=1, sticky=tk.EW, pady=5)
            
            def apply():
                content = content_text.get("1.0", tk.END).strip()
                tag = SSMLTag("emphasis", {"level": level_var.get()}, content)
                self._insert_tag(tag.to_string())
                dialog.destroy()
                
            ttk.Button(main_frame, text="Apply", command=apply).grid(row=2, column=1, sticky=tk.E, pady=10)
    
    def _insert_tag(self, tag_str: str):
        """Insert a tag at the cursor position."""
        try:
            self.ssml_text.insert(tk.INSERT, tag_str)
            self._update_preview()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to insert tag: {str(e)}")
    
    def _play_preview(self):
        """Play TTS preview of the current caption."""
        # Disable preview button while generating
        self.preview_btn.config(state=tk.DISABLED)
        self.preview_btn.config(text="Generating...")
        
        # Use threading to prevent UI freeze
        self.play_thread = threading.Thread(target=self._generate_and_play_audio)
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def _generate_and_play_audio(self):
        """Generate and play TTS audio in a separate thread."""
        try:
            ssml = self.ssml_text.get("1.0", tk.END).strip()
            
            # Extract plain text from SSML since edge_tts doesn't support SSML
            plain_text = SSMLParser.extract_content(ssml)
            
            # Generate content hash for caching
            content_hash = hashlib.md5(plain_text.encode()).hexdigest()
            
            # Check if we already have this audio cached
            if content_hash in self.previewed_captions:
                audio_path = self.previewed_captions[content_hash]
                if os.path.exists(audio_path):
                    # Use cached audio
                    pygame.mixer.music.load(str(audio_path))
                    pygame.mixer.music.play()
                    
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    
                    # Skip generation
                    self.master.after(0, lambda: self.preview_btn.config(state=tk.NORMAL, text="Play Preview"))
                    return
            
            # Create unique filename using content hash
            audio_path = self.temp_dir / f"preview_{content_hash}.mp3"
            
            # Use edge_tts to generate audio with the configured parameters
            # Only include parameters that have actual values
            kwargs = {"voice": self.voice_id}
            if self.rate:
                kwargs["rate"] = self.rate
            if self.volume:
                kwargs["volume"] = self.volume
            if self.pitch:
                kwargs["pitch"] = self.pitch
            
            communicate = edge_tts.Communicate(plain_text, **kwargs)
            
            # Run the async operation in our event loop
            async def process_audio():
                # Save audio to file
                await communicate.save(str(audio_path))
            
            # Execute the async function
            self.loop.run_until_complete(process_audio())
            
            # Cache this audio
            self.previewed_captions[content_hash] = audio_path
            
            # Play the audio using pygame
            pygame.mixer.music.load(str(audio_path))
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            # Clean up old audio files
            self._cleanup_temp_files()
            
        except Exception as e:
            # Show error in main thread
            self.master.after(0, lambda: messagebox.showerror("TTS Error", f"Failed to generate audio: {str(e)}"))
        
        # Re-enable preview button in main thread
        self.master.after(0, lambda: self.preview_btn.config(state=tk.NORMAL, text="Play Preview"))
    
    def _cleanup_temp_files(self):
        """Clean up old temporary audio files."""
        try:
            # Keep only the 5 most recent files
            files = sorted(self.temp_dir.glob("*.mp3"), key=os.path.getmtime, reverse=True)
            for file in files[5:]:
                try:
                    os.remove(file)
                except:
                    pass
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
    
    def get_edited_captions(self) -> List[str]:
        """Get the edited captions."""
        return self.captions
    
    def get_edited_indices(self) -> set:
        """Get indices of captions that were actually modified."""
        return self.edited_captions


def edit_ssml_captions(captions: List[str], elevenlabs_client=None, voice_id="en-AU-WilliamNeural",
                      rate=None, volume=None, pitch=None) -> List[str]:
    """
    Open GUI editor for SSML captions.
    
    Args:
        captions: List of SSML caption strings to edit
        elevenlabs_client: Ignored, kept for backward compatibility
        voice_id: Voice ID to use for previews (for edge_tts)
        rate: Edge TTS speech rate parameter
        volume: Edge TTS volume parameter
        pitch: Edge TTS pitch parameter
        
    Returns:
        List of edited SSML caption strings
    """
    root = tk.Tk()
    editor = SSMLEditorGUI(root, captions, voice_id, rate, volume, pitch)
    root.mainloop()
    return editor.get_edited_captions()


if __name__ == "__main__":
    # Initialize pygame for standalone testing
    pygame.init()
    
    # Test the editor with sample captions
    test_captions = [
        "<speak><prosody rate='fast' pitch='high'>This is a test caption!</prosody></speak>",
        "<speak><break time='500ms'/> <prosody volume='loud'>Another caption?!</prosody></speak>",
        "<speak>Plain caption with no special formatting.</speak>"
    ]
    
    edited = edit_ssml_captions(test_captions)
    print("Edited captions:")
    for caption in edited:
        print(f"  - {caption}")
    
    # Clean up pygame resources
    pygame.quit()