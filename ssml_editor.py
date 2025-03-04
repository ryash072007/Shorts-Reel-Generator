"""
SSML Editor

A dialog for editing SSML (Speech Synthesis Markup Language) captions with voice preview
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import asyncio
import tempfile
from typing import List, Optional

# Import edge-tts for voice synthesis
import edge_tts

def strip_ssml_tags(text):
    """Remove SSML tags from text"""
    if not text:
        return ""
    # Remove speak tags
    text = text.replace("<speak>", "").replace("</speak>", "")
    # Remove other common SSML tags
    tags_to_remove = [
        "break", "emphasis", "prosody", "say-as", "phoneme", 
        "sub", "voice", "p", "s", "mark", "audio"
    ]
    for tag in tags_to_remove:
        # Remove opening tags like <tag> or <tag attribute="value">
        text = text.replace(f"<{tag}>", "")
        # Remove self-closing tags like <tag/>
        while f"<{tag} " in text and ">" in text:
            start = text.find(f"<{tag} ")
            end = text.find(">", start) + 1
            text = text[:start] + text[end:]
        # Remove closing tags
        text = text.replace(f"</{tag}>", "")
    return text

class SSMLEditorDialog(tk.Toplevel):
    """Dialog for editing SSML captions"""

    def __init__(self, parent, captions: List[str], voice_id: str = "en-AU-WilliamNeural", 
                 rate: str = "+0%", volume: str = "+0%", pitch: str = "+0Hz"):
        super().__init__(parent)
        self.parent = parent
        self.captions = captions.copy()  # Make a copy to avoid modifying the original
        self.voice_id = voice_id
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.result = None
        
        # Track if currently playing audio
        self.is_playing = False

        # Configure dialog
        self.title("SSML Caption Editor")
        self.geometry("900x600")
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.transient(parent)
        self.grab_set()
        
        # Create UI elements
        self.create_widgets()
        
        # Initialize with existing captions
        self.load_captions()

    def create_widgets(self):
        """Create all UI widgets for the dialog"""
        # Main container with padding
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="Edit the SSML captions below. You can add SSML tags for emphasis, pauses, etc.",
            wraplength=800
        )
        instructions.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # SSML Help Button
        help_frame = ttk.Frame(main_frame)
        help_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        help_btn = ttk.Button(help_frame, text="SSML Help", command=self.show_ssml_help)
        help_btn.pack(side=tk.RIGHT)
        
        # Voice settings frame
        voice_frame = ttk.LabelFrame(main_frame, text="Voice Settings")
        voice_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # Voice selection
        voice_options = ["en-AU-WilliamNeural", "en-US-GuyNeural", "en-US-AriaNeural", 
                        "en-GB-RyanNeural", "en-CA-ClaraNeural", "en-IE-ConnorNeural"]
        voice_label = ttk.Label(voice_frame, text="Voice:")
        voice_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.voice_var = tk.StringVar(value=self.voice_id)
        voice_combo = ttk.Combobox(voice_frame, textvariable=self.voice_var, values=voice_options, width=25)
        voice_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Rate selection
        rate_label = ttk.Label(voice_frame, text="Rate:")
        rate_label.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.rate_var = tk.StringVar(value=self.rate)
        rate_combo = ttk.Combobox(voice_frame, textvariable=self.rate_var, 
                                 values=["-20%", "-10%", "+0%", "+10%", "+15%", "+20%", "+30%"], width=10)
        rate_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Volume selection
        volume_label = ttk.Label(voice_frame, text="Volume:")
        volume_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.volume_var = tk.StringVar(value=self.volume)
        volume_combo = ttk.Combobox(voice_frame, textvariable=self.volume_var, 
                                   values=["-20%", "-10%", "+0%", "+5%", "+10%", "+20%"], width=10)
        volume_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Pitch selection
        pitch_label = ttk.Label(voice_frame, text="Pitch:")
        pitch_label.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.pitch_var = tk.StringVar(value=self.pitch)
        pitch_combo = ttk.Combobox(voice_frame, textvariable=self.pitch_var, 
                                  values=["-20Hz", "-10Hz", "+0Hz", "+10Hz", "+20Hz", "+30Hz"], width=10)
        pitch_combo.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Caption notebook (tabs for multiple captions)
        self.caption_notebook = ttk.Notebook(main_frame)
        self.caption_notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)
        
        # Caption editors will be added in load_captions()
        self.caption_editors = []
        
        # Button frame at bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        # Add caption button
        add_btn = ttk.Button(button_frame, text="Add Caption", command=self.add_caption)
        add_btn.pack(side=tk.LEFT, padx=5)
        
        # Remove caption button
        remove_btn = ttk.Button(button_frame, text="Remove Caption", command=self.remove_caption)
        remove_btn.pack(side=tk.LEFT, padx=5)
        
        # Preview voice button
        self.preview_btn = ttk.Button(button_frame, text="Preview Voice", command=self.preview_voice)
        self.preview_btn.pack(side=tk.LEFT, padx=20)
        
        # OK/Cancel buttons
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=self.on_cancel)
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        ok_btn = ttk.Button(button_frame, text="OK", command=self.on_ok)
        ok_btn.pack(side=tk.RIGHT, padx=5)
        
    def load_captions(self):
        """Load existing captions into the editor"""
        # Clear existing tabs first
        for tab in self.caption_notebook.tabs():
            self.caption_notebook.forget(tab)
        
        self.caption_editors = []
        
        # Add each caption as a tab
        for i, caption in enumerate(self.captions):
            # Create a frame for this caption
            tab_frame = ttk.Frame(self.caption_notebook)
            self.caption_notebook.add(tab_frame, text=f"Caption {i+1}")
            
            # Add editor with scrollbars
            editor = scrolledtext.ScrolledText(tab_frame, wrap=tk.WORD, height=15)
            editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Strip speak tags for better editing
            text = caption
            if caption.startswith("<speak>") and caption.endswith("</speak>"):
                text = caption[7:-8]  # Remove <speak> and </speak>
            
            editor.insert(tk.END, text)
            self.caption_editors.append(editor)
            
        # Add an empty caption if none exist
        if not self.captions:
            self.add_caption()

    def add_caption(self):
        """Add a new caption tab"""
        # Create a frame for the new caption
        tab_frame = ttk.Frame(self.caption_notebook)
        tab_index = len(self.caption_editors) + 1
        self.caption_notebook.add(tab_frame, text=f"Caption {tab_index}")
        
        # Add editor
        editor = scrolledtext.ScrolledText(tab_frame, wrap=tk.WORD, height=15)
        editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add to the list of editors and select the new tab
        self.caption_editors.append(editor)
        self.caption_notebook.select(len(self.caption_editors) - 1)
        
        # Add empty caption to the list
        self.captions.append("<speak></speak>")

    def remove_caption(self):
        """Remove the current caption tab"""
        current_tab = self.caption_notebook.index(self.caption_notebook.select())
        
        # Don't remove if it's the last caption
        if len(self.caption_editors) <= 1:
            messagebox.showwarning("Warning", "Cannot remove the last caption.")
            return
        
        # Remove from the notebook, editors list, and captions list
        self.caption_notebook.forget(current_tab)
        self.caption_editors.pop(current_tab)
        self.captions.pop(current_tab)
        
        # Rename the tabs
        for i in range(len(self.caption_editors)):
            self.caption_notebook.tab(i, text=f"Caption {i+1}")

    def preview_voice(self):
        """Preview the voice with the current settings"""
        if self.is_playing:
            return  # Prevent multiple simultaneous previews
            
        # Get the current caption
        current_tab = self.caption_notebook.index(self.caption_notebook.select())
        if current_tab >= len(self.caption_editors):
            return
        
        editor = self.caption_editors[current_tab]
        text = editor.get(1.0, tk.END).strip()
        
        # Add speak tags if not present
        if not text.startswith("<speak>"):
            text = f"<speak>{text}</speak>"
        
        # Disable button and show playing status
        self.is_playing = True
        self.preview_btn.config(state=tk.DISABLED, text="Playing...")
        
        # Start playback in a background thread
        threading.Thread(target=self._play_audio, args=(text,), daemon=True).start()

    def _play_audio(self, ssml_text):
        """Play the audio in a background thread"""
        try:
            # Create temp file for audio
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_file.close()
            
            # Get voice settings
            voice_id = self.voice_var.get()
            rate = self.rate_var.get()
            volume = self.volume_var.get() 
            pitch = self.pitch_var.get()
            
            # Generate audio file with edge_tts
            async def generate_audio():
                communicate = edge_tts.Communicate(
                    text=ssml_text, 
                    voice=voice_id,
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
            
            # Reset button in main thread
            self.after(0, self._reset_play_button)
            
        except Exception as e:
            # Handle errors
            self.after(0, lambda: messagebox.showerror("Playback Error", str(e)))
            self.after(0, self._reset_play_button)
    
    def _reset_play_button(self):
        """Reset play button after playback"""
        self.is_playing = False
        self.preview_btn.config(state=tk.NORMAL, text="Preview Voice")

    def show_ssml_help(self):
        """Show help dialog with SSML examples"""
        help_text = """
SSML Tag Examples:

Pauses:
<break time="500ms"/>
<break strength="weak"/>

Emphasis:
<emphasis level="moderate">Important text</emphasis>

Speak rate:
<prosody rate="slow">Slow speech</prosody>
<prosody rate="fast">Fast speech</prosody>

Volume and Pitch:
<prosody volume="loud">Loud speech</prosody>
<prosody pitch="high">High pitch</prosody>

Pronounce as:
<say-as interpret-as="characters">SSML</say-as>
<say-as interpret-as="cardinal">123</say-as>
<say-as interpret-as="ordinal">1</say-as> (spoken as "first")
<say-as interpret-as="date">01/01/2022</say-as>
        """
        
        # Create help dialog
        help_dialog = tk.Toplevel(self)
        help_dialog.title("SSML Help")
        help_dialog.geometry("500x400")
        help_dialog.transient(self)
        
        # Add text widget with examples
        text = scrolledtext.ScrolledText(help_dialog, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert(tk.END, help_text)
        text.config(state=tk.DISABLED)  # Make read-only
        
        # Add close button
        close_btn = ttk.Button(help_dialog, text="Close", command=help_dialog.destroy)
        close_btn.pack(pady=10)

    def on_ok(self):
        """Save changes and close dialog"""
        # Gather captions from editors
        updated_captions = []
        for editor in self.caption_editors:
            text = editor.get(1.0, tk.END).strip()
            # Add speak tags if not present
            if not text.startswith("<speak>"):
                text = f"<speak>{text}</speak>"
            updated_captions.append(text)
        
        # Set result and close
        self.result = updated_captions
        self.destroy()

    def on_cancel(self):
        """Cancel and close dialog"""
        self.result = None
        self.destroy()


def edit_ssml_captions(captions: List[str], voice_id: str = "en-AU-WilliamNeural",
                      rate: str = "+0%", volume: str = "+0%", pitch: str = "+0Hz") -> Optional[List[str]]:
    """
    Open SSML editor dialog and return edited captions
    
    Args:
        captions: List of SSML captions to edit
        voice_id: Voice ID for TTS preview
        rate: Speaking rate
        volume: Voice volume
        pitch: Voice pitch
    
    Returns:
        List of edited captions or None if canceled
    """
    # Create a temporary root if not running from a Tkinter app
    if not tk._default_root:
        root = tk.Tk()
        root.withdraw()
    else:
        root = tk._default_root
    
    # Open dialog
    dialog = SSMLEditorDialog(root, captions, voice_id, rate, volume, pitch)
    
    # Wait for dialog to close
    root.wait_window(dialog)
    
    # Return result
    return dialog.result


if __name__ == "__main__":
    # Test the editor
    test_captions = [
        "<speak>This is the first caption.</speak>",
        "<speak>This is the second caption with <emphasis level='strong'>emphasis</emphasis>.</speak>"
    ]
    
    result = edit_ssml_captions(test_captions)
    print("Result:", result)
