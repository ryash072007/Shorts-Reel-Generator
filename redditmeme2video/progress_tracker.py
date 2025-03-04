"""
Progress Tracker

Utilities for tracking and reporting progress during video generation
"""

import time
import re
from typing import Dict, List, Callable, Optional, Any
from enum import Enum, auto

class ProcessStage(Enum):
    """Enum for different stages of video generation"""
    INITIALIZING = auto()
    FETCHING_MEMES = auto()
    GENERATING_CAPTIONS = auto()
    GENERATING_AUDIO = auto()
    DOWNLOADING_ASSETS = auto()
    RENDERING_VIDEO = auto()
    FINALIZING = auto()
    UPLOADING = auto()
    COMPLETED = auto()
    FAILED = auto()

class ProgressTracker:
    """Track progress of video generation process with detailed stages"""
    
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        # Callback function for progress updates
        self.callback = callback
        
        # Current progress state
        self.current_stage: ProcessStage = ProcessStage.INITIALIZING
        self.progress: int = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        # Stage progress tracking
        self.stage_progress: Dict[ProcessStage, int] = {
            stage: 0 for stage in ProcessStage
        }
        
        # Stage weights (how much each stage contributes to overall progress)
        self.stage_weights: Dict[ProcessStage, float] = {
            ProcessStage.INITIALIZING: 5,
            ProcessStage.FETCHING_MEMES: 10,
            ProcessStage.GENERATING_CAPTIONS: 15,
            ProcessStage.GENERATING_AUDIO: 15,
            ProcessStage.DOWNLOADING_ASSETS: 10,
            ProcessStage.RENDERING_VIDEO: 40,
            ProcessStage.FINALIZING: 5,
            ProcessStage.UPLOADING: 0,  # Optional stage
            ProcessStage.COMPLETED: 0,
            ProcessStage.FAILED: 0,
        }
        
        # Detailed logs for each stage
        self.stage_logs: Dict[ProcessStage, List[str]] = {
            stage: [] for stage in ProcessStage
        }
        
        # Error information
        self.error_message: Optional[str] = None
        
        # Send initial log
        self.log("Starting video generation process")

    def update_stage(self, stage: ProcessStage, progress: int = 0, message: Optional[str] = None):
        """Update the current stage and progress"""
        self.current_stage = stage
        self.stage_progress[stage] = progress
        
        # Calculate overall progress based on stage weights
        total_weight = sum(
            self.stage_weights[s] for s in ProcessStage 
            if s != ProcessStage.FAILED and s != ProcessStage.COMPLETED
        )
        
        weighted_progress = 0
        for s in ProcessStage:
            if s == ProcessStage.COMPLETED:
                # If completed, progress is 100%
                if stage == ProcessStage.COMPLETED:
                    weighted_progress = 100
                    break
            elif s == ProcessStage.FAILED:
                # Handle failed state
                if stage == ProcessStage.FAILED:
                    break
            elif s.value <= stage.value:
                # For completed stages, count full weight
                if s != stage:
                    weighted_progress += self.stage_weights[s]
                # For current stage, count partial weight
                else:
                    weighted_progress += (self.stage_weights[s] * progress) / 100
        
        # Calculate final percentage
        self.progress = int(weighted_progress * 100 / total_weight) if total_weight > 0 else 0
        
        # Ensure progress is between 0-100
        self.progress = max(0, min(100, self.progress))
        
        # Log message if provided
        if message:
            self.log(message)
        else:
            self.log(f"Stage: {stage.name}, Progress: {progress}%, Overall: {self.progress}%")
    
    def log(self, message: str):
        """Add a log message for the current stage"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # Add to appropriate stage log
        self.stage_logs[self.current_stage].append(log_entry)
        
        # Send to callback if available
        if self.callback:
            try:
                self.callback(log_entry)
            except:
                pass  # Ignore callback errors
        
        # Update last update time
        self.last_update_time = time.time()
    
    def set_error(self, error_message: str):
        """Set an error and update to failed state"""
        self.error_message = error_message
        self.update_stage(ProcessStage.FAILED, 0, f"ERROR: {error_message}")
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds since start"""
        return time.time() - self.start_time
    
    def get_time_since_last_update(self) -> float:
        """Get time in seconds since last update"""
        return time.time() - self.last_update_time
    
    def get_all_logs(self) -> List[str]:
        """Get all logs across all stages"""
        all_logs = []
        for stage in ProcessStage:
            all_logs.extend(self.stage_logs[stage])
        return all_logs
    
    def get_stage_name(self) -> str:
        """Get the current stage name in a readable format"""
        stage_name = self.current_stage.name
        return stage_name.replace('_', ' ').title()
    
    def estimate_time_remaining(self) -> Optional[float]:
        """Estimate remaining time in seconds based on progress"""
        if self.progress <= 0:
            return None
            
        elapsed = self.get_elapsed_time()
        if elapsed <= 0:
            return None
            
        rate = self.progress / elapsed  # Progress per second
        if rate <= 0:
            return None
            
        remaining = (100 - self.progress) / rate
        return remaining
    
    def format_time(self, seconds: float) -> str:
        """Format seconds into a readable time string"""
        if seconds is None:
            return "Unknown"
            
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def get_status_summary(self) -> str:
        """Get a summary of current status"""
        stage_name = self.get_stage_name()
        elapsed = self.format_time(self.get_elapsed_time())
        remaining = self.format_time(self.estimate_time_remaining())
        
        if self.current_stage == ProcessStage.COMPLETED:
            return f"Completed in {elapsed}"
        elif self.current_stage == ProcessStage.FAILED:
            return f"Failed after {elapsed}: {self.error_message}"
        else:
            return f"{stage_name} - {self.progress}% complete - Elapsed: {elapsed}, Remaining: {remaining}"

# Helper function to parse log message for progress information
def parse_progress_from_log(log_message: str) -> tuple:
    """Parse progress information from a log message"""
    stage_match = re.search(r"Stage: ([A-Z_]+)", log_message)
    progress_match = re.search(r"Progress: (\d+)%", log_message)
    overall_match = re.search(r"Overall: (\d+)%", log_message)
    
    stage = stage_match.group(1) if stage_match else None
    progress = int(progress_match.group(1)) if progress_match else None
    overall = int(overall_match.group(1)) if overall_match else None
    
    return stage, progress, overall
