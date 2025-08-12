# utils/rec_logger.py
import csv
import os
import time
from typing import Optional
from config import LOG_FILE  # Using central config

class RecognitionLogger:
    """
    Simple CSV logger for recognitions.
    Columns: timestamp_iso, frame_no, track_id, name, distance, registered
    """

    def __init__(self, path: str = LOG_FILE):
        self.path = path
        self._ensure_header()

    def _ensure_header(self):
        """Ensure CSV has header row if file doesn't exist."""
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "frame_no", "track_id", "name", "distance", "registered"])

    def log(self, frame_no: int, track_id: int, name: str, distance: Optional[float], registered: bool = False):
        """Log one recognition event."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts,
                frame_no,
                track_id,
                name,
                "" if distance is None else "{:.6f}".format(distance),
                int(bool(registered))
            ])
