# trackers/track_manager.py
from typing import Dict, List, Optional
from config import CONFIG

MATCH_EVERY_N = CONFIG['MATCH_EVERY_N']
STABLE_FRAMES = CONFIG['STABLE_FRAMES']

class TrackManager:
    """
    Keeps track of recognition stability per track_id.
    Prevents flickering recognition results.
    """

    def __init__(self):
        # Stores recognition data for each track
        self.track_data: Dict[int, Dict] = {}

    def update(self, track_id: int, name: str, distance: Optional[float], frame_no: int) -> Optional[str]:
        """
        Updates recognition info for a track and returns stable name if ready.

        :param track_id: Tracker-assigned ID for the person.
        :param name: Recognition name (can be "Unknown").
        :param distance: Face embedding distance or None.
        :param frame_no: Current frame number.
        :return: Stable recognized name or None if still unstable.
        """
        if track_id not in self.track_data:
            self.track_data[track_id] = {
                "last_seen": frame_no,
                "name_history": [],
                "stable_name": None
            }

        track_info = self.track_data[track_id]
        track_info["last_seen"] = frame_no
        track_info["name_history"].append(name)

        # Only attempt matching every N frames
        if frame_no % MATCH_EVERY_N != 0:
            return track_info["stable_name"]

        # Check if recent N frames agree
        history = track_info["name_history"][-STABLE_FRAMES:]
        if len(history) >= STABLE_FRAMES and len(set(history)) == 1:
            track_info["stable_name"] = history[0]

        return track_info["stable_name"]

    def remove_inactive_tracks(self, current_frame: int, max_age_frames: int = 50):
        """
        Removes tracks that haven't been seen recently.

        :param current_frame: Current frame number.
        :param max_age_frames: Frames to keep inactive track before deletion.
        """
        inactive = [tid for tid, data in self.track_data.items()
                    if current_frame - data["last_seen"] > max_age_frames]
        for tid in inactive:
            del self.track_data[tid]
