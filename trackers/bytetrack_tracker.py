# bytetrack_tracker.py
from yolox.tracker.byte_tracker import BYTETracker
import numpy as np

class BYTETrackerArgs:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20

class ByteTrackWrapper:
    def __init__(self, frame_rate=30, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        """
        Initialize ByteTrack.

        Args:
            frame_rate (int): FPS of the video.
            track_thresh (float): detection confidence threshold.
            track_buffer (int): number of frames to keep lost tracks.
            match_thresh (float): threshold for matching.
        """
        args = BYTETrackerArgs(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            mot20=False  # Set to True if using MOT20 dataset
        )
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

    def update(self, detections, img_info=None, img_size=None):
        """
        Update tracker with new detections.

        Args:
            detections (list or np.ndarray): list of detections, each detection
                format: [x1, y1, x2, y2, confidence]
            img_info (tuple): (height, width) of the current image/frame.
            img_size (tuple): (height, width) of the model input size.

        Returns:
            list of tracks: each track is a dict with keys:
                - 'track_id'
                - 'bbox' (x1,y1,x2,y2)
                - 'score'
        """
        if len(detections) == 0:
            dets = np.empty((0, 5))
        else:
            dets = np.array(detections)

        # If your BYTETracker update requires img_info and img_size, pass them; otherwise, adapt accordingly.
        # If not available, you can pass dummy values like (0,0)
        if img_info is None:
            img_info = (0, 0)
        if img_size is None:
            img_size = (0, 0)

        tracks = self.tracker.update(dets, img_info, img_size)

        output_tracks = []
        for t in tracks:
            track_info = {
                "track_id": t.track_id,
                "bbox": t.tlbr.tolist(),  # Convert numpy array to list if needed
                "score": t.score
            }
            output_tracks.append(track_info)

        return output_tracks
