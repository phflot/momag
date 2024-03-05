"""
magnify_casme_demo
Copyright 2024 by Philipp Flotho, All rights reserved.
"""

from neurovc.util import VideoReader, VideoWriter
from neurovc.momag import OnlineLandmarkMagnifier
import neurovc as nvc


if __name__ == "__main__":
    video_reader = VideoReader("vid.avi")
    motion_magnifier = None
    with VideoWriter("mag.avi",
                     framerate=video_reader.framerate) as video_writer:
        while video_reader.has_frames():
            frame = video_reader.read_frame()[1]
            if motion_magnifier is None:
                motion_magnifier = OnlineLandmarkMagnifier(landmarks=nvc.LM_MOUTH + nvc.LM_EYE_LEFT + nvc.LM_EYE_RIGHT,
                                                           reference=frame, alpha=15)
            video_writer(motion_magnifier(frame))
