__author__ = "Philipp Flotho"
"""
multimodal_cam_calib
Copyright 2024 by Philipp Flotho, All rights reserved.
"""
import numpy as np
import cv2
import os


class VideoLooper:
    def __init__(self):
        self.stop = False

    def __call__(self, keystroke):
        keystroke &= 0xFF
        if keystroke == ord(' '):
            while True:
                keystroke = cv2.waitKey() & 0xFF
                if keystroke == 27:
                    self.stop = True
                    break
                if keystroke == ord(' '):
                    break
        return keystroke == 27 or self.stop


class VideoReader:
    def __init__(self, filepath):
        self.cap = cv2.VideoCapture(filepath)
        self.framerate = self.cap.get(cv2.CAP_PROP_FPS)
        self.n_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def read_frame(self, i=None):
        ret = False
        frame = None
        if self.cap.isOpened():
            if i is not None and i < self.n_frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
        return ret, frame

    def has_frames(self):
        return self.cap.isOpened() and self.cap.get(cv2.CAP_PROP_POS_FRAMES) < self.n_frames

    def read_frames(self, start_idx=0, end_idx=-1, offset=1):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        counter = start_idx
        frames = []
        while counter < end_idx:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
            counter += 1
        return ret, np.array(frames)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start
            stop = item.stop
            step = item.step if item.step is not None else 1
        else:
            start = item
            stop = item
            step = 1



class VideoWriter:
    def __init__(self, filepath, framerate=50):
        self.writer = None
        self.filepath = filepath
        self.width = None
        self.height = None
        self.n_channels = None
        self.framerate = framerate

    def _check_range(self, min_val, max_val, checkrange):
        return

    def __call__(self, frame):
        if self.writer is None:
            self.height, self.width = frame.shape[:2]

            file_type = os.path.splitext(self.filepath)[1]
            if file_type == ".avi":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            elif file_type == ".mp4":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif file_type == ".mov":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.filepath, fourcc, 50, (self.width, self.height))

        if len(frame.shape) < 3:
            self.n_channels = 1
        else:
            self.n_channels = frame.shape[2]

        if self.n_channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        if frame.dtype != np.uint8:
            min_val = np.min(frame)
            max_val = np.max(frame)
            if min_val > 0 and max_val < 1:
                frame = (255 * frame).astype(np.uint8)
            elif min_val > 0 and max_val < 255:
                frame = frame.astype(np.uint8)
            else:
                frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        self.writer.write(frame)

    def write_frames(self, frames):
        for frame in frames:
            self(frame)

    def __del__(self):
        if self.writer is not None:
            self.writer.release()


def remap_RGB(frame, scaling, f1, f2):
    frame = scaling * (frame.astype(float) - f2) - f1
    frame[frame < 0] = 0
    frame[frame > 255] = 255
    return frame.astype(np.uint8)


def imagesc(img, window_name="Imagesc", color_map=cv2.COLORMAP_HOT):
    cv2.imshow(window_name, normalize_color(img, color_map))


def normalize_color(img, color_map=cv2.COLORMAP_HOT, normalize=True):
    img_8b = np.empty(img.shape, np.uint8)
    if normalize:
        cv2.normalize(img, img_8b, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    else:
        img_8b = (255 * img).astype(np.uint8)
    img_8b = cv2.applyColorMap(img_8b, color_map)
    return img_8b


def draw_landmarks(frame, landmarks):
    frame_rgb = normalize_color(frame)
    for (x, y) in landmarks:
        cv2.circle(frame_rgb, (int(np.round(x)), int(np.round(y))), 2, (0, 0, 0), -1)

    return frame_rgb


def imgaussfilt(img, sigma):
    if np.isscalar(sigma):
        width = 2 * (np.ceil(6 * sigma) // 2) + 1
        height = width
    else:
        width = 2 * (np.ceil(6 * sigma[0]) // 2) + 1
        height = 2 * (np.ceil(6 * sigma[1]) // 2) + 1
    width = int(width)
    height = int(height)
    return cv2.GaussianBlur(img, (width, height), sigmaX=sigma, sigmaY=sigma)


def map_temp(data, cam="A655"):
    if cam == "A65":
        return (0.1 * data) - 273.15
    elif cam == "A655":
        return (0.01 * data) - 273.15
    else:
        print("camera not implemented!")


class Debayerer:
    def __init__(self, pattern=cv2.COLOR_BAYER_RG2BGR, nh=2):

        self.pattern = pattern
        colors = np.array([150, 255, 150])

        self.reference = colors.reshape((1, 1, -1))

    def set_white_balance(self, colors):
        if isinstance(colors, np.ndarray):
            assert np.sum(colors.shape) == 3
        else:
            assert len(colors) == 3
        self.reference = np.expand_dims(np.mean(np.squeeze(np.array(colors)), axis=0).astype(float), (0, 1))

    def __call__(self, img):
        tmp = ((cv2.cvtColor(img, self.pattern).astype(float) / self.reference) * 255)
        tmp[tmp > 255] = 255
        return tmp.astype(np.uint8)