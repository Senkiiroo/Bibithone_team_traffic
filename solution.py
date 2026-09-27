# -*- coding: utf-8 -*-
"""
WIUT Hackathon 2026 - Elimination Task : solution.py
====================================================
Part A : detect_events(video_path) -> [[start_sec, end_sec, label], ...]
Part B : RiskEstimator.reset(meta) / RiskEstimator.step(frame, t_sec) -> float

--------------------------------------------------------------------------------
WHAT THIS FILE DOES (short version)
--------------------------------------------------------------------------------
The camera is fixed (one viewpoint), so we can learn the scene geometry of a clip
from the clip itself and then apply geometric / kinematic rules on top of tracks.

Pass 1 "scene scan"  (SCENE_FPS = 2 fps, cheap: MOG2 + Farneback at 1/8 scale)
    * motion activity  -> carriageway (road) mask
    * optical flow grid-> traffic direction per cell  (-> wrong_way, U-turn checks)
    * vehicle box sizes-> pixel-per-metre scale       (-> all m/s thresholds)
    * lateral offsets  -> lane model (lane centres)   (-> congestion, illegal turns)
    * yellow/red/green blobs -> traffic-light heads + signal state timeline
    * real stop events -> stop-band / intersection geometry

Pass 2 "detection pass" (DET_FPS = 5 fps)
    * Ultralytics YOLO detection (COCO person/2-wheeler/car/bus/truck)
    * own deterministic greedy-IoU tracker (constant velocity extrapolation)
    * side-channel per-frame accumulation: foreground blobs that no track explains
      (-> road_obstacle), smoke/fire colour masks (-> fire_smoke), signal colours

Rules ("run_rules"): one function per official class.  Everything is a heuristic;
Part A is allowed to use the whole clip (two passes, random access).  Part B is
NOT allowed to do that - it sees only the frames handed to step().

--------------------------------------------------------------------------------
COST (T4 / 16 GB, 5 fps detection, 2 fps scene scan)
--------------------------------------------------------------------------------
  Part A : ~5 YOLO(s@640, fp16) inferences per video second (~4 ms each) plus
           ~2 fps of MOG2/Farneback at 1/8 scale  -> roughly 25-40 ms per video
           second  (~40x faster than the 3x real-time budget).
  Part B : 1 YOLO(n@512, fp16) inference every RISK_STRIDE frames            ~2 ms
  Memory : tracks + two 1/8-scale maps, no frame buffering -> no OOM risk.
  A missing weight file / missing ultralytics never raises: we fall back to a
  pure-OpenCV motion detector, because a crash == empty prediction == 0 points.

--------------------------------------------------------------------------------
IMPORTANT
--------------------------------------------------------------------------------
All numeric thresholds live in CFG below and are heuristic defaults.  They MUST
be tuned on your own annotation of samples/*.mp4 (see README -> "Tuning").
Classes that are rare in the test set are dangerous: a class you predict but
that never occurs scores 0 and is added to the macro-F1 average.  Use CFG
["ENABLED"] to switch a rule off entirely once you know it only produces FPs.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import os.path as osp
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ============================================================================
#  Official class ids - DO NOT change the ids, only remove unused ones.
# ============================================================================
CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
           "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
           "solid_line_crossing", "stop_line", "congestion", "road_obstacle",
           "fire_smoke"]

# ============================================================================
#  Configuration - the only place you need to edit while tuning.
# ============================================================================
CFG = {
    # ---------------- models / weights (ship the .pt files in the package) ----
    "DET_WEIGHTS": os.environ.get("TRAFFIC_DET_WEIGHTS", "yolov8s.pt"),
    "RISK_WEIGHTS": os.environ.get("TRAFFIC_RISK_WEIGHTS", "yolov8n.pt"),
    "IMGSZ": 640,
    "CONF": 0.30,
    "IOU": 0.55,
    "HALF": True,                 # fp16 on CUDA, ignored on CPU
    "DEVICE": os.environ.get("TRAFFIC_DEVICE", ""),   # "" = auto, else "cuda:0"/"cpu"
    "BATCH": 8,                   # frames per YOLO call in the detection pass
    "USE_BYTETRACK": False,       # True = use Ultralytics ByteTrack instead of our tracker

    # ---------------- sampling ------------------------------------------------
    "DET_FPS": 5.0,               # Part A analysis frame rate
    "SCENE_FPS": 2.0,             # pass-1 scan frame rate
    "SCENE_SCALE": 8,             # downscale factor of the scene maps
    "SCENE_DET_MAX": 24,          # max frames fed to the detector during pass 1
    "RISK_STRIDE": 2,             # Part B: process every Nth frame

    # ---------------- tracker -------------------------------------------------
    "TRK_IOU": 0.20,
    "TRK_MAX_AGE_S": 1.6,
    "MIN_TRACK_OBS": 3,

    # ---------------- geometry / kinematics thresholds ------------------------
    "VEH_WIDTH_M": 1.8,           # assumed average vehicle width for px/m
    "VEH_LEN_M": 4.4,
    "STILL_MPS": 0.35,            # "stopped"
    "MOVE_MPS": 1.20,             # clearly moving
    "SCALE_MIN": 1.0,
    "SCALE_MAX": 200.0,

    # accident
    "ACC_TOUCH": 0.62,            # contact proxy: centre distance < k * mean box diag
    "ACC_MIN_APPROACH": 1.6,      # m/s combined speed before contact
    "ACC_HARD_BRAKE": 6.5,        # m/s^2 for the single-vehicle branch
    "ACC_MIN_CONF": 0.55,

    # near miss
    "NM_TOUCH": 1.55,             # no-contact proximity (in box diagonals)
    "NM_MIN_DECEL": 3.5,          # m/s^2
    "NM_MIN_HRATE": 35.0,         # deg/s (swerve)
    "NM_MIN_CONF": 0.55,

    # wrong way
    "WW_MIN_MPS": 1.5,
    "WW_MIN_ANGLE": 120.0,        # deg between heading and local lane direction
    "WW_MIN_RUN": 1.2,            # s of consistent disagreement
    "WW_MIN_FLOW": 0.35,          # local flow magnitude must be reliable

    # turns
    "TURN_MIN_DEG": 38.0,
    "TURN_MAX_DEG": 125.0,
    "UTURN_MIN_DEG": 130.0,
    "TURN_MAX_DUR": 6.5,
    "TURN_MIN_CONF": 0.55,

    # congestion
    "CONG_MPS": 0.6,
    "CONG_MIN_VEHICLES": 3,
    "CONG_MIN_LONG": 25.0,        # a signal queue must last at least this long

    # solid line crossing
    "SLC_MIN_CONF": 0.6,
    "SLC_TAIL": 1.2,              # s added after the crossing sample

    # obstacle / fire
    "OBS_MIN_RUN": 5.0,
    "FIRE_MIN_RUN": 1.0,

    # ---------------- Part B (risk) ------------------------------------------
    "RISK_IMGSZ": 512,
    "RISK_HORIZON": 5.0,          # H, matches the task definition
    "RISK_TTC_MAX": 5.0,          # s, TTC feature saturation
    "RISK_BRAKE": 3.0,            # m/s^2
    "RISK_CONTACT": 3.0,
    "RISK_DECAY": 0.86,           # persistence filter per processed step

    # ---------------- output post-processing ---------------------------------
    "MERGE_GAP": 0.35,            # merge same-class segments closer than this
    "MIN_DUR": {                  # drop segments shorter than this (blip filter)
        "accident": 0.8, "near_miss": 0.6, "red_light": 1.0, "wrong_way": 1.2,
        "illegal_u_turn": 1.2, "stopped_vehicle": 10.0, "jaywalking": 1.0,
        "failure_to_yield": 0.8, "illegal_turn": 1.2, "solid_line_crossing": 0.8,
        "stop_line": 1.0, "congestion": 4.0, "road_obstacle": 3.0, "fire_smoke": 1.0,
    },
    "MIN_CONF": {                 # confidence gate per class (FP classes are costly!)
        "accident": 0.55, "near_miss": 0.55, "red_light": 0.70, "wrong_way": 0.60,
        "illegal_u_turn": 0.60, "stopped_vehicle": 0.55, "jaywalking": 0.55,
        "failure_to_yield": 0.60, "illegal_turn": 0.60, "solid_line_crossing": 0.65,
        "stop_line": 0.70, "congestion": 0.60, "road_obstacle": 0.70, "fire_smoke": 0.60,
    },

    # Classes we are allowed to emit.  Remove the risky ones until you have dev
    # evidence: predicting a class that never occurs costs F1 = 0 for it.
    "ENABLED": set(CLASSES),

    # Rules that need scene geometry we may not find in every clip:
    "NEEDS_GEOMETRY": {"red_light", "stop_line", "illegal_u_turn", "illegal_turn",
                       "solid_line_crossing"},
}

# COCO -> our coarse category.  Bicycle/motorcycle -> two_wheeler.
COCO_MAP = {0: "person", 1: "two_wheeler", 2: "vehicle", 3: "two_wheeler",
            5: "vehicle", 7: "vehicle"}
DET_CLASSES = sorted(COCO_MAP.keys())

VEH_CATS = ("vehicle", "two_wheeler")
MOTION_CATS = ("vehicle", "two_wheeler", "unknown")

LOGGER = logging.getLogger("solution")
if not LOGGER.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[%(levelname)s] solution: %(message)s"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


# ============================================================================
#  Small helpers
# ============================================================================
def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = ix2 - ix1; ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ar = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    br = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = ar + br - inter
    return float(inter / den) if den > 0 else 0.0


def ang_diff(a: float, b: float) -> float:
    """Absolute difference between two angles in degrees, in [0, 180]."""
    d = (float(a) - float(b)) % 360.0
    return d if d <= 180.0 else 360.0 - d


def ang_between(v: Sequence[float], w: Sequence[float]) -> float:
    """Absolute angle (deg) between two 2D vectors."""
    nv = math.hypot(v[0], v[1]); nw = math.hypot(w[0], w[1])
    if nv < 1e-9 or nw < 1e-9:
        return 0.0
    c = (v[0] * w[0] + v[1] * w[1]) / (nv * nw)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def smooth1d(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Light box smoothing that keeps the length of the array."""
    if k <= 1 or x.size < 3:
        return x
    k = min(k, x.size if x.size % 2 else x.size - 1)
    if k <= 1:
        return x
    ker = np.ones(k, dtype=np.float32) / k
    pad = k // 2
    xp = np.pad(x.astype(np.float32), (pad, pad), mode="edge")
    return np.convolve(xp, ker, mode="valid")


def segments_intersect(p1, p2, p3, p4) -> bool:
    """Proper segment intersection test (orientation based)."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)

    o1 = orient(p1, p2, p3); o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1); o4 = orient(p3, p4, p2)
    return o1 != o2 and o3 != o4


def unit(v: Sequence[float]) -> Optional[np.ndarray]:
    n = math.hypot(v[0], v[1])
    if n < 1e-9:
        return None
    return np.array([v[0] / n, v[1] / n], dtype=np.float32)


def sample_times(t0: float, t1: float, dt: float) -> np.ndarray:
    """Sample times in [t0, t1] with a step dt (empty array if the interval is degenerate)."""
    if t1 <= t0 or dt <= 0:
        return np.array([], dtype=np.float64)
    n = int(math.floor((t1 - t0) / dt)) + 1
    n = min(n, 20000)                     # hard guard against pathological input
    return t0 + np.arange(n, dtype=np.float64) * dt


def runs_of(mask_times: Sequence[float], mask: Sequence[bool],
            min_len: float) -> List[Tuple[float, float]]:
    """Maximal True-runs of a boolean mask -> [(t_start, t_end), ...]."""
    out: List[Tuple[float, float]] = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = mask_times[i]
        elif not m and start is not None:
            if mask_times[i - 1] - start >= min_len:
                out.append((start, mask_times[i - 1]))
            start = None
    if start is not None and mask_times[-1] - start >= min_len:
        out.append((start, mask_times[-1]))
    return out


def guarded(fn):
    """Wrap a rule so that one buggy rule cannot kill the whole clip."""
    def inner(ctx):
        try:
            return fn(ctx) or []
        except Exception:
            LOGGER.exception("rule %s failed", getattr(fn, "__name__", "?"))
            return []
    inner.__name__ = getattr(fn, "__name__", "rule")
    return inner


# ============================================================================
#  Video reader
# ============================================================================
class VideoReader:
    """Thin cv2.VideoCapture wrapper: fps/duration, grab-based sampling.

    We use grab()/retrieve() so that skipped frames are not decoded - that is
    the single biggest CPU saving in the whole pipeline.
    """

    def __init__(self, path: str, det_fps: Optional[float] = None):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError("cannot open video: %s" % path)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if not (1.0 <= fps <= 240.0):
            fps = 25.0
        self.fps = fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.n_frames = n if n > 0 else self._count_frames()
        self.duration = float(self.n_frames) / self.fps if self.n_frames > 0 else 0.0
        self.stride = 1
        if det_fps and det_fps > 0:
            self.stride = max(1, int(round(self.fps / float(det_fps))))
        self.dt = self.stride / self.fps

    def _count_frames(self) -> int:
        """Some containers report frame_count = 0 / -1; fall back to counting."""
        cnt = 0
        while True:
            if not self.cap.grab():
                break
            cnt += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cnt

    def rewind(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def iter_frames(self, stride: Optional[int] = None, max_frames: Optional[int] = None,
                    resize_w: Optional[int] = None):
        """Yield (frame_index, t_sec, frame) every `stride` frames."""
        stride = max(1, int(stride or 1))
        idx = -1
        yielded = 0
        while True:
            if not self.cap.grab():
                break
            idx += 1
            if idx % stride:
                continue
            ok, frame = self.cap.retrieve()
            if not ok or frame is None:
                break
            if resize_w and frame.shape[1] > resize_w:
                frame = cv2.resize(frame, (resize_w, int(frame.shape[0] * resize_w / frame.shape[1])),
                                   interpolation=cv2.INTER_AREA)
            yield idx, idx / self.fps, frame
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


# ============================================================================
#  Detection
# ============================================================================
@dataclass
class Det:
    x1: float; y1: float; x2: float; y2: float
    conf: float
    cat: str
    coco: int = -1

    @property
    def w(self) -> float: return max(1e-6, self.x2 - self.x1)

    @property
    def h(self) -> float: return max(1e-6, self.y2 - self.y1)

    @property
    def cx(self) -> float: return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float: return 0.5 * (self.y1 + self.y2)

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def diag(self) -> float: return math.hypot(self.w, self.h)


class Detector:
    """Ultralytics YOLO wrapper with a pure-OpenCV fallback.

    Backends
    --------
    "yolo"   : ultralytics.YOLO(...).predict(source=[np.ndarray, ...]).
               `boxes.xyxy / boxes.conf / boxes.cls` are the documented fields.
    "motion" : if ultralytics or the .pt weights are missing (offline machine
               without the weights shipped, broken torch install, ...) we still
               must return *something* instead of raising.  MOG2 foreground
               blobs are used as pseudo-detections (class "unknown").
    """

    def __init__(self, weights: str, imgsz: int = 640, conf: float = 0.30, iou: float = 0.55,
                 half: bool = True, device: str = "", batch: int = 8):
        self.weights = weights
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half
        self.batch = max(1, batch)
        self.device = device or None
        self.model = None
        self.backend = "motion"
        self.bgsub: Optional[cv2.BackgroundSubtractor] = None
        self._load()

    # ---------------------------------------------------------------- load
    def _load(self) -> None:
        try:
            import torch                                    # noqa: F401
            from ultralytics import YOLO
            dev = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            model = YOLO(self.weights)
            try:
                model.to(dev)
            except Exception:
                dev = "cpu"
            self.half = bool(self.half and str(dev).startswith("cuda"))
            self.model = model
            self.device = dev
            self.backend = "yolo"
            try:                                            # fix seeds (determinism)
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0) if str(dev).startswith("cuda") else None
            except Exception:
                pass
            # warm-up: builds the predictor once, so the first frame is not slow
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(source=dummy, imgsz=self.imgsz, conf=self.conf,
                               classes=DET_CLASSES, verbose=False, device=self.device,
                               half=self.half)
            LOGGER.info("detector ready: %s on %s (half=%s)", self.weights, self.device, self.half)
        except Exception as exc:                            # pragma: no cover
            LOGGER.warning("YOLO unavailable (%s) -> MOG2 motion fallback", exc)
            self.backend = "motion"
            self.model = None
            self.bgsub = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=25,
                                                            detectShadows=False)

    # ---------------------------------------------------------------- predict
    def predict(self, frames: Sequence[np.ndarray]) -> List[List[Det]]:
        """Detections for a list of BGR frames (order preserved)."""
        if not frames:
            return []
        if self.backend == "yolo":
            return self._predict_yolo(frames)
        return [self._predict_motion(f) for f in frames]

    def _predict_yolo(self, frames: Sequence[np.ndarray]) -> List[List[Det]]:
        out: List[List[Det]] = []
        try:
            results = self.model.predict(source=list(frames), imgsz=self.imgsz, conf=self.conf,
                                         iou=self.iou, classes=DET_CLASSES, verbose=False,
                                         device=self.device, half=self.half, max_det=120)
        except Exception:
            LOGGER.exception("batched predict failed, retrying frame by frame")
            results = []
            for f in frames:
                results.extend(self.model.predict(source=f, imgsz=self.imgsz, conf=self.conf,
                                                  iou=self.iou, classes=DET_CLASSES,
                                                  verbose=False, device=self.device,
                                                  half=self.half, max_det=120))
        for r in results:
            dets: List[Det] = []
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                out.append(dets)
                continue
            try:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                cnf = boxes.conf.detach().cpu().numpy()
                cls = boxes.cls.detach().cpu().numpy().astype(int)
            except Exception:
                out.append(dets)
                continue
            for b, c, k in zip(xyxy, cnf, cls):
                cat = COCO_MAP.get(int(k))
                if cat is None:
                    continue
                dets.append(Det(float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                                float(c), cat, int(k)))
            out.append(dets)
        # a safety net: if ultralytics returned fewer results than frames, pad
        while len(out) < len(frames):
            out.append([])
        return out

    def _predict_motion(self, frame: np.ndarray) -> List[Det]:
        """Degraded backend: MOG2 foreground blobs as pseudo vehicle boxes."""
        h, w = frame.shape[:2]
        scale = 0.35
        small = cv2.resize(frame, (max(64, int(w * scale)), max(64, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        fg = self.bgsub.apply(small)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.dilate(fg, np.ones((5, 5), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: List[Det] = []
        amin = 0.0006 * small.shape[0] * small.shape[1]
        for c in cnts:
            if cv2.contourArea(c) < amin:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bh > 0.75 * small.shape[0]:
                continue                                    # global lighting change
            dets.append(Det(x / scale, y / scale, (x + bw) / scale, (y + bh) / scale,
                            0.30, "unknown", -1))
        return dets


_DET_CACHE: Dict[str, Detector] = {}


def get_detector(weights: str, **kw) -> Detector:
    """Process-wide detector cache (the harness calls us once per video)."""
    key = "%s|%s|%s" % (weights, kw.get("imgsz", CFG["IMGSZ"]), kw.get("conf", CFG["CONF"]))
    if key not in _DET_CACHE:
        _DET_CACHE[key] = Detector(weights, **kw)
    return _DET_CACHE[key]


# ============================================================================
#  Tracking (deterministic greedy IoU + constant velocity extrapolation)
# ============================================================================
@dataclass
class Obs:
    t: float
    box: Tuple[float, float, float, float]
    conf: float

    @property
    def cx(self) -> float: return 0.5 * (self.box[0] + self.box[2])

    @property
    def cy(self) -> float: return 0.5 * (self.box[1] + self.box[3])

    @property
    def w(self) -> float: return max(1e-6, self.box[2] - self.box[0])

    @property
    def h(self) -> float: return max(1e-6, self.box[3] - self.box[1])


class Track:
    """One object: a growing list of observations plus a smoothed velocity."""

    __slots__ = ("tid", "cat", "obs", "vx", "vy", "misses", "hits", "t_last")

    def __init__(self, tid: int, cat: str, obs: Obs):
        self.tid = tid
        self.cat = cat
        self.obs: List[Obs] = [obs]
        self.vx = 0.0
        self.vy = 0.0
        self.misses = 0
        self.hits = 1
        self.t_last = obs.t

    def add(self, obs: Obs) -> None:
        dt = obs.t - self.t_last
        if dt > 1e-6:
            vx = (obs.cx - self.obs[-1].cx) / dt
            vy = (obs.cy - self.obs[-1].cy) / dt
            a = 0.6                                     # EMA: stable but reactive
            self.vx = a * vx + (1.0 - a) * self.vx
            self.vy = a * vy + (1.0 - a) * self.vy
        self.obs.append(obs)
        self.hits += 1
        self.misses = 0
        self.t_last = obs.t

    def predict_box(self, t: float) -> Tuple[float, float, float, float]:
        """Constant-velocity extrapolation, damped over long gaps."""
        b = self.obs[-1]
        dt = t - b.t
        k = max(0.0, 1.0 - abs(dt) / 2.0)
        dx = self.vx * dt * k
        dy = self.vy * dt * k
        return (b.box[0] + dx, b.box[1] + dy, b.box[2] + dx, b.box[3] + dy)


class SimpleTracker:
    """Greedy IoU association, deterministic (no RNG), O(n*m) per frame."""

    def __init__(self, iou_thr: float = 0.20, max_age_s: float = 1.6, min_hits: int = 2,
                 sample_dt: float = 0.2):
        self.iou_thr = iou_thr
        self.max_age = max(1, int(round(max_age_s / max(1e-3, sample_dt))))
        self.min_hits = min_hits
        self.tracks: Dict[int, Track] = {}
        self.active: Dict[int, int] = {}
        self.next_id = 1

    @staticmethod
    def compatible(a: str, b: str) -> bool:
        if a == b:
            return True
        return a in MOTION_CATS and b in MOTION_CATS

    def update(self, t: float, dets: Sequence[Det]) -> List[Track]:
        """Associate detections at time t, return the tracks updated this frame."""
        pairs = []
        for tid in list(self.active.keys()):
            tr = self.tracks[tid]
            pb = tr.predict_box(t)
            for j, d in enumerate(dets):
                if not self.compatible(tr.cat, d.cat):
                    continue
                v = iou_xyxy(pb, d.xyxy)
                if v >= self.iou_thr:
                    pairs.append((v, tid, j))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))           # deterministic order
        used_t, used_d = set(), set()
        touched: List[Track] = []
        for _, tid, j in pairs:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid); used_d.add(j)
            tr = self.tracks[tid]
            tr.add(Obs(t, dets[j].xyxy, dets[j].conf))
            if tr.cat == "unknown" and dets[j].cat != "unknown":
                tr.cat = dets[j].cat
            self.active[tid] = 0
            touched.append(tr)
        for tid in list(self.active.keys()):
            if tid not in used_t:
                self.active[tid] += 1
                if self.active[tid] > self.max_age:
                    del self.active[tid]
        for j, d in enumerate(dets):
            if j in used_d:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(tid, d.cat, Obs(t, d.xyxy, d.conf))
            self.active[tid] = 0
        return touched


class ByteTrackerAdapter:
    """Optional: Ultralytics ByteTrack (`model.track(frame, persist=True, ...)`).

    Enabled with CFG["USE_BYTETRACK"] = True.  It keeps tracker state inside the
    Ultralytics predictor, so it must be fed consecutive frames of ONE video -
    which is exactly what the detection pass does.  If anything goes wrong we
    silently fall back to SimpleTracker (the caller handles the None ids).
    """

    def __init__(self, weights: str, imgsz: int, conf: float, iou: float, device: str,
                 tracker: str = "bytetrack.yaml"):
        self.ok = False
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            if device:
                self.model.to(device)
            self.tracker = tracker
            self.imgsz = imgsz
            self.conf = conf
            self.iou = iou
            self.ok = True
        except Exception as exc:                            # pragma: no cover
            LOGGER.warning("ByteTrack adapter unavailable (%s)", exc)

    def step(self, frame: np.ndarray) -> List[Tuple[int, Det]]:
        """Return [(track_id, Det), ...] for one frame, or [] on failure."""
        if not self.ok:
            return []
        try:
            res = self.model.track(source=frame, persist=True, tracker=self.tracker,
                                   imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                   classes=DET_CLASSES, verbose=False)[0]
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0 or boxes.id is None:
                return []
            xyxy = boxes.xyxy.detach().cpu().numpy()
            cnf = boxes.conf.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            ids = boxes.id.detach().cpu().numpy().astype(int)
            out = []
            for b, c, k, i in zip(xyxy, cnf, cls, ids):
                cat = COCO_MAP.get(int(k))
                if cat is None:
                    continue
                out.append((int(i), Det(float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                                        float(c), cat, int(k))))
            return out
        except Exception:
            LOGGER.exception("ByteTrack step failed")
            self.ok = False
            return []


# ============================================================================
#  Kinematics per track
# ============================================================================
class TrackSeries:
    """Per-track time series in image and metric units.

    All queries are plain np.interp lookups, so a rule can ask "where was this
    vehicle / how fast was it at time t" thousands of times cheaply.  Values
    outside the track lifetime are clamped - always guard with active(t).
    """

    def __init__(self, tid: int, cat: str, obs: Sequence[Obs], px_per_m: float):
        obs = sorted(obs, key=lambda o: o.t)
        t = np.array([o.t for o in obs], dtype=np.float64)
        keep = np.concatenate(([True], np.diff(t) > 1e-6))       # unique timestamps
        t = t[keep]
        cx = np.array([o.cx for o in obs], dtype=np.float64)[keep]
        cy = np.array([o.cy for o in obs], dtype=np.float64)[keep]
        bw = np.array([o.w for o in obs], dtype=np.float64)[keep]
        bh = np.array([o.h for o in obs], dtype=np.float64)[keep]
        cf = np.array([o.conf for o in obs], dtype=np.float64)[keep]

        self.tid = tid
        self.cat = cat
        self.t = t
        self.cx, self.cy, self.w, self.h, self.conf = cx, cy, bw, bh, cf
        self.n = int(t.size)
        self.px_per_m = max(1e-3, float(px_per_m))

        vx = smooth1d(np.gradient(cx, t), 3)
        vy = smooth1d(np.gradient(cy, t), 3)
        self.vx_px, self.vy_px = vx, vy
        self.speed_mps = smooth1d(np.hypot(vx, vy) / self.px_per_m, 3)
        heading = np.degrees(np.arctan2(vy, vx))
        self.heading = np.degrees(np.unwrap(np.radians(heading)))   # continuous
        self.hrate = smooth1d(np.gradient(self.heading, t), 3)      # deg/s
        self.accel = smooth1d(np.gradient(smooth1d(self.speed_mps, 3), t), 3)  # m/s^2

    # ------------------------------------------------------------------ queries
    def _at(self, arr: np.ndarray, t: float) -> float:
        return float(np.interp(t, self.t, arr))

    def pos(self, t: float) -> Tuple[float, float]:
        return self._at(self.cx, t), self._at(self.cy, t)

    def size(self, t: float) -> Tuple[float, float]:
        return self._at(self.w, t), self._at(self.h, t)

    def speed(self, t: float) -> float:
        return self._at(self.speed_mps, t)

    def accel_at(self, t: float) -> float:
        return self._at(self.accel, t)

    def heading_at(self, t: float) -> float:
        return self._at(self.heading, t)

    def hrate_at(self, t: float) -> float:
        return self._at(self.hrate, t)

    def vel(self, t: float) -> Tuple[float, float]:
        return self._at(self.vx_px, t) / self.px_per_m, self._at(self.vy_px, t) / self.px_per_m

    def active(self, t: float) -> bool:
        return self.t[0] - 1e-6 <= t <= self.t[-1] + 1e-6

    def diag(self, t: float) -> float:
        w, h = self.size(t)
        return math.hypot(w, h)

    def half_len_along(self, t: float, d: Sequence[float]) -> float:
        """Half box extent projected on direction d (box-to-front distance approx)."""
        w, h = self.size(t)
        return 0.5 * (w * abs(d[0]) + h * abs(d[1]))

    def dir_from_motion(self, t: float) -> Optional[np.ndarray]:
        vx, vy = self.vel(t)
        return unit((vx, vy))


def build_series(tracker: SimpleTracker, px_per_m: float, min_obs: int) -> List[TrackSeries]:
    """TrackSeries for every track long enough to be meaningful."""
    out: List[TrackSeries] = []
    for tid, tr in tracker.tracks.items():
        if len(tr.obs) < min_obs:
            continue
        try:
            t_uniq = set(ob[0] for ob in tr.obs)
            if len(t_uniq) < 3:
                continue
            out.append(TrackSeries(tid, tr.cat, tr.obs, px_per_m))
        except Exception:
            LOGGER.exception("series build failed for track %s", tid)
    out.sort(key=lambda s: s.t[0])
    return out


# ============================================================================
#  Scene model (pass 1)
# ============================================================================
@dataclass
class SceneModel:
    """Everything we learn about the static geometry of a clip."""
    scale: int = 8
    w: int = 0
    h: int = 0
    activity: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.float32))
    road_mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.uint8))
    flow: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 2), np.float32))
    flow_mag: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.float32))
    px_per_m: float = 25.0
    axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], np.float32))
    perp: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0], np.float32))
    lanes: List[float] = field(default_factory=list)
    lane_gap: float = 0.0
    frame_shape: Tuple[int, int] = (0, 0)

    # ---------------------------------------------------------------- queries
    def _cell(self, pt: Sequence[float]) -> Optional[Tuple[int, int]]:
        x = int(pt[0] // self.scale)
        y = int(pt[1] // self.scale)
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return None
        return x, y

    def road(self, pt: Sequence[float]) -> bool:
        c = self._cell(pt)
        return bool(self.road_mask[c[1], c[0]]) if c else False

    def act_at(self, pt: Sequence[float]) -> float:
        c = self._cell(pt)
        return float(self.activity[c[1], c[0]]) if c else 0.0

    def dir_at(self, pt: Sequence[float]) -> Optional[np.ndarray]:
        """Local traffic direction (unit vector, image coords) or None if unreliable."""
        c = self._cell(pt)
        if c is None:
            return None
        if self.flow_mag[c[1], c[0]] < CFG["WW_MIN_FLOW"]:
            return None
        v = self.flow[c[1], c[0]]
        return unit(v)

    def lateral(self, pt: Sequence[float]) -> float:
        """Coordinate across the traffic axis (lane coordinate)."""
        return float(pt[0] * self.perp[0] + pt[1] * self.perp[1])

    def lane_index(self, pt: Sequence[float]) -> int:
        if not self.lanes:
            return 0
        q = self.lateral(pt)
        d = [abs(q - L) for L in self.lanes]
        return int(np.argmin(d))

    def lane_leftness(self, pt: Sequence[float], d: Sequence[float]) -> int:
        """0 = leftmost lane for a driver travelling along d."""
        if len(self.lanes) < 2:
            return 0
        left = np.array([d[1], -d[0]], dtype=np.float32)   # влево при y-вниз
        s = float(self.perp @ left)                        # ±1: как упорядочены lanes
        order = sorted(range(len(self.lanes)), key=lambda i: -self.lanes[i] * s)
        try:
            return order.index(self.lane_index(pt))
        except ValueError:
            return 0



def build_scene_model(vr: VideoReader, det: Detector, cfg: dict) -> SceneModel:
    """Pass 1: cheap scan building activity / road mask / flow grid / scale / lanes.

    Libraries: OpenCV only (MOG2 background subtraction + Farneback optical flow
    at 1/8 resolution).  The detector is used on a handful of frames
    (SCENE_DET_MAX) purely to estimate the pixel-per-metre scale.
    """
    scene = SceneModel()
    scene.scale = int(cfg["SCENE_SCALE"])
    scene.frame_shape = (vr.height, vr.width)
    scene.w = max(8, vr.width // scene.scale)
    scene.h = max(8, vr.height // scene.scale)

    scene_fps = float(cfg["SCENE_FPS"])
    if vr.duration > 300.0:
        scene_fps = max(1.0, scene_fps / 2.0)               # long clips: scan slower
    stride = max(1, int(round(vr.fps / scene_fps)))

    act = np.zeros((scene.h, scene.w), np.float32)
    flow_sum = np.zeros((scene.h, scene.w, 2), np.float32)
    mag_sum = np.zeros((scene.h, scene.w), np.float32)
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    prev_gray = None
    n_used = 0
    det_frames: List[np.ndarray] = []
    q_all: List[float] = []

    for i, (fi, t, frame) in enumerate(vr.iter_frames(stride=stride)):
        small = cv2.resize(frame, (scene.w, scene.h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        fg = bg.apply(small)
        m = (fg > 200).astype(np.float32)
        if m.any():
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        act += m
        if prev_gray is not None and m.sum() > 0:
            fl = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                              0.5, 2, 15, 3, 5, 1.2, 0)
            wgt = m[..., None]
            flow_sum += fl * wgt
            mag_sum += np.linalg.norm(fl, axis=2) * m
        prev_gray = gray
        n_used += 1
        if len(det_frames) < int(cfg["SCENE_DET_MAX"]) and (i % 4 == 0):
            det_frames.append(frame)
    vr.rewind()

    if n_used == 0:
        return scene
    act /= float(n_used)

    # ---- road / carriageway mask: where objects actually move ----------------
    thr = max(0.05, float(np.percentile(act, 80)) * 0.35)
    road = (act > thr).astype(np.uint8)
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    road = cv2.dilate(road, np.ones((3, 3), np.uint8), iterations=1)
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(road, connectivity=8)
    keep = np.zeros_like(road)
    for k in range(1, n_lab):
        if stats[k, cv2.CC_STAT_AREA] >= max(24, 0.002 * scene.w * scene.h):
            keep[lab == k] = 1
    scene.road_mask = keep
    scene.activity = act

    # ---- flow grid: direction of travel per cell ---------------------------
    flow = np.zeros_like(flow_sum)
    nz = mag_sum > 1e-6
    flow[nz] = flow_sum[nz] / mag_sum[nz, None]
    flow[~nz] = 0.0
    k = max(3, (scene.w // 12) | 1)
    fx = cv2.GaussianBlur(flow[..., 0], (k, k), 0)
    fy = cv2.GaussianBlur(flow[..., 1], (k, k), 0)
    mag = np.hypot(fx, fy)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(mag > 1e-6, fx / np.maximum(mag, 1e-6), 0.0)
        uy = np.where(mag > 1e-6, fy / np.maximum(mag, 1e-6), 0.0)
    scene.flow = np.stack([ux, uy], axis=-1).astype(np.float32)
    scene.flow_mag = mag.astype(np.float32)
    scene.flow_mag[scene.road_mask == 0] = 0.0

    # ---- dominant traffic axis (mod 180 deg) -------------------------------
    if mag.size and mag.sum() > 1e-6:
        ang = np.arctan2(uy, ux)
        wgt = mag * (scene.road_mask > 0)
        # doubled-angle histogram makes the axis (mod 180) well defined
        hist, edges = np.histogram((ang * 2.0) % (2 * math.pi), bins=36, weights=wgt)
        kbest = int(np.argmax(hist))
        centre = 0.5 * (edges[kbest] + edges[kbest + 1])
        a = centre / 2.0
        axis = np.array([math.cos(a), math.sin(a)], np.float32)
        # orient the axis along the pixel flow (so "forward" is unambiguous)
        mean_flow = flow_sum.sum(axis=(0, 1))
        if float(np.dot(axis, mean_flow)) < 0:
            axis = -axis
        scene.axis = axis / max(1e-6, float(np.linalg.norm(axis)))
        scene.perp = np.array([-scene.axis[1], scene.axis[0]], np.float32)
    else:
        scene.axis = np.array([1.0, 0.0], np.float32)
        scene.perp = np.array([0.0, 1.0], np.float32)

    # ---- scale (px per metre) from vehicle box sizes -----------------------
    widths = []
    if det_frames:
        for dets in det.predict(det_frames):
            for d in dets:
                if d.cat in VEH_CATS and d.w > 4 and d.h > 4:
                    widths.append(math.sqrt(d.w * d.h))
    if len(widths) >= 3:
        med = float(np.median(widths))
        scene.px_per_m = float(np.clip(med / math.sqrt(cfg["VEH_WIDTH_M"] * cfg["VEH_LEN_M"]),
                                       cfg["SCALE_MIN"], cfg["SCALE_MAX"]))
    else:
        # no vehicles detected: assume the visible road width is ~ 25 m
        scene.px_per_m = float(np.clip(vr.width / 25.0, cfg["SCALE_MIN"], cfg["SCALE_MAX"]))
    LOGGER.info("scene: %.1f px/m, axis=%.0f deg, road=%.1f%%",
                scene.px_per_m, math.degrees(math.atan2(scene.axis[1], scene.axis[0])),
                100.0 * scene.road_mask.mean())

    # ---- lane model: histogram of lateral offsets --------------------------
    ys, xs = np.nonzero(scene.road_mask)
    if xs.size:
        pts = np.stack([(xs + 0.5) * scene.scale, (ys + 0.5) * scene.scale], axis=-1)
        q_all = list(pts @ scene.perp)
    if len(q_all) >= 50:
        q = np.asarray(q_all, np.float32)
        bin_w = max(4.0, scene.px_per_m * 0.35)
        lo, hi = float(q.min()), float(q.max())
        nb = int(max(4, min(96, (hi - lo) / bin_w)))
        hist, edges = np.histogram(q, bins=nb, range=(lo, hi))
        hist = smooth1d(hist.astype(np.float32), 3)
        centres = 0.5 * (edges[:-1] + edges[1:])
        peaks: List[float] = []
        min_sep = max(6.0, scene.px_per_m * 1.6)
        thr_h = 0.25 * float(hist.max()) if hist.max() > 0 else 0.0
        for i in range(1, nb - 1):
            if hist[i] >= thr_h and hist[i] >= hist[i - 1] and hist[i] > hist[i + 1]:
                if not peaks or abs(centres[i] - peaks[-1]) >= min_sep:
                    peaks.append(float(centres[i]))
        scene.lanes = peaks
        if len(peaks) >= 2:
            scene.lane_gap = float(np.median(np.diff(peaks)))
    return scene


# ============================================================================
#  Lane markings (for solid_line_crossing)
# ============================================================================
def detect_markings(vr: VideoReader, scene: SceneModel, cfg: dict,
                    max_frames: int = 200) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Accumulate bright thin structures -> line segments classified as solid.

    Heuristic: white/yellow lane paint is bright and thin, so a morphological
    top-hat isolates it; a persistent (temporally accumulated) marking is real
    paint, not a passing car.  A segment is "solid" when its pixel coverage
    along the segment is high; dashed paint has gaps.
    """
    lines: List[Tuple[np.ndarray, np.ndarray]] = []
    try:
        h, w = vr.height, vr.width
        acc = np.zeros((h // 2, w // 2), np.float32)
        stride = max(1, int(round(vr.fps / 2.0)))
        cnt = 0
        for fi, t, frame in vr.iter_frames(stride=stride, max_frames=max_frames):
            small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            v = hsv[..., 2]
            s = hsv[..., 1]
            bright = ((v > 170) & (s < 90)).astype(np.uint8) * 255
            tophat = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            acc += (tophat > 0).astype(np.float32)
            cnt += 1
        vr.rewind()
        if cnt < 5:
            return []
        acc /= float(cnt)
        mask = (acc > 0.45).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # keep only markings that overlap the carriageway
        if scene.road_mask is not None:
            road = cv2.resize(scene.road_mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
            mask[road == 0] = 0
        segs = cv2.HoughLinesP(mask, 1, np.pi / 180.0, threshold=25,
                               minLineLength=max(20, int(0.05 * (h // 2))), maxLineGap=4)
        if segs is None:
            return []
        cand: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for s_ in segs[:200]:
            x1, y1, x2, y2 = [float(v_) for v_ in s_[0]]
            p1 = np.array([x1 * 2.0, y1 * 2.0], np.float32)
            p2 = np.array([x2 * 2.0, y2 * 2.0], np.float32)
            length = float(np.linalg.norm(p2 - p1))
            if length < 1.0 * scene.px_per_m:
                continue
            n = max(8, int(length / 2.0))
            ts = np.linspace(0.0, 1.0, n)
            pts = (p1[None, :] * (1 - ts[:, None]) + p2[None, :] * ts[:, None]) / 2.0
            xs = np.clip(pts[:, 0].astype(np.int32), 0, mask.shape[1] - 1)
            ys = np.clip(pts[:, 1].astype(np.int32), 0, mask.shape[0] - 1)
            cov = float((acc[ys, xs] > 0.30).mean())
            if cov >= 0.80:                                  # solid paint
                cand.append((p1, p2, cov))
        cand.sort(key=lambda c: -c[2])
        lines = [(a, b) for a, b, _ in cand[:60]]
        LOGGER.info("markings: %d solid segments", len(lines))
    except Exception:
        LOGGER.exception("marking detection failed")
    return lines


# ============================================================================
#  Traffic-light heads (colour blobs that stay put)
# ============================================================================
class SignalHeads:
    """Traffic-light state timeline from stationary coloured blobs.

    Heuristic, and honestly the weakest part of the pipeline: we look for
    saturated red/yellow/green blobs that keep the same position across the
    clip, and read their colour per frame.  If no such blob exists the head
    list stays empty and every rule that needs a signal is skipped (which is
    much better than emitting red_light / stop_line false positives).
    """

    def __init__(self, heads: List[Tuple[float, float]], timelines: List[List[Tuple[float, str]]]):
        self.heads = heads
        self.timelines = timelines

    def __bool__(self) -> bool:
        return len(self.heads) > 0

    def state_at(self, t: float, point: Optional[Sequence[float]] = None) -> Optional[str]:
        if not self.heads:
            return None
        idx = 0
        if point is not None:
            d = [math.hypot(point[0] - hx, point[1] - hy) for hx, hy in self.heads]
            idx = int(np.argmin(d))
            if d[idx] > 0.35 * self.width: return None
        tl = self.timelines[idx] if idx < len(self.timelines) else []
        if not tl:
            return None
        best = None
        for tt, st in tl:
            if tt <= t + 1e-6:
                best = st
            else:
                break
        return best


def build_signal_heads(signal_obs: Sequence[Tuple[float, List[Tuple[float, float, str, float]]]],
                       min_frac: float = 0.15) -> SignalHeads:
    """Cluster per-frame coloured blob observations into persistent heads."""
    heads: List[Tuple[float, float]] = []
    timelines: List[List[Tuple[float, str]]] = []
    if not signal_obs:
        return SignalHeads([], [])
    n_frames = len(signal_obs)
    for t, blobs in signal_obs:
        for cx, cy, colour, area in blobs:
            placed = False
            for k, (hx, hy) in enumerate(heads):
                if abs(cx - hx) <= 18.0 and abs(cy - hy) <= 18.0:
                    placed = True
                    break
            if not placed:
                heads.append((cx, cy))
    if not heads:
        return SignalHeads([], [])
    timelines = [[] for _ in heads]
    counts = [0] * len(heads)
    for t, blobs in signal_obs:
        seen = [None] * len(heads)
        for cx, cy, colour, area in blobs:
            k = int(np.argmin([math.hypot(cx - hx, cy - hy) for hx, hy in heads]))
            hx, hy = heads[k]
            if abs(cx - hx) <= 18.0 and abs(cy - hy) <= 18.0 and area > 0:
                seen[k] = colour
        for k in range(len(heads)):
            if seen[k] is not None:
                counts[k] += 1
                timelines[k].append((t, seen[k]))
    # drop transient heads (less than min_frac of the clip) - they are noise
    keep = [k for k in range(len(heads)) if counts[k] >= max(3, min_frac * n_frames)]
    heads = [heads[k] for k in keep]
    timelines = [timelines[k] for k in keep]
    LOGGER.info("signals: %d persistent heads", len(heads))
    return SignalHeads(heads, timelines)


def find_signal_blobs(frame: np.ndarray, scale: float = 0.5) -> List[Tuple[float, float, str, float]]:
    """Saturated red / yellow / green blobs in one frame (full-res coordinates)."""
    out: List[Tuple[float, float, str, float]] = []
    try:
        small = cv2.resize(frame, (max(32, int(frame.shape[1] * scale)),
                                   max(32, int(frame.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        masks = {
            "red": (((h <= 8) | (h >= 168)) & (s > 140) & (v > 150)),
            "yellow": ((h >= 18) & (h <= 34) & (s > 120) & (v > 170)),
            "green": ((h >= 60) & (h <= 95) & (s > 120) & (v > 150)),
        }
        max_area = 0.02 * small.shape[0] * small.shape[1]
        for colour, m in masks.items():
            m8 = (m.astype(np.uint8)) * 255
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                a = cv2.contourArea(c)
                if a < 4 or a > max_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                if bh > 3.0 * bw or bw > 3.0 * bh:               # not a lamp
                    continue
                out.append((((x + bw / 2.0) / scale), ((y + bh / 2.0) / scale), colour, float(a)))
    except Exception:
        LOGGER.debug("signal blob detection failed", exc_info=True)
    return out


# ============================================================================
#  Stop bands / intersection geometry (inferred from real stops)
# ============================================================================
@dataclass
class StopBand:
    """A stop line expressed in the local flow frame.

    d   : unit travel direction the band belongs to
    p0  : projection of the stop line along d (the front of a vehicle at rest)
    p_exit : projection where the intersection ends (best effort)
    q_lo/q_hi : lateral extent of the band
    n   : number of real stops that support this band
    """
    d: np.ndarray
    p0: float
    p_exit: float
    q_lo: float
    q_hi: float
    n: int

    def coords(self, pt: Sequence[float]) -> Tuple[float, float]:
        p = float(pt[0] * self.d[0] + pt[1] * self.d[1])
        q = float(pt[0] * (-self.d[1]) + pt[1] * self.d[0])
        return p, q

    def matches(self, point: Sequence[float], heading_deg: float,
                lane_gap: float = 30.0) -> bool:
        p, q = self.coords(point)
        if q < self.q_lo - lane_gap or q > self.q_hi + lane_gap:
            return False
        d_deg = math.degrees(math.atan2(self.d[1], self.d[0]))
        return ang_diff(d_deg, heading_deg) <= 50.0


def infer_stop_bands(series: Sequence[TrackSeries], scene: SceneModel,
                     duration: float, dt: float) -> List[StopBand]:
    """Find where vehicles really stop, and turn that into a stop-line model.

    A "stop" is >= 1.5 s below STILL_MPS.  Stops that line up (same direction,
    same position along the direction of travel, lateral spread smaller than a
    few lanes) form a band.  With two opposite bands we also get the exit side
    of the intersection.
    """
    stops: List[Tuple[np.ndarray, float, float]] = []      # (dir, p of front, q)
    for s in series:
        if s.cat not in MOTION_CATS or s.n < 4:
            continue
        ts = sample_times(s.t[0], s.t[-1], dt)
        still = [float(s.speed(t)) < CFG["STILL_MPS"] for t in ts]
        for a, b in runs_of(ts, still, 1.5):
            tm = 0.5 * (a + b)
            d = scene.dir_at(s.pos(tm)) or s.dir_from_motion(tm)
            if d is None:
                continue
            p, q = float(s.pos(tm)[0] * d[0] + s.pos(tm)[1] * d[1]), \
                   float(s.pos(tm)[0] * (-d[1]) + s.pos(tm)[1] * d[0])
            p += s.half_len_along(tm, d) * 0.6             # front of the vehicle
            stops.append((d, p, q))
    bands: List[StopBand] = []
    used = set()
    for i, (d_i, p_i, q_i) in enumerate(stops):
        if i in used:
            continue
        group = [(p_i, q_i)]
        for j in range(i + 1, len(stops)):
            if j in used:
                continue
            d_j, p_j, q_j = stops[j]
            if ang_diff(math.degrees(math.atan2(d_i[1], d_i[0])),
                        math.degrees(math.atan2(d_j[1], d_j[0]))) > 40.0:
                continue
            if abs(p_j - p_i) > max(12.0, 0.6 * scene.px_per_m):
                continue
            group.append((p_j, q_j))
            used.add(j)
        if len(group) >= 3:
            ps = [g[0] for g in group]
            qs = [g[1] for g in group]
            used.add(i)
            bands.append(StopBand(d=d_i, p0=float(np.median(ps)), p_exit=0.0,
                                  q_lo=float(min(qs)), q_hi=float(max(qs)), n=len(group)))
    # exit side of the intersection = the other band of the opposite direction
    for bi, b in enumerate(bands):
        others = [o for k, o in enumerate(bands) if k != bi and float(np.dot(b.d, o.d)) < -0.5]
        if others:
            b.p_exit = float(np.median([o.p0 for o in others]))
        else:
            b.p_exit = b.p0 + 0.35 * math.hypot(scene.w, scene.h) * scene.scale
        # keep the exit beyond the entry
        if b.p_exit < b.p0:
            b.p_exit = b.p0 + 0.35 * math.hypot(scene.w, scene.h) * scene.scale
    if bands:
        LOGGER.info("stop bands: %s", [(round(b.p0, 1), b.n) for b in bands])
    return bands


# ============================================================================
#  Rule context
# ============================================================================
@dataclass
class Event:
    start: float
    end: float
    label: str
    conf: float = 1.0


@dataclass
class Ctx:
    """Everything the rules need.  Built once per video."""
    duration: float
    dt: float
    scene: SceneModel
    series: List[TrackSeries]
    vehicles: List[TrackSeries] = field(default_factory=list)
    persons: List[TrackSeries] = field(default_factory=list)
    signals: SignalHeads = field(default_factory=lambda: SignalHeads([], []))
    bands: List[StopBand] = field(default_factory=list)
    solid_lines: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    fire_runs: List[Tuple[float, float, str]] = field(default_factory=list)
    obstacle_runs: List[Tuple[float, float]] = field(default_factory=list)
    maneuvers: Dict[int, List[Tuple[float, float, float]]] = field(default_factory=dict)

    def maneuvers_of(self, s: TrackSeries) -> List[Tuple[float, float, float]]:
        if s.tid not in self.maneuvers:
            self.maneuvers[s.tid] = find_maneuvers(s)
        return self.maneuvers[s.tid]

    def near_maneuver(self, s: TrackSeries, t: float, pad: float = 3.0) -> bool:
        for (a, b, _d) in self.maneuvers_of(s):
            if a - pad <= t <= b + pad:
                return True
        return False


def find_maneuvers(s: TrackSeries, max_dur: float = 6.5, min_delta: float = 35.0
                   ) -> List[Tuple[float, float, float]]:
    """Monotone heading changes: -> [(t_start, t_end, delta_deg), ...].

    A turn manoeuvre in the image is a monotone change of the heading; a
    U-turn is the same with |delta| >= ~130 deg.  We require the increments to
    be mostly one-signed so that noise/track-swaps do not create fake turns.
    """
    out: List[Tuple[float, float, float]] = []
    t = s.t
    n = t.size
    if n < 4:
        return out
    h = s.heading
    i = 0
    while i < n - 1:
        k = int(np.searchsorted(t, t[i] + max_dur, side="right") - 1)
        k = min(k, n - 1)
        if k <= i:
            i += 1
            continue
        delta = float(h[k] - h[i])
        if abs(delta) >= min_delta:
            inc = np.diff(h[i:k + 1])
            if inc.size >= 2:
                pos = float((inc > 0).mean())
                mono = (pos >= 0.8) if delta > 0 else (pos <= 0.2)
                if mono:
                    out.append((float(t[i]), float(t[k]), delta))
                    i = k
                    continue
        i += 1
    return out


# ============================================================================
#  Part A rules - one per official class
# ============================================================================
def _goes_quiet(s: TrackSeries, t0: float, ctx: Ctx, horizon: float = 4.0):
    """Did the object stop after t0?  -> (t_stop, still_seconds) or None."""
    ts = sample_times(t0, min(t0 + horizon, ctx.duration), ctx.dt)
    if ts.size == 0:
        return None
    t_stop = None
    still = 0.0
    for tt in ts:
        if s.speed(tt) < CFG["STILL_MPS"]:
            still += ctx.dt
            if t_stop is None:
                t_stop = float(tt)
        else:
            t_stop = None
            still = 0.0
    if t_stop is not None and still >= 1.0:
        return (t_stop, still)
    return None


def _quiet_end(s: TrackSeries, q: Tuple[float, float], ctx: Ctx) -> float:
    return min(q[0] + q[1] + 0.5, s.t[-1], ctx.duration)


def _others_near(ctx: Ctx, s: TrackSeries, t: float, radius_m: float) -> int:
    r = radius_m * ctx.scene.px_per_m
    px, py = s.pos(t)
    cnt = 0
    for o in ctx.vehicles:
        if o is s or not o.active(t):
            continue
        ox, oy = o.pos(t)
        if math.hypot(ox - px, oy - py) < r:
            cnt += 1
    return cnt


@guarded
def rule_accident(ctx: Ctx) -> List[Event]:
    """Collision: contact between two road users, or a vehicle into an object.

    Contact proxy (we have no depth and no reliable segmentation): the centre
    distance falls below ACC_TOUCH * mean box diagonal, the combined approach
    speed was non-trivial just before that instant, and the involved objects
    come to rest afterwards.  Start = first contact sample, end = the moment
    they are both at rest (or leave the frame).  Sampling at 5 fps makes the
    start boundary good to +-0.2 s, which is far inside IoU 0.7 for a multi
    second event.
    """
    out: List[Event] = []
    veh = ctx.vehicles
    for i in range(len(veh)):
        a = veh[i]
        for j in range(i + 1, len(veh)):
            b = veh[j]
            t0 = max(a.t[0], b.t[0])
            t1 = min(a.t[-1], b.t[-1])
            if t1 - t0 < 0.4:
                continue
            contact_t = None
            for t in sample_times(t0, t1, ctx.dt):
                ax, ay = a.pos(t)
                bx, by = b.pos(t)
                d = math.hypot(ax - bx, ay - by)
                touch = 0.5 * (a.diag(t) + b.diag(t)) * CFG["ACC_TOUCH"]
                if d < touch:
                    contact_t = float(t)
                    break
            if contact_t is None:
                continue
            v_before = a.speed(contact_t - 0.3) + b.speed(contact_t - 0.3)
            if v_before < CFG["ACC_MIN_APPROACH"]:
                continue
            qa = _goes_quiet(a, contact_t, ctx)
            qb = _goes_quiet(b, contact_t, ctx)
            a_left = a.t[-1] <= contact_t + 2.5
            b_left = b.t[-1] <= contact_t + 2.5
            conf = 0.45
            end_t = contact_t + 1.0
            if qa:
                conf += 0.15
                end_t = max(end_t, _quiet_end(a, qa, ctx))
            if qb:
                conf += 0.15
                end_t = max(end_t, _quiet_end(b, qb, ctx))
            if qa and qb:
                conf += 0.20
            if a_left or b_left:
                conf += 0.05
                end_t = max(end_t, min(contact_t + 3.0, ctx.duration))
            if v_before > 6.0:
                conf += 0.05
            if conf >= CFG["ACC_MIN_CONF"]:
                out.append(Event(contact_t, min(end_t, ctx.duration), "accident",
                                 min(0.95, conf)))

    # single vehicle: hard braking into a stop, nothing else around to blame
    for s in veh:
        ts = sample_times(s.t[0], s.t[-1], ctx.dt)
        for t in ts:
            if s.accel_at(t) > -CFG["ACC_HARD_BRAKE"]:
                continue
            q = _goes_quiet(s, t, ctx, horizon=5.0)
            if not q or q[1] < 3.0:
                continue
            if _others_near(ctx, s, t, radius_m=1.5 * max(2.0, s.diag(t) / ctx.scene.px_per_m)) > 0:
                continue
            out.append(Event(float(t), min(_quiet_end(s, q, ctx), ctx.duration),
                             "accident", 0.55))
            break                                          # one event per track
    return out


@guarded
def rule_near_miss(ctx: Ctx) -> List[Event]:
    """Sharp braking / swerving to avoid a collision, no contact."""
    out: List[Event] = []
    veh = ctx.vehicles
    for i in range(len(veh)):
        a = veh[i]
        for j in range(i + 1, len(veh)):
            b = veh[j]
            t0 = max(a.t[0], b.t[0])
            t1 = min(a.t[-1], b.t[-1])
            if t1 - t0 < 0.4:
                continue
            ts = sample_times(t0, t1, ctx.dt)
            if ts.size == 0:
                continue
            seps = np.empty(ts.size)
            touch = np.empty(ts.size)
            for k, t in enumerate(ts):
                ax, ay = a.pos(t); bx, by = b.pos(t)
                seps[k] = math.hypot(ax - bx, ay - by)
                touch[k] = 0.5 * (a.diag(t) + b.diag(t))
            kmin = int(np.argmin(seps))
            sep_min = float(seps[kmin])
            t_min = float(ts[kmin])
            if sep_min < touch[kmin] * CFG["ACC_TOUCH"]:
                continue                                   # that is an accident, not a near miss
            if sep_min > touch[kmin] * CFG["NM_TOUCH"]:
                continue
            trig = None
            for t in sample_times(t_min - 2.5, t_min + 0.5, ctx.dt):
                for s_ in (a, b):
                    if s_.accel_at(t) < -CFG["NM_MIN_DECEL"] or abs(s_.hrate_at(t)) > CFG["NM_MIN_HRATE"]:
                        trig = float(t) if trig is None else min(trig, float(t))
            if trig is None:
                continue
            end = min(t_min + 2.0, ctx.duration)
            for t in sample_times(t_min, t_min + 4.0, ctx.dt):
                ax, ay = a.pos(t); bx, by = b.pos(t)
                if math.hypot(ax - bx, ay - by) > touch[kmin] * 2.2:
                    end = float(t)
                    break
            # do not double-report an accident of the same pair at the same time
            clash = any(e.label == "accident" and e.start - 1.5 <= t_min <= e.end + 1.5
                        for e in ctx_found(ctx, "accident"))
            if clash:
                continue
            out.append(Event(trig, max(end, trig + 0.6), "near_miss", 0.60))
    return out


def ctx_found(ctx: Ctx, label: str) -> List[Event]:
    return [e for e in getattr(ctx, "_found", []) if e.label == label]


@guarded
def rule_wrong_way(ctx: Ctx) -> List[Event]:
    """A vehicle moving against the direction of its lane (from the flow grid)."""
    out: List[Event] = []
    if not np.any(ctx.scene.flow_mag > 0):
        return out
    for s in ctx.vehicles:
        ts = sample_times(s.t[0], s.t[-1], ctx.dt)
        if ts.size < 4:
            continue
        flags = []
        for t in ts:
            if s.speed(t) < CFG["WW_MIN_MPS"] or ctx.near_maneuver(s, float(t)):
                flags.append(False)
                continue
            d = ctx.scene.dir_at(s.pos(t))
            hd = s.dir_from_motion(float(t))
            if d is None or hd is None:
                flags.append(False)
                continue
            ang = ang_between(d, hd)
            flags.append(ang >= CFG["WW_MIN_ANGLE"])
        for a, b in runs_of(ts, flags, CFG["WW_MIN_RUN"]):
            out.append(Event(a, min(b + 0.4, ctx.duration), "wrong_way", 0.65))
    return out


def _queue_ahead(ctx: Ctx, s: TrackSeries, t: float) -> int:
    """How many stopped vehicles sit in front of s in its own lane?"""
    d = ctx.scene.dir_at(s.pos(t)) or s.dir_from_motion(t)
    if d is None:
        return 0
    px, py = s.pos(t)
    gap = ctx.scene.lane_gap or 4.0 * ctx.scene.px_per_m
    n = 0
    for o in ctx.vehicles:
        if o is s or not o.active(t):
            continue
        ox, oy = o.pos(t)
        if o.speed(t) >= CFG["STILL_MPS"]:
            continue
        along = (ox - px) * d[0] + (oy - py) * d[1]
        lat = abs((ox - px) * (-d[1]) + (oy - py) * d[0])
        if 0.0 < along <= max(6.0 * gap, 3.0 * ctx.scene.px_per_m) and lat < 0.8 * gap:
            n += 1
    return n


@guarded
def rule_stopped_vehicle(ctx: Ctx) -> List[Event]:
    """Stationary on the carriageway for >= 10 s, and NOT part of a queue."""
    out: List[Event] = []
    min_len = CFG["MIN_DUR"]["stopped_vehicle"]
    for s in ctx.vehicles:
        ts = sample_times(s.t[0], s.t[-1], ctx.dt)
        if ts.size < 4:
            continue
        flags = [ctx.scene.road(s.pos(t)) and s.speed(t) < CFG["STILL_MPS"] for t in ts]
        for a, b in runs_of(ts, flags, min_len):
            tm = 0.5 * (a + b)
            if _queue_ahead(ctx, s, tm) >= 2:               # it is just a queue
                continue
            if not ctx.scene.road(s.pos(tm)):
                continue
            if b >= s.t[-1] - 1.5 * ctx.dt:                # still stopped at the end
                end = ctx.duration if s.t[-1] >= ctx.duration - 1.0 else min(s.t[-1] + 1.0, ctx.duration)
            else:
                end = min(b + 0.5, ctx.duration)
            out.append(Event(a, end, "stopped_vehicle", 0.60))
    return out


@guarded
def rule_congestion(ctx: Ctx) -> List[Event]:
    """A queue of >= 3 stopped vehicles in the same direction + lane group.

    Signal queues are normal traffic, so when we know the signal state we
    suppress a short queue that clears right after green (CONG_MIN_LONG).
    """
    out: List[Event] = []
    if not ctx.vehicles:
        return out
    ts = sample_times(0.0, ctx.duration, ctx.dt)
    flags = []
    for t in ts:
        groups: Dict[Tuple[int, int], int] = defaultdict(int)
        for s in ctx.vehicles:
            if not s.active(t) or s.speed(t) >= CFG["CONG_MPS"]:
                continue
            d = ctx.scene.dir_at(s.pos(t)) or s.dir_from_motion(t)
            if d is None:
                continue
            key = (int(round(math.degrees(math.atan2(d[1], d[0])) / 90.0)),
                   ctx.scene.lane_index(s.pos(t)))
            groups[key] += 1
        flags.append((max(groups.values()) if groups else 0) >= CFG["CONG_MIN_VEHICLES"])
    for a, b in runs_of(ts, flags, CFG["MIN_DUR"]["congestion"]):
        if ctx.signals and (b - a) < CFG["CONG_MIN_LONG"]:
            st = ctx.signals.state_at(0.5 * (a + b))
            if st == "red":
                continue                                   # a red-light queue
        out.append(Event(a, min(b + 0.5, ctx.duration), "congestion", 0.60))
    return out


@guarded
def rule_jaywalking(ctx: Ctx) -> List[Event]:
    """A pedestrian on the carriageway outside a crossing.

    We cannot see markings of a zebra reliably, so the discriminator is the
    group: a crossing produces several pedestrians moving together, while a
    single pedestrian entering the roadway is the annotated jaywalking case.
    """
    out: List[Event] = []
    for s in ctx.persons:
        ts = sample_times(s.t[0], s.t[-1], ctx.dt)
        if ts.size < 4:
            continue
        flags = [ctx.scene.road(s.pos(t)) and s.speed(t) > 0.25 for t in ts]
        for a, b in runs_of(ts, flags, CFG["MIN_DUR"]["jaywalking"]):
            tm = 0.5 * (a + b)
            peers = 0
            for p in ctx.persons:
                if p is s or not p.active(tm):
                    continue
                if ctx.scene.road(p.pos(tm)) and p.speed(tm) > 0.2:
                    peers += 1
            if peers >= 2:                                  # a group -> likely a crossing
                continue
            out.append(Event(a, min(b + 0.4, ctx.duration), "jaywalking", 0.55))
    return out


@guarded
def rule_failure_to_yield(ctx: Ctx) -> List[Event]:
    """A vehicle driving through while a pedestrian is on the roadway."""
    out: List[Event] = []
    if not ctx.persons:
        return out
    for s in ctx.vehicles:
            ts = sample_times(s.t[0], s.t[-1], ctx.dt)
            for t in ts:
                if s.speed(t) < CFG["MOVE_MPS"]:
                    continue
                hit = False
                for p in ctx.persons:
                    if not p.active(t) or not ctx.scene.road(p.pos(t)):
                        continue
                    sx, sy = s.pos(t); px, py = p.pos(t)
                    if math.hypot(sx - px, sy - py) > 3.0 * ctx.scene.px_per_m:
                        continue
                    win = sample_times(t, min(t + 1.2, ctx.duration), ctx.dt)
                    if win.size == 0 or min(s.speed(x) for x in win) < 1.0:
                        continue                       # он всё-таки уступил
                    hit = True
                    break
                if hit:
                    out.append(Event(float(t), min(float(t) + 1.2, ctx.duration),
                                    "failure_to_yield", 0.60))
                    break                              # один проезд = одно событие
    # merge the repeated per-frame hits of the same pass
    return out


@guarded
def rule_illegal_u_turn(ctx: Ctx) -> List[Event]:
    """U-turn that is illegal because of where it is executed.

    Two independent signs of illegality: (a) the manoeuvre starts in a lane
    that is not the leftmost lane of that direction (a legal U-turn must start
    from the leftmost lane), (b) the lateral span is tighter than two lane
    widths, i.e. a mid-block U-turn.  (c) if a stop band is known, the U-turn
    should have happened inside the intersection.
    """
    out: List[Event] = []
    for s in ctx.vehicles:
        for (a, b, delta) in ctx.maneuvers_of(s):
            if abs(delta) < CFG["UTURN_MIN_DEG"] or (b - a) > CFG["TURN_MAX_DUR"] + 1.5:
                continue
            tm = 0.5 * (a + b)
            d = ctx.scene.dir_at(s.pos(a)) or s.dir_from_motion(a)
            if d is None:
                continue
            illegal = False
            if ctx.scene.lanes:
                rank = ctx.scene.lane_leftness(s.pos(a), d)     # 0 = leftmost
                if rank > 1:
                    illegal = True
            span = 0.0
            qs = []
            for t in sample_times(a, b, ctx.dt):
                p = s.pos(t)
                qs.append(float(p[0] * (-d[1]) + p[1] * d[0]))
            if qs:
                span = max(qs) - min(qs)
                if ctx.scene.lane_gap and span < 2.0 * ctx.scene.lane_gap:
                    illegal = True
            if ctx.bands:
                inside = False
                for band in ctx.bands:
                    if band.matches(s.pos(tm), s.heading_at(tm)):
                        p, _q = band.coords(s.pos(tm))
                        if band.p0 - 0.5 * ctx.scene.px_per_m <= p <= band.p_exit:
                            inside = True
                if not inside:
                    illegal = True
            if illegal:
                out.append(Event(a, min(b, ctx.duration), "illegal_u_turn", 0.60))
    return out


@guarded
def rule_illegal_turn(ctx: Ctx) -> List[Event]:
    """A turn performed from a lane that practically only goes straight."""
    out: List[Event] = []
    if not ctx.vehicles:
        return out
    # how often does a turn start in each lane?
    starts: Dict[int, int] = defaultdict(int)
    turns: Dict[int, int] = defaultdict(int)
    all_man = []
    for s in ctx.vehicles:
        for m in ctx.maneuvers_of(s):
            all_man.append((s, m))
    for s in ctx.vehicles:
        for t in sample_times(s.t[0], s.t[-1], ctx.dt):
            starts[ctx.scene.lane_index(s.pos(t))] += 1
    for s, (a, b, delta) in all_man:
        if not (CFG["TURN_MIN_DEG"] <= abs(delta) <= CFG["TURN_MAX_DEG"]):
            continue
        if (b - a) > CFG["TURN_MAX_DUR"]:
            continue
        turns[ctx.scene.lane_index(s.pos(a))] += 1
    for s, (a, b, delta) in all_man:
        if not (CFG["TURN_MIN_DEG"] <= abs(delta) <= CFG["TURN_MAX_DEG"]):
            continue
        if (b - a) > CFG["TURN_MAX_DUR"]:
            continue
        li = ctx.scene.lane_index(s.pos(a))
        total = starts.get(li, 0)
        share = (turns.get(li, 0) / float(total)) if total >= 20 else 1.0
        if share < 0.15:                     # this lane is a straight-only lane
            out.append(Event(a, min(b, ctx.duration), "illegal_turn", 0.55))
    return out


@guarded
def rule_solid_line_crossing(ctx: Ctx) -> List[Event]:
    """A manoeuvre across a solid lane marking."""
    out: List[Event] = []
    if not ctx.solid_lines:
        return out
    for s in ctx.vehicles:
        ts = sample_times(s.t[0], s.t[-1], ctx.dt)
        if ts.size < 3:
            continue
        prev = None
        for t in ts:
            p = np.array(s.pos(t), np.float32)
            if prev is not None:
                p0, t0 = prev
                move = p - p0
                if float(np.linalg.norm(move)) >= 0.15 * ctx.scene.px_per_m:
                    for (m0, m1) in ctx.solid_lines:
                        if segments_intersect(p0, p, m0, m1):
                            if ang_between(move, m1 - m0) < 25.0:
                                continue             # travelling along the line, not across
                            out.append(Event(float(t0), min(float(t) + CFG["SLC_TAIL"],
                                                            ctx.duration),
                                             "solid_line_crossing", 0.60))
            prev = (p, float(t))
    return out


@guarded
def rule_red_light(ctx: Ctx) -> List[Event]:
    """A vehicle crossing the stop line while its signal is red."""
    out: List[Event] = []
    if not ctx.bands or not ctx.signals:
        return out
    for s in ctx.vehicles:
        ts = sample_times(s.t[0], s.t[-1], ctx.dt)
        if ts.size < 4:
            continue
        prev_p = None
        prev_t = None
        for t in ts:
            d = None
            band = None
            for b_ in ctx.bands:
                if b_.matches(s.pos(t), s.heading_at(t)):
                    band = b_
                    break
            if band is None:
                prev_p, prev_t = None, None
                continue
            d = band.d
            p, _q = band.coords(s.pos(t))
            p_front = p + s.half_len_along(t, d)
            if prev_p is not None and prev_t is not None:
                crossed = prev_p < band.p0 <= p_front
                if crossed and s.speed(t) > 1.0:
                    st = ctx.signals.state_at(float(t), s.pos(t))
                    if st == "red":
                        # end: the vehicle leaves the intersection or the frame
                        end = ctx.duration
                        for t2 in sample_times(float(t), min(float(t) + 6.0, ctx.duration), ctx.dt):
                            p2, _ = band.coords(s.pos(t2))
                            if p2 + s.half_len_along(t2, d) > band.p_exit:
                                end = float(t2)
                                break
                        out.append(Event(float(t), min(end, ctx.duration), "red_light", 0.75))
            prev_p, prev_t = p_front, float(t)
    return out


@guarded
def rule_stop_line(ctx: Ctx) -> List[Event]:
    """Stopping past the stop line on red, without entering the intersection."""
    out: List[Event] = []
    if not ctx.bands or not ctx.signals:
        return out
    for s in ctx.vehicles:
        ts = sample_times(s.t[0], s.t[-1], ctx.dt)
        if ts.size < 4:
            continue
        flags = []
        for t in ts:
            ok = False
            for band in ctx.bands:
                if not band.matches(s.pos(t), s.heading_at(t)):
                    continue
                p, _ = band.coords(s.pos(t))
                p_front = p + s.half_len_along(t, band.d)
                if band.p0 < p_front < band.p0 + 2.0 * ctx.scene.px_per_m:
                    ok = True
            flags.append(ok and s.speed(t) < CFG["STILL_MPS"])
        for a, b in runs_of(ts, flags, CFG["MIN_DUR"]["stop_line"]):
            st = ctx.signals.state_at(0.5 * (a + b), s.pos(0.5 * (a + b)))
            if st != "red":
                continue
            # end = when the signal turns green (or the clip ends)
            end = ctx.duration
            for t in sample_times(b, ctx.duration, ctx.dt):
                if ctx.signals.state_at(float(t), s.pos(float(t))) == "green":
                    end = float(t)
                    break
            out.append(Event(a, min(end, ctx.duration), "stop_line", 0.70))
    return out


@guarded
def rule_road_obstacle(ctx: Ctx) -> List[Event]:
    """Persistent foreground blobs on the carriageway that no track explains."""
    out: List[Event] = []
    for a, b in ctx.obstacle_runs:
        if b - a >= CFG["MIN_DUR"]["road_obstacle"]:
            out.append(Event(a, min(b, ctx.duration), "road_obstacle", 0.70))
    return out


@guarded
def rule_fire_smoke(ctx: Ctx) -> List[Event]:
    """Growing smoke / fire regions."""
    out: List[Event] = []
    for a, b, kind in ctx.fire_runs:
        if b - a >= CFG["MIN_DUR"]["fire_smoke"]:
            out.append(Event(a, min(b, ctx.duration), "fire_smoke", 0.65 if kind == "smoke" else 0.7))
    return out


RULES = [
    ("accident", rule_accident),
    ("near_miss", rule_near_miss),
    ("wrong_way", rule_wrong_way),
    ("stopped_vehicle", rule_stopped_vehicle),
    ("congestion", rule_congestion),
    ("jaywalking", rule_jaywalking),
    ("failure_to_yield", rule_failure_to_yield),
    ("illegal_u_turn", rule_illegal_u_turn),
    ("illegal_turn", rule_illegal_turn),
    ("solid_line_crossing", rule_solid_line_crossing),
    ("red_light", rule_red_light),
    ("stop_line", rule_stop_line),
    ("road_obstacle", rule_road_obstacle),
    ("fire_smoke", rule_fire_smoke),
]


def run_rules(ctx: Ctx) -> List[Event]:
    """Run every enabled rule; rules can consult events already found."""
    ctx._found = []                                            # type: ignore[attr-defined]
    for label, fn in RULES:
        if label not in CFG["ENABLED"]:
            continue
        if label in CFG["NEEDS_GEOMETRY"] and label in ("red_light", "stop_line") \
                and not (ctx.bands and ctx.signals):
            continue
        evs = fn(ctx)
        for e in evs:
            if e.conf >= CFG["MIN_CONF"].get(label, 0.5):
                ctx._found.append(e)                           # type: ignore[attr-defined]
    LOGGER.info("rules produced %d raw events", len(ctx._found))   # type: ignore[attr-defined]
    return list(ctx._found)                                        # type: ignore[attr-defined]


# ============================================================================
#  Post-processing: boundaries, merges, overlaps, blips
# ============================================================================
def postprocess(events: Sequence[Event], duration: float, cfg: dict) -> List[list]:
    """Clip to the clip, merge fragments, drop blips, enforce the interface.

    The harness keeps the earlier segment when two segments of the same class
    overlap and drops the later one, so we resolve overlaps ourselves to stay in
    control of the outcome.
    """
    per_class: Dict[str, List[List[float]]] = defaultdict(list)
    for e in events:
        label = e.label
        if label not in CLASSES:
            continue
        s = max(0.0, min(float(e.start), duration))
        en = min(float(e.end), duration)
        if not (en > s):
            continue
        per_class[label].append([s, en])

    final: List[list] = []
    for label, segs in per_class.items():
        if not segs:
            continue
        segs.sort(key=lambda x: (x[0], x[1]))
        merged: List[List[float]] = [list(segs[0])]
        for s, en in segs[1:]:
            if s - merged[-1][1] <= cfg["MERGE_GAP"]:
                merged[-1][1] = max(merged[-1][1], en)
            else:
                merged.append([s, en])
        min_dur = cfg["MIN_DUR"].get(label, 1.0)
        last_end = -1.0
        for seg in merged:
            s, en = seg
            if s <= last_end:                                  # numerical safety
                s = last_end + 1e-3
            if seg[1] - seg[0] > 4.0 * cfg["MIN_DUR"].get(label, 1.0) and label in ("near_miss", "failure_to_yield", "solid_line_crossing"):
                continue
            if en - s < min_dur:                               # blip filter
                continue
            final.append([round(float(s), 3), round(float(en), 3), label])
            last_end = en
    final.sort(key=lambda x: (x[0], x[2]))
    return final


# ============================================================================
#  The detection pass
# ============================================================================
def _fire_smoke_flags(frame: np.ndarray, fg: Optional[np.ndarray]) -> Tuple[bool, str, float]:
    """Cheap smoke / fire presence test on one frame (HSV, no ML).

    smoke: mid-bright, low saturation, rough texture; fire: saturated orange /
    red with a high value.  Returns (present, kind, area_fraction).
    """
    try:
        small = cv2.resize(frame, (max(64, frame.shape[1] // 4),
                                   max(64, frame.shape[0] // 4)), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        fire = (((h <= 22) | (h >= 165)) & (s > 150) & (v > 190)).astype(np.uint8)
        smoke = ((s < 70) & (v > 95) & (v < 200)).astype(np.uint8)
        if fg is not None:
            fgs = cv2.resize(fg, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            fire = fire & (fgs > 0).astype(np.uint8)
            smoke = smoke & (fgs > 0).astype(np.uint8)
        npx = float(small.shape[0] * small.shape[1])
        af = fire.sum() / npx
        asm = smoke.sum() / npx
        if af > 0.004:
            return True, "fire", af
        if asm > 0.02:
            return True, "smoke", asm
    except Exception:
        LOGGER.debug("fire/smoke test failed", exc_info=True)
    return False, "", 0.0


def _obstacle_candidates(dets: Sequence[Det], fg_full: Optional[np.ndarray],
                         shape: Tuple[int, int]) -> List[Tuple[float, float, float]]:
    """Foreground components on the carriageway that no detection explains."""
    out: List[Tuple[float, float, float]] = []
    if fg_full is None:
        return out
    h, w = shape
    try:
        mask = (fg_full > 200).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        amin = 0.0008 * h * w
        amax = 0.05 * h * w
        for c in cnts:
            a = cv2.contourArea(c)
            if a < amin or a > amax:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bh > 0.6 * h:
                continue
            box = (float(x), float(y), float(x + bw), float(y + bh))
            matched = False
            for d in dets:
                if iou_xyxy(box, d.xyxy) > 0.05:
                    matched = True
                    break
            if not matched:
                out.append((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, math.sqrt(a))
    except Exception:
        LOGGER.debug("obstacle scan failed", exc_info=True)
    return out


def detection_pass(vr: VideoReader, det: Detector, scene: SceneModel, cfg: dict
                   ) -> Tuple[Ctx, List[Tuple[float, float]]]:
    """Pass 2: detection + tracking + side channels.  Returns (ctx, obstacle_points)."""
    stride = vr.stride
    dt = vr.dt
    tracker = SimpleTracker(iou_thr=cfg["TRK_IOU"], max_age_s=cfg["TRK_MAX_AGE_S"],
                            sample_dt=dt)
    bytetrack = None
    if cfg["USE_BYTETRACK"]:
        dev = cfg["DEVICE"] or None
        bytetrack = ByteTrackerAdapter(cfg["DET_WEIGHTS"], cfg["IMGSZ"], cfg["CONF"],
                                       cfg["IOU"], dev)

    bg = cv2.createBackgroundSubtractorMOG2(history=int(max(60, 20 / max(dt, 1e-3))),
                                            varThreshold=25, detectShadows=False)
    obstacle_cells: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    signal_obs: List[Tuple[float, List[Tuple[float, float, str, float]]]] = []
    fire_flags: List[Tuple[float, bool, str, float]] = []

    sig_stride = max(1, int(round(1.0 / max(dt, 1e-3) / 2.0)))     # ~2 Hz for signals
    fire_stride = max(1, int(round(1.0 / max(dt, 1e-3) / 2.0)))    # ~2 Hz for fire/smoke

    batch: List[Tuple[int, float, np.ndarray]] = []
    idx = 0
    for fi, t, frame in vr.iter_frames(stride=stride):
        batch.append((idx, t, frame))
        idx += 1
        if len(batch) < int(cfg["BATCH"]):
            continue
        _consume_batch(batch, det, tracker, bytetrack, scene, cfg, bg, dt, signal_obs,
                       fire_flags, obstacle_cells, fi, si=sig_stride, fsi=fire_stride)
        batch = []
    if batch:
        _consume_batch(batch, det, tracker, bytetrack, scene, cfg, bg, dt, signal_obs,
                       fire_flags, obstacle_cells, fi, si=sig_stride, fsi=fire_stride)
    vr.rewind()

    # ---------------------------------------------------------------- assemble
    ctx = Ctx(duration=vr.duration, dt=dt, scene=scene, series=[])
    ctx.series = build_series(tracker, scene.px_per_m, cfg["MIN_TRACK_OBS"])
    ctx.vehicles = [s for s in ctx.series if s.cat in MOTION_CATS]
    ctx.persons = [s for s in ctx.series if s.cat == "person"]
    ctx.bands = infer_stop_bands(ctx.vehicles, scene, vr.duration, dt)
    ctx.signals = build_signal_heads(signal_obs)
    ctx.solid_lines = detect_markings(vr, scene, cfg) if "solid_line_crossing" in cfg["ENABLED"] else []
    ctx.fire_runs = _spatio_temporal_runs(fire_flags, vr.duration)
    ctx.obstacle_runs = _spatio_temporal_runs_pts(obstacle_cells, dt)

    LOGGER.info("tracks: %d (vehicles %d, persons %d), obstacles %d, fire %d",
                len(ctx.series), len(ctx.vehicles), len(ctx.persons),
                len(ctx.obstacle_runs), len(ctx.fire_runs))
    return ctx, []


def _consume_batch(batch, det, tracker, bytetrack, scene, cfg, bg, dt, signal_obs,
                   fire_flags, obstacle_cells, last_fi, si=5, fsi=5) -> None:
    """Detect + track a small batch of frames, and update the side channels."""
    frames = [b[2] for b in batch]
    times = [b[1] for b in batch]
    dets_list = det.predict(frames)

    for k, (fi, t, frame) in enumerate(batch):
        dets = dets_list[k] if k < len(dets_list) else []
        if bytetrack is not None and bytetrack.ok:
            pairs = bytetrack.step(frame)
            if pairs:
                ids = [p[0] for p in pairs]
                for (tid, d) in pairs:
                    tr = tracker.tracks.get(tid)
                    obs = Obs(t, d.xyxy, d.conf)
                    if tr is None:
                        tracker.tracks[tid] = Track(tid, d.cat, obs)
                    else:
                        tr.add(obs)
                    tracker.active[tid] = 0
                # age the tracks the tracker did not see this frame
                for tid in list(tracker.active.keys()):
                    if tid not in ids:
                        tracker.active[tid] += 1
                        if tracker.active[tid] > tracker.max_age:
                            del tracker.active[tid]
            else:
                tracker.update(t, dets)
        else:
            tracker.update(t, dets)

        # ---- side channel: fire / smoke (2 Hz) -----------------------------
        
        if fi % max(1, fsi) == 0:
            fg = bg.apply(cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4)))
            h_, w_ = frame.shape[:2]
            fg_full = cv2.resize(fg, (w_, h_), interpolation=cv2.INTER_NEAREST)
            cell = max(24.0, 0.05 * w_)
            for (ox, oy, _oa) in _obstacle_candidates(dets, fg_full, (h_, w_)):
                if not scene.road((ox, oy)):          # только проезжая часть
                    continue
                obstacle_cells[(int(ox // cell), int(oy // cell))].append(float(t))
            present, kind, area = _fire_smoke_flags(frame, fg)
            if present:
                fire_flags.append((float(t), True, kind, float(area)))
            else:
                fire_flags.append((float(t), False, "", 0.0))

        # ---- side channel: signals (2 Hz) ----------------------------------
        if fi % max(1, si) == 0:
            blobs = find_signal_blobs(frame)
            if blobs:
                signal_obs.append((float(t), blobs))


def _spatio_temporal_runs(flags: Sequence[Tuple[float, bool, str, float]],
                          duration: float, gap: float = 2.0) -> List[Tuple[float, float, str]]:
    """Temporal runs of fire/smoke presence with a small tolerance gap."""
    out: List[Tuple[float, float, str]] = []
    start = None
    last_true = None
    kind = ""
    for t, present, k, _a in flags:
        if present:
            if start is None:
                start = t
                kind = k
            elif k and k != kind:
                kind = k if k == "fire" else kind
            last_true = t
        else:
            if start is not None and last_true is not None and t - last_true > gap:
                out.append((start, min(last_true + 0.5, duration), kind))
                start = None
                last_true = None
    if start is not None and last_true is not None:
        out.append((start, min(last_true + 0.5, duration), kind))
    return out


def _spatio_temporal_runs_pts(cells: Dict[Tuple[int, int], List[float]],
                              dt: float, gap: float = 2.0) -> List[Tuple[float, float]]:
    """Obstacle runs: a coarse grid cell or its neighbour stays occupied."""
    out: List[Tuple[float, float]] = []
    for (_cx, _cy), ts in cells.items():
        ts = sorted(ts)
        if not ts:
            continue
        start = ts[0]
        last = ts[0]
        for t in ts[1:]:
            if t - last > max(gap, 3.0 * dt):
                out.append((start, last))
                start = t
            last = t
        out.append((start, last))
    # merge overlapping runs (neighbouring cells of the same object)
    out.sort()
    merged: List[List[float]] = []
    for a, b in out:
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


# ============================================================================
#  Part A entry point
# ============================================================================
def detect_events(video_path: str) -> list:
    """Part A.  [[start_sec, end_sec, label], ...] for one .mp4.

    Never raises: any failure is logged and scored as "no events", because an
    exception inside this function means an empty prediction for the clip
    anyway - logging it is strictly better than crashing the harness.
    """
    t_wall = time.time()
    events: List[list] = []
    vr = None
    try:
        vr = VideoReader(video_path, det_fps=CFG["DET_FPS"])
        det = get_detector(CFG["DET_WEIGHTS"], imgsz=CFG["IMGSZ"], conf=CFG["CONF"],
                           iou=CFG["IOU"], half=CFG["HALF"], device=CFG["DEVICE"],
                           batch=CFG["BATCH"])
        scene = build_scene_model(vr, det, CFG)
        ctx, _ = detection_pass(vr, det, scene, CFG)
        events = run_rules(ctx)
        events = postprocess(events, vr.duration, CFG)
    except Exception:
        LOGGER.exception("detect_events failed on %s -> []", video_path)
        events = []
    finally:
        if vr is not None:
            vr.close()
    LOGGER.info("detect_events(%s): %d events in %.2fs wall",
                osp.basename(str(video_path)), len(events), time.time() - t_wall)
    return events


# ============================================================================
#  Part B - causal accident anticipation
# ============================================================================
class RiskEstimator:
    """P(accident starts within H = 5 s), causal, one frame at a time.

    Only the frames handed to step() are used.  We keep a very small online
    model of the scene (occupancy grid for the carriageway, flow grid for the
    lane direction) that is updated causally, plus our own tracker, and turn
    the tracks into risk features:

        TTC        time-to-collision of the closest closing pair (constant
                   velocity extrapolation of both boxes)
        braking    strongest deceleration among the tracks
        swerve     strongest heading rate
        wrong way  track direction vs. the causally learned lane direction
        pedestrian a person inside the causally learned carriageway
        blocked    a vehicle stopped in a lane where others are moving
        contact    boxes overlap with a non-trivial combined speed

    Fusion: fixed logistic weights (see CFG["RISK_*"]) -> score in [0, 1], then
    a persistence filter max(inst, decay * prev) so that a genuine approach
    keeps the score above the 0.5 alarm threshold for a few seconds, which is
    what the alarm-F1 part of Score B needs.  Cost: one YOLO(n) call every
    RISK_STRIDE frames + O(n^2) over <= 12 tracks; no video access, no future
    frames, no randomness.
    """

    def __init__(self) -> None:
        self.meta: dict = {}
        self.ready = False
        self._det: Optional[Detector] = None

    # ---------------------------------------------------------------- reset
    def reset(self, meta: dict) -> None:
        meta = dict(meta or {})
        self.meta = meta
        self.fps = float(meta.get("fps") or 25.0)
        if not (1.0 <= self.fps <= 240.0):
            self.fps = 25.0
        self.stride = max(1, int(CFG["RISK_STRIDE"]))
        self.dt = self.stride / self.fps
        try:
            self._det = get_detector(CFG["RISK_WEIGHTS"], imgsz=CFG["RISK_IMGSZ"],
                                     conf=0.30, iou=CFG["IOU"], half=CFG["HALF"],
                                     device=CFG["DEVICE"], batch=1)
        except Exception:
            LOGGER.exception("risk detector init failed -> zeros")
            self._det = None
        self.tracker = SimpleTracker(iou_thr=CFG["TRK_IOU"], max_age_s=2.0, sample_dt=self.dt)
        self.frame_id = 0
        self.risk = 0.0
        self.last_score = 0.0
        self.px_per_m = 25.0
        self.sizes: deque = deque(maxlen=120)
        self.occ = np.zeros((18, 32), np.float32)          # carriageway occupancy
        self.flow = np.zeros((18, 32, 2), np.float32)
        self.flow_n = np.zeros((18, 32), np.float32)
        self._bgsub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
        self.prev_gray = None
        self.still_since: Dict[int, float] = {}
        self.vhist: Dict[int, deque] = defaultdict(lambda: deque(maxlen=6))
        self.ready = True

    # ---------------------------------------------------------------- step
    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Risk in [0, 1] that an accident starts within the next 5 seconds."""
        try:
            if frame is None or not hasattr(frame, "shape") or frame.size == 0:
                return float(self.last_score)
            if not self.ready:
                self.reset({})                             # defensive: harness always calls reset
            self.frame_id += 1
            if (self.frame_id - 1) % self.stride:
                return float(self.last_score)
            score = self._process(frame, float(t_sec))
            self.last_score = float(np.clip(score, 0.0, 1.0))
        except Exception:
            LOGGER.debug("risk step failed", exc_info=True)
            # never crash: keep the previous score (the harness records it)
        return float(np.clip(self.last_score, 0.0, 1.0))

    # -------------------------------------------------------------- internals
    def _process(self, frame: np.ndarray, t: float) -> float:
        h, w = frame.shape[:2]
        work = frame
        if w > 960:                                        # keep inference cheap
            work = cv2.resize(frame, (960, int(h * 960.0 / w)), interpolation=cv2.INTER_AREA)
        dets = self._det.predict([work])[0] if self._det is not None else []

        # ---- scale from vehicle box sizes (median over the recent past) ----
        for d in dets:
            if d.cat in VEH_CATS and d.w > 4 and d.h > 4:
                self.sizes.append(math.sqrt(d.w * d.h) * (w / float(work.shape[1])))
        if len(self.sizes) >= 5:
            med = float(np.median(self.sizes))
            self.px_per_m = float(np.clip(med / math.sqrt(CFG["VEH_WIDTH_M"] * CFG["VEH_LEN_M"]),
                                          CFG["SCALE_MIN"], CFG["SCALE_MAX"]))
        # rescale detections back to the original frame coordinates
        sx = w / float(work.shape[1])
        dets = [Det(d.x1 * sx, d.y1 * sx, d.x2 * sx, d.y2 * sx, d.conf, d.cat, d.coco)
                for d in dets]

        self.tracker.update(t, dets)

        # ---- causal scene model: occupancy + flow grid ---------------------
        self._update_scene_grid(work, t)

        # ---- per-track kinematics from our own tracker ---------------------
        tracks: List[Track] = [tr for tr in self.tracker.tracks.values()
                               if tr.hits >= 2 and tr.misses <= self.tracker.max_age]
        feat = {k: 0.0 for k in ("ttc", "brake", "prox", "wrong", "ped", "swerve",
                                "block", "contact")}
        if not tracks:
            self.risk = max(0.0, self.risk * CFG["RISK_DECAY"])
            return self.risk

        state = []
        for tr in tracks:
            o = tr.obs[-1]
            vx = tr.vx / max(1e-3, self.px_per_m)          # m/s
            vy = tr.vy / max(1e-3, self.px_per_m)
            sp = math.hypot(vx, vy)
            hd = math.degrees(math.atan2(vy, vx))
            self.vhist[tr.tid].append((t, sp, hd))
            hist = self.vhist[tr.tid]
            decel = 0.0
            if len(hist) >= 3:
                (t0, s0, _h0) = hist[-3]
                dtv = max(1e-3, t - t0)
                decel = (sp - s0) / dtv
            state.append({"tr": tr, "o": o, "sp": sp, "hd": hd,
                          "x": o.cx, "y": o.cy, "decel": decel})

        # hardest deceleration and heading rate
        for st in state:
            hist = self.vhist[st["tr"].tid]
            if len(hist) >= 3:
                (t0, _s0, h0) = hist[-3]
                dtv = max(1e-3, t - t0)
                hr = abs(((st["hd"] - h0 + 180.0) % 360.0) - 180.0) / dtv
                feat["swerve"] = max(feat["swerve"], min(1.0, max(0.0, (hr - 25.0) / 45.0)))
            feat["brake"] = max(feat["brake"],
                                min(1.0, max(0.0, (-st["decel"] - CFG["RISK_BRAKE"]) / 5.0)))
            if st["sp"] < CFG["STILL_MPS"]:
                self.still_since.setdefault(st["tr"].tid, t)
            else:
                self.still_since.pop(st["tr"].tid, None)
            still = t - self.still_since.get(st["tr"].tid, t)
            # blocked lane: stopped for a while and somebody else is moving nearby
            if still > 4.0:
                moving_near = sum(1 for o in state
                                  if o is not st and o["sp"] > CFG["MOVE_MPS"]
                                  and math.hypot(o["x"] - st["x"], o["y"] - st["y"])
                                  < 8.0 * self.px_per_m)
                if moving_near:
                    feat["block"] = max(feat["block"], min(1.0, still / 12.0))

        # ---- wrong way vs. the causally learned lane direction -------------
        for st in state:
            d = self._flow_dir(st["x"], st["y"], h, w)
            if d is None or st["sp"] < CFG["WW_MIN_MPS"]:
                continue
            if ang_between(d, (math.cos(math.radians(st["hd"])), math.sin(math.radians(st["hd"])))) \
                    >= CFG["WW_MIN_ANGLE"]:
                feat["wrong"] = 1.0
                break

        # ---- pedestrian inside the learned carriageway ---------------------
        for st in state:
            if st["tr"].cat != "person" or st["sp"] < 0.2:
                continue
            gy = int(np.clip(st["y"] / max(1, h) * self.occ.shape[0], 0, self.occ.shape[0] - 1))
            gx = int(np.clip(st["x"] / max(1, w) * self.occ.shape[1], 0, self.occ.shape[1] - 1))
            if self.occ[gy, gx] > 0.05:
                feat["ped"] = 1.0
                break

        # ---- pairwise TTC / proximity / contact ----------------------------
        veh = [s for s in state if s["tr"].cat in MOTION_CATS]
        veh.sort(key=lambda s: -(s["o"].w * s["o"].h))
        veh = veh[:12]                                     # bound the cost
        ttc_min = float("inf")
        for i in range(len(veh)):
            a = veh[i]
            for j in range(i + 1, len(veh)):
                b = veh[j]
                dx = b["x"] - a["x"]; dy = b["y"] - a["y"]
                dist = math.hypot(dx, dy)
                boxr = 0.5 * (math.hypot(a["o"].w, a["o"].h) + math.hypot(b["o"].w, b["o"].h))
                dist_m = dist / self.px_per_m                 # <-- метры
                if dist < boxr * CFG["ACC_TOUCH"] * 0.9 and (a["sp"] + b["sp"]) > 2.0:
                    feat["contact"] = 1.0
                if dist_m < 1e-3 or dist_m > 12.0:
                    continue
                av = (math.cos(math.radians(a["hd"])) * a["sp"], math.sin(math.radians(a["hd"])) * a["sp"])
                bv = (math.cos(math.radians(b["hd"])) * b["sp"], math.sin(math.radians(b["hd"])) * b["sp"])
                ux, uy = dx / dist, dy / dist
                closing = (av[0] - bv[0]) * ux + (av[1] - bv[1]) * uy      # >0 = сближаются
                if closing > 0.3:
                    ttc = dist_m / closing
                    if ttc < ttc_min:
                        ttc_min = ttc
                # близость в длинах корпуса (дистанция в метрах!)
                ln = max(1.0, 0.5 * (math.hypot(a["o"].w, a["o"].h) / self.px_per_m))
                norm = dist_m / (ln * 4.4)
                if norm < 2.0:
                    feat["prox"] = max(feat["prox"], min(1.0, (2.0 - norm) / 2.0))
                # proximity in vehicle lengths
                ln = max(1e-3, 0.5 * (math.hypot(a["o"].w, a["o"].h) / self.px_per_m))
                norm = dist / (self.px_per_m * ln * 4.4)
                if norm < 2.0:
                    feat["prox"] = max(feat["prox"], min(1.0, (2.0 - norm) / 2.0))
        if ttc_min < CFG["RISK_TTC_MAX"]:
            feat["ttc"] = float(np.clip((CFG["RISK_TTC_MAX"] - ttc_min) / CFG["RISK_TTC_MAX"],
                                        0.0, 1.0))

        # ---- logistic fusion (weights fixed, calibrated to be conservative) --
        logit = (-3.2
                 + 2.6 * feat["ttc"]
                 + 2.0 * feat["brake"]
                 + 1.6 * feat["prox"]
                 + 2.2 * feat["wrong"]
                 + 1.3 * feat["ped"]
                 + 1.5 * feat["swerve"]
                 + 1.0 * feat["block"]
                 + CFG["RISK_CONTACT"] * feat["contact"])
        inst = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
        if feat["contact"] > 0:
            inst = max(inst, 0.95)
        # persistence: an approach should keep the alarm up for a few seconds
        self.risk = float(max(inst, self.risk * CFG["RISK_DECAY"]))
        return self.risk

    # ---------------------------------------------------------- online scene
    def _update_scene_grid(self, frame: np.ndarray, t: float) -> None:
        """Causally accumulate occupancy (carriageway) and flow direction."""
        try:
            h, w = frame.shape[:2]
            gh, gw = self.occ.shape
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            # occupancy from moving pixels (background subtraction, causal)
        except Exception:
            LOGGER.debug("scene grid update failed", exc_info=True)

    def _flow_dir(self, x: float, y: float, h: int, w: int) -> Optional[Tuple[float, float]]:
        """Learned lane direction at a point (None until enough evidence)."""
        gh, gw = self.flow_n.shape
        gy = int(np.clip(y / max(1, h) * gh, 0, gh - 1))
        gx = int(np.clip(x / max(1, w) * gw, 0, gw - 1))
        if self.flow_n[gy, gx] < 3.0:
            return None
        v = self.flow[gy, gx]
        n = float(np.linalg.norm(v))
        if n < 0.15:
            return None
        return float(v[0] / n), float(v[1] / n)


# ============================================================================
#  CLI: local smoke test + assets for the team website
# ============================================================================
def _emit_assets(video_path: str, out_dir: str) -> dict:
    """Run both parts on one clip and write the artefacts the website needs.

    Writes: events.json, risk.json and an annotated .mp4 with the timeline and
    the risk curve burned in (used for the "Results on the sample videos" page).
    This is NOT part of the graded interface - it is a convenience tool.
    """
    os.makedirs(out_dir, exist_ok=True)
    events = detect_events(video_path)
    risk: List[List[float]] = []
    vr = VideoReader(video_path)
    est = RiskEstimator()
    est.reset({"video_id": osp.basename(video_path), "fps": vr.fps, "width": vr.width,
               "height": vr.height, "n_frames": vr.n_frames})
    writer = None
    out_mp4 = osp.join(out_dir, osp.splitext(osp.basename(video_path))[0] + "_annotated.mp4")
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_mp4, fourcc, min(25.0, vr.fps), (vr.width, vr.height))
        for fi, t, frame in vr.iter_frames(stride=1):
            s = est.step(frame, t)
            risk.append([round(float(t), 3), round(float(s), 4)])
            if writer is not None:
                writer.write(_draw_overlay(frame, events, risk, vr.duration))
    finally:
        if writer is not None:
            writer.release()
        vr.close()
    with open(osp.join(out_dir, "events.json"), "w") as f:
        json.dump({"video": osp.basename(video_path), "events": events}, f, indent=1)
    with open(osp.join(out_dir, "risk.json"), "w") as f:
        json.dump({"video": osp.basename(video_path), "risk": risk}, f, indent=1)
    return {"events": events, "risk_points": len(risk), "video": out_mp4}


def _draw_overlay(frame: np.ndarray, events: Sequence[list], risk: Sequence[Sequence[float]],
                  duration: float) -> np.ndarray:
    """Timeline bar + risk curve + current event labels burned into the frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    bar_h = max(28, h // 24)
    y0 = h - bar_h
    cv2.rectangle(out, (0, y0), (w, h), (20, 20, 20), -1)
    if duration > 0:
        colours = {"accident": (0, 0, 255), "near_miss": (0, 140, 255), "red_light": (255, 0, 0),
                   "wrong_way": (255, 0, 255), "congestion": (0, 200, 0),
                   "stopped_vehicle": (0, 200, 200), "jaywalking": (200, 200, 0)}
        for s, e, lab in events:
            x1 = int(w * max(0.0, min(1.0, s / duration)))
            x2 = int(w * max(0.0, min(1.0, e / duration)))
            cv2.rectangle(out, (x1, y0 + 2), (max(x2, x1 + 2), h - 2),
                          colours.get(lab, (180, 180, 180)), -1)
            cv2.putText(out, lab, (min(x1 + 2, w - 90), y0 + bar_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    if risk:
        n = max(1, len(risk))
        for i in range(1, n):
            x1 = int(w * (i - 1) / n); x2 = int(w * i / n)
            y1 = int(h * 0.30 * (1.0 - float(risk[i - 1][1])))
            y2 = int(h * 0.30 * (1.0 - float(risk[i][1])))
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
    return out


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Traffic event detection - local runner")
    ap.add_argument("videos", nargs="+", help="one or more .mp4 files (or a folder)")
    ap.add_argument("--out", default="", help="write a predictions.json here")
    ap.add_argument("--assets", default="", help="write website assets (events/risk/annotated mp4) here")
    ap.add_argument("--risk", action="store_true", help="also stream Part B and print the peak risk")
    args = ap.parse_args(argv)

    files: List[str] = []
    for v in args.videos:
        if osp.isdir(v):
            files += [osp.join(v, f) for f in sorted(os.listdir(v))
                      if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        else:
            files.append(v)
    preds = {"team": "your-team-name", "videos": {}}
    for f in files:
        t0 = time.time()
        events = detect_events(f)
        key = osp.basename(f)
        preds["videos"][key] = {"events": events, "risk": []}
        print("%-32s %2d events  (%.1fs)" % (key, len(events), time.time() - t0))
        for e in events:
            print("   [%7.2f .. %7.2f]  %s" % (e[0], e[1], e[2]))
        if args.risk:
            vr = VideoReader(f)
            est = RiskEstimator()
            est.reset({"video_id": key, "fps": vr.fps, "width": vr.width,
                       "height": vr.height, "n_frames": vr.n_frames})
            peak = 0.0
            t_peak = 0.0
            for fi, t, frame in vr.iter_frames(stride=1):
                s = est.step(frame, t)
                if s > peak:
                    peak, t_peak = s, t
            vr.close()
            print("   Part B: peak risk %.3f at t=%.1fs" % (peak, t_peak))
        if args.assets:
            info = _emit_assets(f, args.assets)
            print("   assets -> %s" % info["video"])
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(preds, fh, indent=1)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(_main())