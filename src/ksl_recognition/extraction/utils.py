"""
utils.py
------------------
Shared core for every extract_*_landmarks.py script in this project.

Every one of those scripts does the exact same three things: (1) run
MediaPipe Holistic over a video and turn it into (coords, vis) arrays, (2)
farm that out across a worker pool with a progress bar, (3) print a summary.
The ONLY thing that actually differs between datasets is step 0 -- figuring
out which (video_path, output_npz_path) pairs exist, since every dataset on
disk organizes its videos differently (flat folders, nested categories,
filename-encoded classes, CSV label files, ...).

So each extract_*_landmarks.py script only needs to implement its own
`discover_pairs()` (dataset-layout-specific) and then hand the resulting
list to `run_extraction()` here. That keeps the actual extraction logic --
node layout, visibility handling, worker isolation -- defined exactly once,
so a MediaPipe tuning change or a landmark-format fix doesn't need to be
copy-pasted across N files.

Output format written by every script that uses this module:
    coords : float32  (T, 51, 2)   pixel-space x, y
    vis    : float32  (T, 51)      per-node visibility (1 = trust this node)

Node layout (MUST match features.py / sign_model.py):
    [ 0.. 8]  upper-body pose (9)   [ 9..29]  left hand (21)   [30..50]  right hand (21)

Pose is trimmed to the 9 MediaPipe landmarks that matter for signing --
nose, shoulders, elbows, wrists, hips (see UPPER_BODY_POSE_IDX) -- legs and
feet (knees, ankles, heels, foot index) are dropped at extraction time since
sign language carries no information below the hips and the model's
anatomical adjacency never wired them in anyway.
"""
import os
import re
import multiprocessing as mp_proc
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

try:
    from tqdm import tqdm
except ImportError:                       # graceful fallback if tqdm isn't installed
    def tqdm(it=None, **kw):
        print("(install tqdm for progress bars:  pip install tqdm)")
        return it if it is not None else []

# ----- layout constants (keep in lockstep with features.py) -------------------
# Raw MediaPipe pose indices kept, in order -> compact node 0..8:
#   0 nose, 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips.
# Everything below the hips (knees, ankles, heels, foot index) carries no
# signing information and is dropped here rather than stored and ignored.
UPPER_BODY_POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]
POSE = len(UPPER_BODY_POSE_IDX)           # 9
HAND = 21
NUM_NODES = POSE + 2 * HAND               # 51
LH_OFFSET = POSE                          # 9
RH_OFFSET = POSE + HAND                   # 30
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

_NUM_PREFIX_RE = re.compile(r"^\d+\.\s*")


# --------------------------------------------------------------------------- #
# Name sanitizing -- shared by every dataset that derives a class name from
# free text (CSV labels, user-supplied gloss maps) rather than an existing
# filesystem folder name (folder names can never contain '/', free text can).
# --------------------------------------------------------------------------- #
def sanitize_class_name(label: str) -> str:
    """Replace path-separator characters before a label is ever used as a
    directory component. A literal '/' in a gloss (e.g. a date like "8/3", or
    a synonym pair like "một ít/ một chút") would otherwise be silently
    treated as a path separator, splitting one class into nested folders
    instead of erroring."""
    return label.replace("/", "-").replace(os.sep, "-")


def strip_numeric_prefix(name: str) -> str:
    """'4. sad' -> 'sad'. For datasets whose leaf folders carry a download-
    ordering prefix that isn't actually part of the gloss."""
    return _NUM_PREFIX_RE.sub("", name).strip()


def unique_relative_stem(video_path, input_dir) -> str:
    """Full relative path (minus extension), joined with '__'. For datasets
    where generic per-folder filenames (MVI_0001.mov, Class_001.mov, ...)
    would collide once every video is regrouped under a class name taken
    from a shallower folder level than its immediate parent."""
    rel = Path(video_path).relative_to(input_dir).with_suffix("")
    return "__".join(rel.parts)


# --------------------------------------------------------------------------- #
# MediaPipe extraction core
# --------------------------------------------------------------------------- #
def _fill_pose_frame(c, v, landmarks, w, h, min_visibility):
    raw = landmarks.landmark
    for i, raw_idx in enumerate(UPPER_BODY_POSE_IDX):
        lm = raw[raw_idx]
        if lm.visibility < min_visibility:
            continue                       # leave coords 0, vis 0 -> masked node
        c[i, 0] = lm.x * w
        c[i, 1] = lm.y * h
        v[i] = lm.visibility


def _fill_hand_frame(c, v, landmarks, offset, w, h):
    for i, lm in enumerate(landmarks.landmark):
        c[offset + i, 0] = lm.x * w
        c[offset + i, 1] = lm.y * h
    v[offset:offset + HAND] = 1.0          # hand present -> binary presence flag


def _process_frame(frame_bgr, holistic, min_visibility):
    """One BGR frame array (from cv2.VideoCapture.read() or cv2.imread()) ->
    (c (51,2), v (51,)) float32 arrays. Shared by both extract_clip() (video
    file source) and extract_clip_from_frames() (pre-extracted image source)
    so the actual MediaPipe/node-layout logic is defined exactly once."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = holistic.process(rgb)

    c = np.zeros((NUM_NODES, 2), dtype=np.float32)
    v = np.zeros((NUM_NODES,), dtype=np.float32)
    if res.pose_landmarks:
        _fill_pose_frame(c, v, res.pose_landmarks, w, h, min_visibility)
    if res.left_hand_landmarks:
        _fill_hand_frame(c, v, res.left_hand_landmarks, LH_OFFSET, w, h)
    if res.right_hand_landmarks:
        _fill_hand_frame(c, v, res.right_hand_landmarks, RH_OFFSET, w, h)
    return c, v


def extract_clip(video_path, holistic, min_visibility=0.5, max_frames=None):
    """One video -> (coords (T,51,2), vis (T,51)) float32, or (None, None) if empty."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open {video_path}")
    coords_frames, vis_frames = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        c, v = _process_frame(frame, holistic, min_visibility)
        coords_frames.append(c)
        vis_frames.append(v)
        if max_frames and len(coords_frames) >= max_frames:
            break
    cap.release()
    if not coords_frames:
        return None, None
    return np.stack(coords_frames), np.stack(vis_frames)


def extract_clip_from_frames(frame_paths, holistic, min_visibility=0.5, max_frames=None):
    """Same as extract_clip() but for datasets shipped as a sorted sequence of
    per-frame image files instead of a single video file (e.g. KArSL-style
    pre-extracted RGB frame folders -- Arabic SL (2)). frame_paths must
    already be in temporal order; MediaPipe's tracker is fed one array at a
    time regardless of source, so this is otherwise identical to extract_clip().
    Unreadable/corrupt individual frames are skipped rather than failing the
    whole clip."""
    coords_frames, vis_frames = [], []
    for p in frame_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        c, v = _process_frame(frame, holistic, min_visibility)
        coords_frames.append(c)
        vis_frames.append(v)
        if max_frames and len(coords_frames) >= max_frames:
            break
    if not coords_frames:
        return None, None
    return np.stack(coords_frames), np.stack(vis_frames)


def _worker_init():
    """Pool initializer, runs once per worker process (not per clip).

    OpenCV spawns its own internal thread pool (size = all visible cores) the
    first time it's used. That's fine for a single process, but under
    multiprocessing.Pool every worker does it independently -- with W workers
    on a C-core machine you end up with W*C threads fighting over C cores,
    which is the classic "more workers = somehow slower" trap. Pinning each
    worker to a single OpenCV thread makes the process pool the only source
    of parallelism, which is what we actually want.

    Re: GPU -- mediapipe's Python Holistic API (mp.solutions.holistic, used
    below) has no GPU delegate on macOS; Google only ships that for
    Linux/Android in Python. There is no flag here that makes this run on
    the GPU -- CPU-side worker throughput is the only lever available."""
    cv2.setNumThreads(1)


def process_one(pair, model_complexity, min_visibility, max_frames, overwrite):
    """Worker: extract landmarks for one clip. pair: (video_path, out_path).
    One fresh Holistic instance per video -- correct tracking, no cross-clip leak."""
    video_path, out_path = pair
    if out_path.exists() and not overwrite:
        return ("skip", video_path.name)
    try:
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,       # track within a single clip
            model_complexity=model_complexity,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            coords, vis = extract_clip(video_path, holistic, min_visibility, max_frames)
    except Exception as e:                 # noqa: BLE001 - report, don't kill the pool
        return ("error", f"{video_path.name}: {e}")
    if coords is None:
        return ("empty", video_path.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, coords=coords, vis=vis)
    return ("ok", video_path.name)


def process_one_from_frames(pair, model_complexity, min_visibility, max_frames, overwrite):
    """Worker variant of process_one() for frame-sequence clips.
    pair: (frame_paths: list[Path], out_path). Same fresh-Holistic-per-clip,
    skip/error/empty/ok contract as process_one()."""
    frame_paths, out_path = pair
    if out_path.exists() and not overwrite:
        return ("skip", out_path.stem)
    try:
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            coords, vis = extract_clip_from_frames(frame_paths, holistic, min_visibility, max_frames)
    except Exception as e:                 # noqa: BLE001 - report, don't kill the pool
        return ("error", f"{out_path.stem}: {e}")
    if coords is None:
        return ("empty", out_path.stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, coords=coords, vis=vis)
    return ("ok", out_path.stem)


# --------------------------------------------------------------------------- #
# Shared pool-driving loop
# --------------------------------------------------------------------------- #
def run_extraction(pairs, *, desc, workers=4, model_complexity=1, min_visibility=0.5,
                   max_frames=None, overwrite=False, worker_fn=None):
    """
    Drive the worker pool over a prebuilt list of (clip_source, out_path) pairs,
    where clip_source is whatever `worker_fn` expects -- a video Path for the
    default process_one(), or a list of frame-image Paths for
    process_one_from_frames(). Every extract_*_landmarks.py script calls this
    after its own dataset-specific discover_pairs() -- this part (pool,
    progress bar, skip-existing, summary) never changes per dataset, only the
    discovery step and (occasionally) which worker_fn does the actual
    per-clip extraction.
    """
    if not pairs:
        print("No videos to process -- check your discovery step / input path.")
        return {"ok": 0, "empty": 0, "error": 0, "skip": 0}

    if not overwrite:
        todo = [p for p in pairs if not p[1].exists()]
        skipped = len(pairs) - len(todo)
    else:
        todo = pairs
        skipped = 0

    classes = sorted({out_path.parent.name for _, out_path in pairs})
    print(f"{len(pairs)} clips across {len(classes)} classes | "
          f"{len(todo)} to process | {skipped} already done | {workers} worker(s)\n")

    worker_fn = partial(worker_fn or process_one, model_complexity=model_complexity,
                        min_visibility=min_visibility, max_frames=max_frames,
                        overwrite=overwrite)

    totals = {"ok": 0, "empty": 0, "error": 0, "skip": skipped}
    errors = []

    pool = mp_proc.Pool(workers, initializer=_worker_init) if workers > 1 else None
    try:
        bar = tqdm(total=len(todo), unit="clip", desc=desc)
        results = (pool.imap_unordered(worker_fn, todo, chunksize=1)
                   if pool else (worker_fn(t) for t in todo))
        for status, info in results:
            totals[status] = totals.get(status, 0) + 1
            if status == "error":
                errors.append(info)
            bar.set_postfix_str(
                f"ok={totals['ok']} empty={totals['empty']} err={totals['error']}",
                refresh=False)
            bar.update(1)
        bar.close()
    finally:
        if pool:
            pool.close()
            pool.join()

    print(f"\nDone. ok={totals['ok']}  empty={totals['empty']}  "
          f"skipped={totals['skip']}  errors={totals['error']}")
    for e in errors[:10]:
        print("  ERROR:", e)
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more errors")
    return totals


def add_common_args(ap):
    """Shared CLI flags every extract_*_landmarks.py script exposes."""
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="parallel processes. Each worker is pinned to a single "
                         "OpenCV thread (see _worker_init in extraction/utils.py) so "
                         "this can safely use nearly all cores instead of half; "
                         "default leaves 2 cores free for the main process/OS. "
                         "There's no GPU path here -- mediapipe's Python Holistic "
                         "API has no GPU delegate on macOS, only Linux/Android.")
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--min-visibility", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    return ap
