"""
infer_realtime.py
-----------------
Real-time KSL sign recognition from webcam.

Run as a module (requires `pip install -e .` from the repo root, see README).

Two modes:

  Rolling (default) -- inference runs automatically every --interval frames
  on the last MAX_LEN frames in the buffer:
      python -m ksl_recognition.inference.infer_realtime --model checkpoints/ksl_best.pt

  Manual -- press SPACE to start recording, SPACE again to stop and predict:
      python -m ksl_recognition.inference.infer_realtime --model checkpoints/ksl_best.pt --manual

Keys (both modes):
  SPACE  start/stop recording (manual) | force infer now (rolling)
  R      reset frame buffer
  Q      quit
"""
import argparse
import collections
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from ksl_recognition.data.features import assemble_features, frame_usable
from ksl_recognition.models.sign_model import MultiHeadSignModel

# inference/ -> ksl_recognition/ -> assets/, shipped as package data
FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "NotoSansKhmer_Condensed-Black.ttf"
# Fallback for scripts the Khmer font can't render (e.g. Cyrillic labels from
# the RSL pretrain checkpoint) -- macOS ships this system-wide.
FONT_PATH_FALLBACK = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")

MAX_LEN = 128
MIN_FRAMES = 10          # need at least this many frames before inferring
# Raw MediaPipe pose indices kept, in order -> compact node 0..8 (must match
# extraction.utils.UPPER_BODY_POSE_IDX): nose, shoulders, elbows, wrists, hips.
UPPER_BODY_POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]
POSE = len(UPPER_BODY_POSE_IDX)
HAND = 21
NUM_NODES = POSE + 2 * HAND
LH_OFFSET = POSE
RH_OFFSET = POSE + HAND

# Colours (RGB for PIL)
C_GREEN  = (80,  220,  0)
C_YELLOW = (220, 200,  0)
C_RED    = (220,  60,  0)
C_WHITE  = (240, 240, 240)
C_GRAY   = (180, 180, 180)
C_LGRAY  = (200, 200, 200)
C_DARK   = (30,   30,  30)


def load_fonts(frame_h=720, font_path=FONT_PATH):
    s = frame_h / 480
    sizes = {
        "large":  max(22, int(34 * s)),
        "small":  max(12, int(17 * s)),
        "tiny":   max(10, int(13 * s)),
    }
    out = {}
    for k, v in sizes.items():
        try:
            # RAQM uses HarfBuzz for proper Khmer/complex-script shaping
            out[k] = ImageFont.truetype(
                str(font_path), v, layout_engine=ImageFont.Layout.RAQM
            )
        except Exception:
            try:
                out[k] = ImageFont.truetype(str(font_path), v)
            except Exception:
                out[k] = ImageFont.load_default()
    return out


# --------------------------------------------------------------------------- #
# Device / model
# --------------------------------------------------------------------------- #
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path, device, head=None):
    """Load a checkpoint. If `head` is None, auto-detect the single head name
    stored in the checkpoint's state dict (e.g. "ksl" or "rsl")."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    if head is None:
        head_names = sorted({k.split(".")[1] for k in ckpt["model"] if k.startswith("heads.")})
        if len(head_names) != 1:
            raise ValueError(
                f"Checkpoint has {len(head_names)} heads {head_names}; pass --head explicitly."
            )
        head = head_names[0]

    # Read arch from checkpoint when available; fall back to current training defaults.
    arch = ckpt.get("arch", dict(
        in_ch=5, d_model=256, gcn_hidden=(128, 256),
        nhead=8, num_layers=6, dim_ff=512, dropout=0.1, max_len=MAX_LEN,
    ))
    arch.setdefault("max_len", MAX_LEN)

    model = MultiHeadSignModel(
        head_classes={head: num_classes}, **arch,
    ).to(device)

    for m in model.modules():
        if isinstance(m, nn.TransformerEncoder):
            m.enable_nested_tensor = False
            m.use_nested_tensor = False

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, idx_to_class, head


# --------------------------------------------------------------------------- #
# Landmark extraction (matches extract_landmarks.py exactly)
# --------------------------------------------------------------------------- #
def extract_frame_landmarks(result, w, h, min_visibility=0.5):
    """MediaPipe Holistic result -> (coords (51,2), vis (51,)) float32."""
    c = np.zeros((NUM_NODES, 2), dtype=np.float32)
    v = np.zeros((NUM_NODES,),  dtype=np.float32)

    if result.pose_landmarks:
        raw = result.pose_landmarks.landmark
        for i, raw_idx in enumerate(UPPER_BODY_POSE_IDX):
            lm = raw[raw_idx]
            if lm.visibility >= min_visibility:
                c[i, 0] = lm.x * w
                c[i, 1] = lm.y * h
                v[i] = lm.visibility

    if result.left_hand_landmarks:
        for i, lm in enumerate(result.left_hand_landmarks.landmark):
            c[LH_OFFSET + i, 0] = lm.x * w
            c[LH_OFFSET + i, 1] = lm.y * h
        v[LH_OFFSET:LH_OFFSET + HAND] = 1.0

    if result.right_hand_landmarks:
        for i, lm in enumerate(result.right_hand_landmarks.landmark):
            c[RH_OFFSET + i, 0] = lm.x * w
            c[RH_OFFSET + i, 1] = lm.y * h
        v[RH_OFFSET:RH_OFFSET + HAND] = 1.0

    return c, v


# --------------------------------------------------------------------------- #
# Inference on a list of (coords, vis) frames
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_inference(frames, model, idx_to_class, device, head, topk=3):
    """frames: list of (coords (51,2), vis (51,)) numpy arrays."""
    coords = torch.from_numpy(np.stack([f[0] for f in frames])).float()  # (T,51,2)
    vis    = torch.from_numpy(np.stack([f[1] for f in frames])).float()  # (T,51)

    T = coords.shape[0]
    if T > MAX_LEN:
        idx = torch.linspace(0, T - 1, MAX_LEN).round().long()
        coords, vis = coords[idx], vis[idx]

    feats = assemble_features(coords, vis, use_velocity=True)   # (T,51,5)
    vmask = (vis > 0).float()                                   # (T,51)
    fmask = frame_usable(vis)                                   # (T,)

    feats = feats.unsqueeze(0).to(device)
    vmask = vmask.unsqueeze(0).to(device)
    fmask = fmask.unsqueeze(0).to(device)

    logits = model(feats, vmask, fmask, head=head)
    probs  = F.softmax(logits, dim=-1).squeeze(0)

    k = min(topk, len(idx_to_class))
    top_probs, top_idxs = probs.topk(k)
    return [(idx_to_class[i.item()], p.item()) for i, p in zip(top_idxs, top_probs)]


# --------------------------------------------------------------------------- #
# Async inference wrapper (keeps the display thread from stalling)
# --------------------------------------------------------------------------- #
class AsyncInferencer:
    def __init__(self, model, idx_to_class, device, head):
        self.model = model
        self.idx_to_class = idx_to_class
        self.device = device
        self.head = head
        self.result = None          # [(label, conf), ...]
        self.busy = False
        self._lock = threading.Lock()

    def submit(self, frames):
        """Snapshot current frames and kick off inference in a background thread."""
        if self.busy or len(frames) < MIN_FRAMES:
            return
        snapshot = list(frames)
        self.busy = True
        t = threading.Thread(target=self._worker, args=(snapshot,), daemon=True)
        t.start()

    def _worker(self, frames):
        try:
            result = run_inference(
                frames, self.model, self.idx_to_class, self.device, self.head
            )
            with self._lock:
                self.result = result
        finally:
            self.busy = False

    def get_result(self):
        with self._lock:
            return self.result


# --------------------------------------------------------------------------- #
# Resource monitor (background thread, samples every second)
# --------------------------------------------------------------------------- #
class ResourceMonitor:
    def __init__(self, device):
        self._device = device
        self.cpu = 0.0
        self.ram = 0.0
        self._gpu_mb = -1
        self._lock = threading.Lock()
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        while True:
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory().percent
            gpu_mb = -1
            try:
                if self._device.type == "cuda":
                    gpu_mb = torch.cuda.memory_allocated(self._device) // (1024 * 1024)
                elif self._device.type == "mps":
                    gpu_mb = torch.mps.current_allocated_memory() // (1024 * 1024)
            except Exception:
                pass
            with self._lock:
                self.cpu, self.ram, self._gpu_mb = cpu, ram, gpu_mb

    def text(self):
        with self._lock:
            s = f"CPU {self.cpu:.0f}%  RAM {self.ram:.0f}%"
            if self._gpu_mb >= 0:
                s += f"  GPU {self._gpu_mb}MB"
            return s


# --------------------------------------------------------------------------- #
# Drawing helpers (PIL-based for Khmer font support)
# --------------------------------------------------------------------------- #

def _label(draw, text, xy, font, fg=C_WHITE, bg=C_DARK):
    x, y = xy
    bb = draw.textbbox((x, y), text, font=font)
    draw.rectangle([(bb[0] - 4, bb[1] - 3), (bb[2] + 4, bb[3] + 3)], fill=bg)
    draw.text((x, y), text, font=font, fill=fg)


def draw_overlay(frame, inferencer, recording, manual_mode, buf_len, fps_str, fonts,
                 res_mon=None, hands_present=True):
    h, w   = frame.shape[:2]
    result = inferencer.get_result()

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d   = ImageDraw.Draw(pil)

    # ---- top-left: prediction ----
    if result:
        label, conf = result[0]
        color = C_GREEN if conf >= 0.6 else C_YELLOW if conf >= 0.35 else C_RED
        main_txt = f"{label}  {conf * 100:.1f}%"
        lbb = d.textbbox((10, 8), main_txt, font=fonts["large"])
        _label(d, main_txt, (10, 8), fonts["large"], fg=color)
        if len(result) > 1:
            alt = "  |  ".join(f"{l} {p*100:.0f}%" for l, p in result[1:])
            _label(d, alt, (10, lbb[3] + 6), fonts["small"], fg=C_GRAY)
    else:
        _label(d, "waiting...", (10, 8), fonts["large"], fg=C_GRAY)

    # ---- top-right: buffer / fps ----
    stats = f"buf: {buf_len}/{MAX_LEN}  {fps_str}"
    sb = d.textbbox((0, 0), stats, font=fonts["tiny"])
    line_h = sb[3] - sb[1]
    _label(d, stats, (w - (sb[2] - sb[0]) - 14, 8), fonts["tiny"])

    # ---- resource stats (below fps line) ----
    if res_mon is not None:
        res_txt = res_mon.text()
        rb = d.textbbox((0, 0), res_txt, font=fonts["tiny"])
        _label(d, res_txt, (w - (rb[2] - rb[0]) - 14, 8 + line_h + 4),
               fonts["tiny"], fg=C_GRAY)

    # ---- centered banner: no hand visible (kept off the prediction label) ----
    if not hands_present:
        warn = "⚠  No hands detected"
        wb = d.textbbox((0, 0), warn, font=fonts["small"])
        ww, wh = wb[2] - wb[0], wb[3] - wb[1]
        wx, wy = (w - ww) // 2, int(h * 0.45) - wh // 2
        _label(d, warn, (wx, wy), fonts["small"], fg=C_RED)

    # ---- recording indicator ----
    if recording:
        d.ellipse([(w - 26, 42), (w - 10, 58)], fill=C_RED)
        _label(d, "REC", (w - 58, 44), fonts["tiny"], fg=C_RED)

    # ---- bottom: key hints ----
    if manual_mode:
        hint = "SPACE: start/stop  |  R: reset  |  Q: quit"
    else:
        hint = "SPACE: infer now  |  R: reset  |  Q: quit"
    hb = d.textbbox((0, 0), hint, font=fonts["tiny"])
    _label(d, hint, (10, h - (hb[3] - hb[1]) - 10), fonts["tiny"], fg=C_LGRAY)

    # ---- inferring indicator ----
    if inferencer.busy:
        _label(d, "inferring...", (10, h - (hb[3] - hb[1]) - 34), fonts["tiny"],
               fg=C_YELLOW)

    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True,
                    help="path to a checkpoint, e.g. checkpoints/ksl_best.pt or "
                         "checkpoints/rsl_pretrain.pt (to sanity-check the RSL pretrain model)")
    ap.add_argument("--head",     default=None,
                    help="head name inside the checkpoint (default: auto-detect, e.g. "
                         "'ksl' or 'rsl')")
    ap.add_argument("--font",     default=None,
                    help="font file to render labels with (default: auto -- Khmer font for "
                         "the 'ksl' head, a Unicode fallback font otherwise)")
    ap.add_argument("--camera",   type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--topk",     type=int, default=3)
    ap.add_argument("--interval", type=int, default=15,
                    help="(rolling) run inference every N frames (default 15)")
    ap.add_argument("--manual",   action="store_true",
                    help="manual mode: press SPACE to start/stop a recording segment")
    ap.add_argument("--width",    type=int, default=720)
    ap.add_argument("--height",   type=int, default=1280)
    ap.add_argument("--mp-skip",  type=int, default=2,
                    help="run MediaPipe every N frames (default 2, higher = faster)")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    model, idx_to_class, head = load_model(args.model, device, head=args.head)
    print(f"loaded model  |  head='{head}'  |  {len(idx_to_class)} classes")
    print(f"mode: {'manual' if args.manual else 'rolling'}\n")

    font_path = args.font
    if font_path is None:
        font_path = FONT_PATH if head == "ksl" else FONT_PATH_FALLBACK
    fonts = load_fonts(args.height, font_path)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    mp_holistic = mp.solutions.holistic
    mp_draw     = mp.solutions.drawing_utils
    mp_styles   = mp.solutions.drawing_styles

    inferencer = AsyncInferencer(model, idx_to_class, device, head)
    res_mon    = ResourceMonitor(device)
    buffer     = collections.deque(maxlen=MAX_LEN)
    recording  = False         # manual mode state
    frame_idx  = 0
    fps_str    = ""
    t_fps      = time.time()
    last_mp    = None          # last MediaPipe result (reused on skipped frames)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed"); break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # Run MediaPipe only every --mp-skip frames; reuse last result otherwise
            if frame_idx % args.mp_skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                last_mp = holistic.process(rgb)
                rgb.flags.writeable = True

            mp_r = last_mp

            # Extract landmarks for this frame
            if mp_r is not None:
                c, v = extract_frame_landmarks(mp_r, w, h)
            else:
                c = np.zeros((NUM_NODES, 2), np.float32)
                v = np.zeros(NUM_NODES, np.float32)

            # ---- MediaPipe landmark drawing ----
            if mp_r and mp_r.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame, mp_r.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
            if mp_r and mp_r.left_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, mp_r.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec((121, 22, 76),  2, 2),
                    mp_draw.DrawingSpec((121, 44, 250), 2, 1))
            if mp_r and mp_r.right_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, mp_r.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec((245, 117, 66), 2, 2),
                    mp_draw.DrawingSpec((245, 66, 230), 2, 1))

            # ---- Hand-presence check: skip inference (and its model forward
            # pass) whenever neither hand is visible -- no point running the
            # network on an empty gesture, so this saves real compute/perf.
            hands_present = bool(
                mp_r and (mp_r.left_hand_landmarks or mp_r.right_hand_landmarks)
            )
            if not hands_present and frame_idx % 30 == 0:
                print("no hands detected -- skipping inference")

            # ---- Buffer + inference logic ----
            if args.manual:
                if recording:
                    buffer.append((c, v))
            else:
                buffer.append((c, v))
                if frame_idx % args.interval == 0 and hands_present:
                    inferencer.submit(buffer)

            # ---- FPS counter ----
            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.time() - t_fps
                fps_str = f"{30 / elapsed:.0f} fps"
                t_fps = time.time()

            # ---- HUD ----
            draw_overlay(frame, inferencer, recording, args.manual,
                         len(buffer), fps_str, fonts, res_mon,
                         hands_present=hands_present)
            cv2.imshow("KSL Real-time Inference", frame)

            # ---- Key handling ----
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                buffer.clear()
                inferencer.result = None
                recording = False
                print("buffer reset")
            elif key == ord(' ') or key == 32:
                if args.manual:
                    if not recording:
                        buffer.clear()
                        recording = True
                        print("recording started...")
                    else:
                        recording = False
                        if hands_present:
                            print(f"recording stopped ({len(buffer)} frames) — inferring...")
                            inferencer.submit(buffer)
                        else:
                            print("recording stopped — no hands detected, skipping inference")
                else:
                    # Rolling mode: force an immediate inference
                    if hands_present:
                        inferencer.submit(buffer)
                    else:
                        print("no hands detected — skipping inference")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
