"""
infer.py
--------
Run inference on one or more .npz landmark files.

Run as a module (requires `pip install -e .` from the repo root, see README).

Single file:
    python -m ksl_recognition.inference.infer --model checkpoints/ksl_best.pt --input data/processed_landmarks/ksl/កង់/កង់_25.npz

Directory (all .npz files inside, recursively):
    python -m ksl_recognition.inference.infer --model checkpoints/ksl_best.pt --input data/test_clips/

Output shows the predicted class label, confidence, and top-k alternatives.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from ksl_recognition.data.features import assemble_features, frame_usable
from ksl_recognition.models.sign_model import MultiHeadSignModel

HEAD = "ksl"
MAX_LEN = 128


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    model = MultiHeadSignModel(
        in_ch=5, head_classes={HEAD: num_classes},
        d_model=128, gcn_hidden=(64, 128), nhead=4, num_layers=4,
        dim_ff=256, dropout=0.2, max_len=MAX_LEN,
    ).to(device)

    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.TransformerEncoder):
            m.enable_nested_tensor = False
            m.use_nested_tensor = False

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, idx_to_class


def load_npz(path, device):
    d = np.load(path)
    coords = torch.from_numpy(d["coords"]).float()   # (T, 75, 2)
    vis = torch.from_numpy(d["vis"]).float()         # (T, 75)

    T = coords.shape[0]
    if T > MAX_LEN:
        idx = torch.linspace(0, T - 1, MAX_LEN).round().long()
        coords, vis = coords[idx], vis[idx]

    feats = assemble_features(coords, vis, use_velocity=True)   # (T, 75, 5)
    vmask = (vis > 0).float()                                   # (T, 75)
    fmask = frame_usable(vis)                                   # (T,)

    feats = feats.unsqueeze(0).to(device)    # (1, T, 75, 5)
    vmask = vmask.unsqueeze(0).to(device)    # (1, T, 75)
    fmask = fmask.unsqueeze(0).to(device)    # (1, T)
    return feats, vmask, fmask


@torch.no_grad()
def infer_one(model, npz_path, idx_to_class, device, topk=3):
    feats, vmask, fmask = load_npz(npz_path, device)
    logits = model(feats, vmask, fmask, head=HEAD)          # (1, C)
    probs = F.softmax(logits, dim=-1).squeeze(0)            # (C,)

    k = min(topk, len(idx_to_class))
    top_probs, top_idxs = probs.topk(k)

    results = [(idx_to_class[i.item()], p.item()) for i, p in zip(top_idxs, top_probs)]
    return results


def collect_npz(input_path):
    p = Path(input_path)
    if p.is_file():
        if p.suffix != ".npz":
            raise ValueError(f"Expected a .npz file, got: {p}")
        return [p]
    if p.is_dir():
        files = sorted(p.rglob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found under {p}")
        return files
    raise FileNotFoundError(f"Path not found: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to ksl_best.pt checkpoint")
    ap.add_argument("--input", required=True, help="Path to a .npz file or a directory of .npz files")
    ap.add_argument("--topk", type=int, default=3, help="Number of top predictions to show")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")

    model, idx_to_class = load_model(args.model, device)
    print(f"loaded model  |  {len(idx_to_class)} classes\n")

    npz_files = collect_npz(args.input)
    print(f"running inference on {len(npz_files)} file(s)\n")

    for npz_path in npz_files:
        results = infer_one(model, npz_path, idx_to_class, device, topk=args.topk)
        top_label, top_conf = results[0]
        print(f"{npz_path.name}")
        print(f"  prediction : {top_label}  ({top_conf * 100:.1f}%)")
        if len(results) > 1:
            alts = "  |  ".join(f"{lbl} {p * 100:.1f}%" for lbl, p in results[1:])
            print(f"  top-{args.topk}      : {alts}")
        print()


if __name__ == "__main__":
    main()
