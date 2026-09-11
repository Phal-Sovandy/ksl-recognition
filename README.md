# KSL Recognition

Khmer Sign Language (KSL) recognition from MediaPipe body/hand landmarks: a
mask-aware Graph Convolutional Network + Transformer classifier, trained with
a multilingual pretrain stage across several public sign-language datasets
before fine-tuning on KSL.

```
video → MediaPipe Holistic → (coords, visibility) landmarks
      → feature assembly (position + velocity + visibility)
      → mask-aware GCN (spatial) → Transformer encoder (temporal) → classifier
```

## Project layout

```
src/ksl_recognition/
├── data/          # dataset loading, feature assembly, augmentations
├── models/        # the GCN+Transformer model and RNN/LSTM/GRU baselines
├── extraction/    # per-dataset MediaPipe landmark extraction
├── training/      # single-language, multilingual-pretrain, baseline training loops
├── inference/     # batch (.npz) and real-time (webcam) inference
├── export/        # dataset repackaging for Kaggle / anonymized release
└── analysis/      # EDA and embedding visualization (t-SNE / UMAP)

notebooks/         # exploratory / interactive training notebook
scripts/           # thin shell wrappers around the CLIs above
data/              # datasets (gitignored — see Data below)
checkpoints/       # trained model weights (gitignored)
plots/             # generated figures (gitignored)
```

## Setup

Requires Python ≥ 3.9.

```bash
git clone <this-repo>
cd ksl-recognition
pip install -e ".[dev]"
```

`pip install -e .` alone pulls the core runtime dependencies (torch, numpy,
opencv, mediapipe, pandas, pillow, tqdm); `[dev]` also adds the optional
extras used by the real-time demo, dataset export, and EDA scripts
(`psutil`, `openpyxl`, `matplotlib`, `scikit-learn`, `umap-learn`, `scipy`).
See `pyproject.toml` for the individual extras if you only need one.

Every command below is a Python module and expects to run from the repo
root, e.g. `python -m ksl_recognition.training.train --help`.

## Data

Datasets are not tracked in git (100+ GB of video/landmarks) and are
expected on disk as:

```
data/raw_data/<dataset>/...            # source videos
data/processed_landmarks/<dataset>/<class>/*.npz   # coords (T,51,2) + vis (T,51)
```

Landmark extraction is per-dataset (layouts vary), one script per source
under `src/ksl_recognition/extraction/`:

| Dataset | Script |
|---|---|
| KSL (core) | `extract_landmarks.py` |
| KSL — new words | `extract_new_ksl_landmarks.py` |
| Slovo (Russian SL) | `extract_slovo_landmarks.py` |
| WLASL (American SL) | `extract_wlasl_landmarks.py` |
| Arabic SL (2) | `extract_arabic2_landmarks.py` |
| Argentina SL | `extract_argentina_landmarks.py` |
| Indian SL (2) | `extract_indian2_landmarks.py` |
| Sri Lanka SL | `extract_srilanka_landmarks.py` |
| Vietnam SL | `extract_vietnam_landmarks.py` |

```bash
python -m ksl_recognition.extraction.extract_landmarks \
    --input data/raw_data/ksl_dataset --output data/processed_landmarks/ksl --workers 8
# or:
scripts/extract_ksl.sh
```

`ksl_recognition.extraction.count_files` gives a quick raw-vs-extracted
progress check across every dataset.

## Training

Two-stage recipe: pretrain a shared encoder across every available
language, then fine-tune it on KSL.

```bash
# Stage 1 — joint multilingual pretrain (or scripts/pretrain_rsl.sh for a
# single-dataset Slovo pretrain via the same train.py entry point)
python -m ksl_recognition.training.train_multilingual --epochs 120 --steps-per-lang 40

# Stage 2 — fine-tune on KSL from the Stage-1 encoder
python -m ksl_recognition.training.train \
    --data data/processed_landmarks/ksl \
    --pretrained checkpoints/multilingual_pretrain.pt \
    --enc-lr 1e-4 --head-lr 1e-3
```

Classic recurrent baselines (RNN/LSTM/BiLSTM/GRU/BiGRU), for an ablation
against the GCN+Transformer model:

```bash
python -m ksl_recognition.training.train_baselines --models lstm bilstm --epochs 100
```

`notebooks/train.ipynb` covers the same pipeline interactively.

## Inference

```bash
# One .npz clip or a directory of them
python -m ksl_recognition.inference.infer --model checkpoints/ksl_best.pt --input data/processed_landmarks/ksl/<gloss>/<clip>.npz

# Live webcam demo
python -m ksl_recognition.inference.infer_realtime --model checkpoints/ksl_best.pt
```

## Model

`ksl_recognition.models.sign_model.MultiHeadSignModel` — 51 upper-body
landmarks (9 pose + 21 per hand) laid out as an anatomical graph; a
mask-aware GCN aggregates spatial structure per frame (absent landmarks
contribute nothing rather than a fake origin point), then a Transformer
encoder with a CLS token models motion over time. One shared encoder can
carry several per-language classifier heads (`.add_head()`), which is what
the multilingual pretrain stage relies on. Full design rationale is in the
module docstring.

## License

MIT — see [LICENSE](LICENSE).
