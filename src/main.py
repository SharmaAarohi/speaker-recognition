import argparse
import csv
import json
import os
from pathlib import Path
import random
import time
from typing import List, Optional, Dict

import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ----------------------------
# Utility
# ----------------------------
def find_speaker_dirs(root: Path) -> List[Path]:
    speakers = []
    for p in root.iterdir():
        if p.is_dir():
            if any(p.glob("*.wav")):
                speakers.append(p)
    speakers.sort()
    return speakers

def pick_files(folder: Path, k: int, seed: int) -> List[Path]:
    rng = np.random.default_rng(seed)
    wavs = sorted(folder.glob("*.wav"))
    if len(wavs) < k:
        raise RuntimeError(f"{folder.name} has {len(wavs)} wav files, need {k}.")
    picks_idx = rng.choice(len(wavs), size=k, replace=False)
    return [wavs[i] for i in picks_idx]

def load_audio(path: Path, sr: int) -> tuple[np.ndarray, int]:
    y, s = librosa.load(str(path), sr=sr, mono=True)
    return y, s

def extract_features(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    if y.size == 0:
        return None
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)

    # STFT magnitude for chroma + spectral contrast
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    spec_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    spec_contrast_mean = np.mean(spec_contrast, axis=1)

    feat = np.concatenate([mfcc_mean, chroma_mean, spec_contrast_mean], axis=0)
    if np.isnan(feat).any():
        return None
    return feat

def save_histogram(durations: Dict[str, List[float]], out_dir: Path):
    for spk, durs in durations.items():
        fig = plt.figure()
        plt.hist(durs, bins=10)
        plt.title(f"Duration Distribution (s) - {spk}")
        plt.xlabel("Duration (s)")
        plt.ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_dir / f"durations_{spk}.png", dpi=140)
        plt.close(fig)

def plot_confusion(cm: np.ndarray, labels: List[str], title: str, path: Path):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

def bar_plot(labels: List[str], values: List[float], title: str, path: Path):
    fig = plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Accuracy")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------
# Main pipeline
# ----------------------------
def run_pipeline(
    data_root: Path,
    out_root: Path,
    num_speakers: int,
    samples_per_speaker: int,
    sr: int,
    seed: int,
    fixed_speakers: Optional[List[str]] = None
):
    out_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Choose speakers
    speaker_dirs = find_speaker_dirs(data_root)
    if fixed_speakers:
        chosen = []
        for name in fixed_speakers:
            p = data_root / name
            if not p.exists():
                raise FileNotFoundError(f"Speaker folder not found: {name}")
            if not any(p.glob("*.wav")):
                raise RuntimeError(f"No wav files in: {name}")
            chosen.append(p)
        if len(chosen) != num_speakers:
            raise RuntimeError(f"You specified {len(chosen)} speakers, expected {num_speakers}.")
    else:
        eligible = [d for d in speaker_dirs if len(list(d.glob("*.wav"))) >= samples_per_speaker]
        if len(eligible) < num_speakers:
            raise RuntimeError(f"Not enough eligible speakers (need {num_speakers}).")
        random.seed(seed)
        chosen = random.sample(eligible, num_speakers)

    print("Chosen speakers:")
    for d in chosen:
        print(" -", d.name)

    # 2) Pick K samples per speaker + save mapping
    mapping = []
    for d in chosen:
        picks = pick_files(d, samples_per_speaker, seed)
        for p in picks:
            mapping.append({"filepath": str(p), "speaker": d.name})

    # Shuffle ("mix")
    random.seed(seed)
    random.shuffle(mapping)

    # Save mapping CSV
    mapping_csv = out_dir / "sample_speaker_mapping.csv"
    with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "speaker"])
        writer.writeheader()
        writer.writerows(mapping)
    print(f"Saved mapping to {mapping_csv}")

    # 3) Load & explore durations
    durations = {}
    for item in tqdm(mapping, desc="Loading for durations"):
        spk = item["speaker"]
        durations.setdefault(spk, [])
        y, _ = load_audio(Path(item["filepath"]), sr)
        dur = librosa.get_duration(y=y, sr=sr)
        durations[spk].append(dur)

    # Save duration plots
    save_histogram(durations, out_dir)

    # 4) Feature extraction + pooling
    X = []
    y_labels = []
    failed = 0

    for item in tqdm(mapping, desc="Extracting features"):
        path = Path(item["filepath"])
        label = item["speaker"]
        try:
            y, _ = load_audio(path, sr)
            feat = extract_features(y, sr)
            if feat is None:
                failed += 1
                continue
            X.append(feat)
            y_labels.append(label)
        except Exception as e:
            print(f"Feature extraction failed for {path}: {e}")
            failed += 1

    X = np.array(X)
    y_labels = np.array(y_labels)
    print(f"Feature matrix: {X.shape}, labels: {y_labels.shape}, failed: {failed}")

    # Save features CSV
    feat_csv = out_dir / "features.csv"
    header = [f"mfcc_{i+1}" for i in range(20)] + \
             [f"chroma_{i+1}" for i in range(12)] + \
             [f"spec_contrast_{i+1}" for i in range(X.shape[1] - 32)]
    with open(feat_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speaker"] + header)
        for i in range(len(y_labels)):
            w.writerow([y_labels[i]] + list(map(float, X[i])))
    print(f"Saved features to {feat_csv}")

    # 5) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_labels, test_size=0.3, random_state=seed, stratify=y_labels
    )

    # 6) Models
    # RandomForest
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"RandomForest accuracy: {rf_acc:.4f}")

    # Logistic Regression (with standardization)
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=seed, n_jobs=-1, multi_class="auto"))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"LogisticRegression accuracy: {lr_acc:.4f}")

    # 7) Reports + Confusion matrices
    labels_sorted = sorted(list(set(y_labels)))
    rf_cm = confusion_matrix(y_test, rf_pred, labels=labels_sorted)
    lr_cm = confusion_matrix(y_test, lr_pred, labels=labels_sorted)

    # Save text reports
    with open(out_dir / "classification_report_randomforest.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, rf_pred, zero_division=0))
    with open(out_dir / "classification_report_logreg.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, lr_pred, zero_division=0))

    # Save confusion matrix plots
    plot_confusion(rf_cm, labels_sorted, "Confusion Matrix - RandomForest", out_dir / "cm_randomforest.png")
    plot_confusion(lr_cm, labels_sorted, "Confusion Matrix - Logistic Regression", out_dir / "cm_logreg.png")

    # Accuracy bar chart
    bar_plot(["RandomForest", "LogReg"], [rf_acc, lr_acc],
             "Model Accuracies", out_dir / "accuracies.png")

    # 8) Save run metadata
    meta = {
        "data_root": str(data_root),
        "chosen_speakers": [d.name for d in chosen],
        "num_speakers": num_speakers,
        "samples_per_speaker": samples_per_speaker,
        "sample_rate": sr,
        "seed": seed,
        "n_samples_used": int(X.shape[0]),
        "failed_files": int(failed),
        "rf_accuracy": float(rf_acc),
        "lr_accuracy": float(lr_acc),
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nAll done ✅")
    print(f"Outputs folder: {out_dir.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description="Speaker Recognition (3-class) pipeline")
    p.add_argument("--data_root", type=str, required=True,
                   help="Path to dataset root (contains speaker subfolders).")
    p.add_argument("--out_root", type=str, default="outputs",
                   help="Where to write results.")
    p.add_argument("--num_speakers", type=int, default=3,
                   help="How many speakers to include.")
    p.add_argument("--samples_per_speaker", type=int, default=10,
                   help="How many samples per speaker.")
    p.add_argument("--sr", type=int, default=16000,
                   help="Target sample rate.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed.")
    p.add_argument("--speakers", type=str, default=None,
                   help="Comma-separated speaker folder names to force selection "
                        "(e.g., \"Speaker_0001,Speaker_0002,Speaker0033\").")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    fixed = [s.strip() for s in args.speakers.split(",")] if args.speakers else None

    run_pipeline(
        data_root=data_root,
        out_root=out_root,
        num_speakers=args.num_speakers,
        samples_per_speaker=args.samples_per_speaker,
        sr=args.sr,
        seed=args.seed,
        fixed_speakers=fixed
    )
