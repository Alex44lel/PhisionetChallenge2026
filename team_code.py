#!/usr/bin/env python

################################################################################
#
# team_code.py — PhysioNet Challenge 2026 Submission
#
# CNN classifier on REM-sleep EEG epochs for cognitive impairment prediction.
# Uses a two-stage preprocessing pipeline (preprocess → pack memmap) matching
# the offline scripts, with a fast path when preprocessed data already exists.
#
################################################################################

from helper_code import *
import os
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import mode, zscore
from tqdm import tqdm

import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)


################################################################################
# Constants
################################################################################

SEED = 42
SIGNAL_SHAPE = (9, 3840)          # 9 channels x 30s @ 128 Hz
TARGET_FS = 128
EPOCH_DURATION = 30
CHANNELS_OF_INTEREST = ["f3", "f4", "c3", "c4", "o1", "o2", "e1", "e2", "chin"]
REM_STAGE_LABEL = 4               # After STAGE_MAP remapping
MEMMAP_DTYPE = np.float16

# Stage/resp remapping (from scripts/utils.py)
STAGE_MAP = {9: -1, 5: 0, 3: 1, 2: 2, 1: 3, 4: 4}
RESP_MAP = {0: 0, 1: 1, 8: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4, 9: 4, 7: 5}

# Training hyperparameters (same as model_to_submit.py)
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 12

################################################################################
# Reproducibility
################################################################################


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


################################################################################
# Preprocessing helpers (from scripts/utils.py)
################################################################################


def process_and_epoch_eeg(eeg_file_path):
    """
    Read a raw physiological EDF, filter, resample, z-score normalise, and
    segment into 30-second epochs.  Missing channels are zero-padded.

    Returns
    -------
    X_epochs : torch.Tensor, shape [N_epochs, 9, 3840]
    channel_names : list[str]
    """
    raw_eeg = mne.io.read_raw_edf(str(eeg_file_path), preload=True, verbose=False)

    # Map target channels to available channels in the EDF
    ch_mapping = {}  # target_idx -> actual_channel_name
    for i, target_ch in enumerate(CHANNELS_OF_INTEREST):
        for ch in raw_eeg.ch_names:
            if target_ch in ch.lower():
                ch_mapping[i] = ch
                break

    available_picks = list(ch_mapping.values())
    if not available_picks:
        raise ValueError(f"Missing ALL target channels in {eeg_file_path}")

    raw_eeg.pick(available_picks)

    # Filter & resample
    raw_eeg.filter(0.3, 35.0, verbose=False)
    if raw_eeg.info["sfreq"] != TARGET_FS:
        raw_eeg.resample(TARGET_FS)

    # Z-score normalise
    norm_data = zscore(raw_eeg.get_data(), axis=-1)

    # Zero-padded tensor for all 9 target channels
    num_samples = norm_data.shape[1]
    padded_signal = np.zeros((len(CHANNELS_OF_INTEREST), num_samples), dtype=np.float32)

    for target_idx, avail_name in ch_mapping.items():
        avail_idx = raw_eeg.ch_names.index(avail_name)
        padded_signal[target_idx, :] = norm_data[avail_idx, :]

    # Cut into epochs
    samples_per_epoch = int(EPOCH_DURATION * TARGET_FS)
    total_epochs = padded_signal.shape[1] // samples_per_epoch

    X_epochs = []
    for i in range(total_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        X_epochs.append(padded_signal[:, start:end])

    return torch.tensor(np.stack(X_epochs), dtype=torch.float32), CHANNELS_OF_INTEREST


def get_mode_label(data_array, start_idx, end_idx, mapping_dict=None, is_limb=False):
    """Extract the most frequent annotation value in a 30s window."""
    if end_idx > len(data_array):
        return -1

    epoch_data = np.round(data_array[start_idx:end_idx]).astype(int)
    predominant_val = int(mode(epoch_data, keepdims=False)[0])

    if is_limb:
        if predominant_val == 0:
            return 0
        elif predominant_val == 2:
            return 2
        else:
            return 1

    if mapping_dict is not None:
        return mapping_dict.get(predominant_val, -1)

    return predominant_val


################################################################################
# Stage 1: Preprocess EDFs → .pt files (adapted from preprocess_psg.py)
################################################################################


def preprocess_edfs(data_folder, output_dir, verbose):
    """
    Read raw EDF files from the challenge data_folder, preprocess signals,
    extract per-epoch annotations, and save per-patient .pt files.

    Adapted from scripts/preprocess_psg.py to work with the challenge
    data_folder structure (no 'training_set' subfolder).
    """
    physio_path = Path(data_folder) / PHYSIOLOGICAL_DATA_SUBFOLDER
    algo_path = Path(data_folder) / ALGORITHMIC_ANNOTATIONS_SUBFOLDER
    human_path = Path(data_folder) / HUMAN_ANNOTATIONS_SUBFOLDER
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build set of valid patients from demographics.csv so we only
    # preprocess patients that actually appear in the dataset
    demo_path = Path(data_folder) / DEMOGRAPHICS_FILE
    df_demo = pd.read_csv(demo_path)
    valid_patients = set()
    for _, row in df_demo.iterrows():
        bids = str(row[HEADERS["bids_folder"]])
        site = str(row[HEADERS["site_id"]])
        session = str(row[HEADERS["session_id"]])
        valid_patients.add((site, f"{bids}_ses-{session}"))

    master_index_records = []

    hospitals = sorted([d for d in physio_path.iterdir() if d.is_dir()])

    for hosp_dir in hospitals:
        hosp_id = hosp_dir.name
        if verbose:
            print(f"\n>>> Processing site: {hosp_id}")
        (out_path / hosp_id).mkdir(exist_ok=True)

        eeg_files = sorted(hosp_dir.glob("*.edf"))
        # Filter to only patients listed in demographics.csv
        eeg_files = [f for f in eeg_files if (hosp_id, f.stem) in valid_patients]

        for eeg_file in tqdm(eeg_files, desc=f"  {hosp_id}", disable=not verbose):
            subject_id = eeg_file.stem
            caisr_file = algo_path / hosp_id / f"{subject_id}_caisr_annotations.edf"

            # CAISR annotations are required; expert annotations are optional
            if not caisr_file.exists():
                continue

            expert_file = human_path / hosp_id / f"{subject_id}_expert_annotations.edf"
            has_expert = expert_file.exists()

            try:
                # 1. Process EEG
                X_tensor, channel_names = process_and_epoch_eeg(eeg_file)
                total_epochs = X_tensor.shape[0]

                # 2. Load CAISR (algorithmic) annotations
                raw_ia = mne.io.read_raw_edf(str(caisr_file), preload=True, verbose=False)
                ia_sfreq = raw_ia.info["sfreq"]

                Y_ia = {"stage": [], "arousal": [], "resp": [], "limb": []}

                # 3. Load expert annotations if available
                Y_expert = {"stage": [], "arousal": [], "resp": [], "limb": []}
                raw_exp = None
                exp_sfreq = None
                if has_expert:
                    raw_exp = mne.io.read_raw_edf(str(expert_file), preload=True, verbose=False)
                    exp_sfreq = raw_exp.info["sfreq"]

                # 4. Extract per-epoch labels
                for i in range(total_epochs):
                    # CAISR labels
                    start_ia = int(i * EPOCH_DURATION * ia_sfreq)
                    end_ia = int(start_ia + EPOCH_DURATION * ia_sfreq)
                    Y_ia["stage"].append(get_mode_label(
                        raw_ia.get_data(picks="stage_caisr")[0], start_ia, end_ia, STAGE_MAP))
                    Y_ia["arousal"].append(get_mode_label(
                        raw_ia.get_data(picks="arousal_caisr")[0], start_ia, end_ia))
                    Y_ia["resp"].append(get_mode_label(
                        raw_ia.get_data(picks="resp_caisr")[0], start_ia, end_ia, RESP_MAP))
                    Y_ia["limb"].append(get_mode_label(
                        raw_ia.get_data(picks="limb_caisr")[0], start_ia, end_ia, is_limb=True))

                    # Expert labels (if available)
                    if has_expert:
                        start_exp = int(i * EPOCH_DURATION * exp_sfreq)
                        end_exp = int(start_exp + EPOCH_DURATION * exp_sfreq)
                        Y_expert["stage"].append(get_mode_label(
                            raw_exp.get_data(picks="stage_expert")[0], start_exp, end_exp, STAGE_MAP))
                        Y_expert["arousal"].append(get_mode_label(
                            raw_exp.get_data(picks="arousal_expert")[0], start_exp, end_exp))
                        Y_expert["resp"].append(get_mode_label(
                            raw_exp.get_data(picks="resp_expert")[0], start_exp, end_exp, RESP_MAP))
                        Y_expert["limb"].append(get_mode_label(
                            raw_exp.get_data(picks="limb_expert")[0], start_exp, end_exp, is_limb=True))
                    else:
                        for k in Y_expert:
                            Y_expert[k].append(-1)

                    master_index_records.append({
                        "subject_id": subject_id,
                        "hospital": hosp_id,
                        "epoch_idx": i,
                        "pt_file_path": str(out_path / hosp_id / f"{subject_id}.pt"),
                    })

                # 5. Save .pt file
                save_dict = {
                    "signal": X_tensor,
                    "expert": {k: torch.tensor(v, dtype=torch.long) for k, v in Y_expert.items()},
                    "ia": {k: torch.tensor(v, dtype=torch.long) for k, v in Y_ia.items()},
                    "channels": channel_names,
                }
                torch.save(save_dict, out_path / hosp_id / f"{subject_id}.pt")

            except Exception as e:
                if verbose:
                    tqdm.write(f"    [X] Error processing {subject_id}: {e}")
                continue

    # Save master index CSV
    pd.DataFrame(master_index_records).to_csv(out_path / "master_index.csv", index=False)
    if verbose:
        n_subjects = len(set(r["subject_id"] for r in master_index_records))
        print(f"\nPreprocessing complete. {n_subjects} patients, "
              f"{len(master_index_records)} epochs indexed.")


################################################################################
# Stage 2: Pack .pt files → memmap (adapted from pack_memmap.py)
################################################################################


def pack_to_memmap(preproc_dir, verbose):
    """
    Repack per-patient .pt files into a flat numpy memmap for fast random
    access during training.

    Adapted from scripts/pack_memmap.py.
    """
    preproc_dir = Path(preproc_dir)
    index_path = preproc_dir / "master_index.csv"
    memmap_path = preproc_dir / "signals.bin"
    packed_csv_path = preproc_dir / "master_index_packed.csv"

    df = pd.read_csv(index_path)
    N = len(df)
    if verbose:
        print(f"\nPacking {N:,} epochs into memmap "
              f"({N * 9 * 3840 * 2 / 1e9:.1f} GB)...")

    # Pre-allocate memmap
    mmap = np.memmap(memmap_path, dtype=MEMMAP_DTYPE, mode="w+", shape=(N, *SIGNAL_SHAPE))

    # Label columns
    label_keys = ["stage", "arousal", "resp", "limb"]
    for prefix in ("expert", "ia"):
        for k in label_keys:
            df[f"{prefix}_{k}"] = np.int16(-1)
    df["memmap_idx"] = np.int32(-1)

    # Process one .pt file at a time
    groups = list(df.groupby("pt_file_path", sort=False))
    global_offset = 0

    for pt_path, group in tqdm(groups, desc="Packing", unit="file", disable=not verbose):
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            if verbose:
                tqdm.write(f"  [X] Could not load {pt_path}: {e}")
            global_offset += len(group)
            continue

        signal = data["signal"]
        expert = data.get("expert", {})
        ia = data.get("ia", {})

        epoch_indices = group["epoch_idx"].values

        for local_pos, epoch_idx in enumerate(epoch_indices):
            mmap_row = global_offset + local_pos

            # Write signal
            mmap[mmap_row] = signal[epoch_idx].numpy().astype(MEMMAP_DTYPE)

            # Record position
            df_row = group.index[local_pos]
            df.at[df_row, "memmap_idx"] = mmap_row

            # Write annotation labels
            for k in label_keys:
                df.at[df_row, f"expert_{k}"] = int(expert[k][epoch_idx]) if k in expert else -1
                df.at[df_row, f"ia_{k}"] = int(ia[k][epoch_idx]) if k in ia else -1

        global_offset += len(group)

    mmap.flush()
    del mmap

    df.to_csv(packed_csv_path, index=False)
    if verbose:
        assigned = (df["memmap_idx"] >= 0).sum()
        print(f"Done. {assigned:,} / {N:,} epochs packed.")

    # Clean up .pt files to free disk space
    for pt_path, _ in groups:
        try:
            os.remove(pt_path)
        except OSError:
            pass
    # Remove empty site directories
    for d in preproc_dir.iterdir():
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass


################################################################################
# CAISR annotation helpers (for inference)
################################################################################


def get_rem_epoch_indices(caisr_path, n_eeg_epochs):
    """
    Read the CAISR algorithmic annotation EDF and return epoch indices
    where the predominant sleep stage is REM.
    Uses STAGE_MAP to remap raw values, then checks for REM_STAGE_LABEL (4).
    """
    raw = mne.io.read_raw_edf(str(caisr_path), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    stage_data = raw.get_data(picks="stage_caisr")[0]

    rem_indices = []
    for i in range(n_eeg_epochs):
        start = int(i * EPOCH_DURATION * sfreq)
        end = int(start + EPOCH_DURATION * sfreq)
        if end > len(stage_data):
            break
        epoch_stages = np.round(stage_data[start:end]).astype(int)
        predominant_raw = int(mode(epoch_stages, keepdims=False)[0])
        remapped = STAGE_MAP.get(predominant_raw, -1)
        if remapped == REM_STAGE_LABEL:
            rem_indices.append(i)

    return rem_indices


################################################################################
# Group assignment (from data_pipeline.py / model_to_submit.py)
################################################################################

DAYS_3Y = 1095
DAYS_7Y = 2555
DAYS_6Y = 2190   # Exception for site I0002


def assign_group(row):
    """Assign challenge group: 1 = CI, 0 = healthy control, 3 = excluded."""
    site = str(row["SiteID"])
    req = DAYS_6Y if site == "I0002" else DAYS_7Y

    if row["Cognitive_Impairment"] is True or row["Cognitive_Impairment"] == "True":
        if pd.notna(row["Time_to_Event"]) and (DAYS_3Y <= row["Time_to_Event"] <= DAYS_7Y):
            return 1
        return 3
    else:
        if pd.notna(row["Time_to_Last_Visit"]) and row["Time_to_Last_Visit"] >= req:
            return 0
        return 3


################################################################################
# Model architecture (from model_to_submit.py)
################################################################################


class SE(nn.Module):
    """Squeeze-and-Excitation block for 1-D signals."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)
        return x * w


class SimpleCNN(nn.Module):
    """
    CNN with SE blocks for 30-second REM-epoch classification.
    Input:  [B, 9, 3840]
    Output: [B, 1]
    """

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=15, stride=8, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.se1 = SE(32)

        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.se2 = SE(64)

        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.se3 = SE(128)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.se1(self.block1(x))
        x = self.se2(self.block2(x))
        x = self.se3(self.block3(x))
        return self.head(x)


################################################################################
# Dataset (memmap-backed, from model_to_submit.py)
################################################################################


class SleepDataset(Dataset):
    """Dataset backed by a memory-mapped signal array."""

    def __init__(self, df, mmap):
        self.df = df.reset_index(drop=True)
        self.mmap = mmap

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal = torch.from_numpy(self.mmap[int(row["memmap_idx"])].astype(np.float32))
        target = torch.tensor(row["group_label"], dtype=torch.float32)
        return {"signal": signal, "target": target, "subject_id": row["subject_id"]}


################################################################################
# Evaluation (patient-level, from model_to_submit.py)
################################################################################


@torch.no_grad()
def predict_patient(model, epochs, device, batch_size=64):
    """Mean sigmoid probability over all REM epochs of one patient."""
    model.eval()
    probs = []
    for i in range(0, len(epochs), batch_size):
        batch = epochs[i: i + batch_size].to(device)
        probs.append(torch.sigmoid(model(batch)).squeeze(1).cpu())
    return torch.cat(probs).mean().item()


def evaluate(model, df_val, mmap, device):
    """
    Patient-level evaluation.
    Returns dict with auroc, balanced_accuracy, threshold.
    """
    model.eval()
    patient_probs, patient_labels = [], []

    for _, group in df_val.groupby("subject_id"):
        label = group["group_label"].iloc[0]
        memmap_indices = group["memmap_idx"].values.astype(int)
        epochs = torch.from_numpy(mmap[memmap_indices].astype(np.float32))
        prob = predict_patient(model, epochs, device)
        patient_probs.append(prob)
        patient_labels.append(label)

    y_true = np.array(patient_labels)
    y_prob = np.array(patient_probs)

    if len(set(patient_labels)) < 2:
        return {"auroc": 0.5, "balanced_accuracy": 0.0, "threshold": 0.5}

    # Find threshold that maximises balanced accuracy
    best_thr, best_bacc = 0.5, 0.0
    for thr in np.arange(0.05, 0.96, 0.01):
        bacc = balanced_accuracy_score(y_true, (y_prob >= thr).astype(int))
        if bacc > best_bacc:
            best_bacc = bacc
            best_thr = thr

    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "balanced_accuracy": best_bacc,
        "threshold": round(best_thr, 2),
    }


################################################################################
# Data loading (from model_to_submit.py load_data)
################################################################################


def load_training_data(packed_csv, signals_bin, demo_path, verbose):
    """
    Load preprocessed memmap data, assign group labels, filter to REM epochs,
    and return train/val DataLoaders plus the memmap reference.

    Returns (train_loader, val_loader, df_val, mmap)
    """
    df_index = pd.read_csv(packed_csv)
    df_index = df_index[df_index["memmap_idx"] >= 0]
    df_demo = pd.read_csv(demo_path)

    # Clean subject IDs (match format between index and demographics)
    df_demo = df_demo.rename(columns={"BidsFolder": "subject_id"})
    df_demo["subject_id"] = (
        df_demo["subject_id"].astype(str).str.replace("sub-", "", regex=False).str.strip()
    )
    df_index["subject_id"] = df_index["subject_id"].astype(str).str.replace("sub-", "", regex=False)
    df_index["subject_id"] = df_index["subject_id"].str.split("_").str[0].str.strip()

    # Assign group labels
    df_demo["group_label"] = df_demo.apply(assign_group, axis=1)
    df_demo = df_demo[df_demo["group_label"] != 3].copy()
    df_demo = df_demo[["subject_id", "group_label"]].copy()

    # Filter to REM epochs
    if "ia_stage" in df_index.columns:
        rem_mask = (df_index["ia_stage"] == REM_STAGE_LABEL) | \
                   ((df_index["ia_stage"] == -1) & (df_index["expert_stage"] == REM_STAGE_LABEL))
    else:
        rem_mask = df_index["expert_stage"] == REM_STAGE_LABEL
    df_index = df_index[rem_mask].copy()

    # Merge
    df = df_index.merge(df_demo, on="subject_id", how="inner")
    if verbose:
        print(f"REM epochs: {len(df):,} | Patients: {df['subject_id'].nunique()}")

    # Open memmap
    memmap_n = int(pd.read_csv(packed_csv)["memmap_idx"].max()) + 1
    mmap = np.memmap(signals_bin, dtype=MEMMAP_DTYPE, mode="r", shape=(memmap_n, *SIGNAL_SHAPE))

    # Train/val split (patient-level)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df, groups=df["subject_id"]))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    if verbose:
        print(f"Train: {len(df_train):,} epochs | Val: {len(df_val):,} epochs")

    train_loader = DataLoader(
        SleepDataset(df_train, mmap),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        SleepDataset(df_val, mmap),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, val_loader, df_val, mmap


################################################################################
#
# Required functions
#
################################################################################


def train_model(data_folder, model_folder, verbose):
    """
    Two-stage preprocessing pipeline with fast-path caching:

      1. Check for existing preprocessed data (skip if found):
         a) data_folder/preprocessed_data/  (your local pre-built data)
         b) model_folder/preprocessed/      (cached from a previous run)
      2. If not found, run:
         a) Stage 1: preprocess_edfs() — raw EDFs → per-patient .pt files
         b) Stage 2: pack_to_memmap()  — .pt files → signals.bin + index CSV
      3. Load memmap, filter REM epochs, assign group labels
      4. Train SimpleCNN with augmentation + early stopping
      5. Save BEST checkpoint (by val AUROC) to model_folder/model.pt
    """
    t_start = time.time()
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(model_folder, exist_ok=True)

    if verbose:
        print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Find or create preprocessed data
    # ------------------------------------------------------------------
    preproc_dir = None
    search_dirs = [
        os.path.join(data_folder, "preprocessed_data"),   # local fast path
        os.path.join(model_folder, "preprocessed"),        # cached from previous run
    ]

    for candidate in search_dirs:
        packed_csv = os.path.join(candidate, "master_index_packed.csv")
        signals_bin = os.path.join(candidate, "signals.bin")
        print(packed_csv)
        print(signals_bin)
        print(os.path.exists(packed_csv), os.path.exists(signals_bin))
        if os.path.exists(packed_csv) and os.path.exists(signals_bin):
            preproc_dir = candidate
            if verbose:
                print(f"Found existing preprocessed data at: {preproc_dir}")
            break

    if preproc_dir is None:
        # Run both preprocessing stages
        preproc_dir = os.path.join(model_folder, "preprocessed")
        if verbose:
            print("No preprocessed data found. Running preprocessing pipeline...")
            print(f"Output directory: {preproc_dir}")

        if verbose:
            print("\n=== Stage 1: Preprocessing EDFs → .pt files ===")
        preprocess_edfs(data_folder, preproc_dir, verbose)

        if verbose:
            print("\n=== Stage 2: Packing .pt files → memmap ===")
        pack_to_memmap(preproc_dir, verbose)

    packed_csv = os.path.join(preproc_dir, "master_index_packed.csv")
    signals_bin = os.path.join(preproc_dir, "signals.bin")
    demo_path = os.path.join(data_folder, DEMOGRAPHICS_FILE)

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    if verbose:
        print("\n=== Loading training data ===")

    train_loader, val_loader, df_val, mmap = load_training_data(
        packed_csv, signals_bin, demo_path, verbose,
    )

    # ------------------------------------------------------------------
    # 3. Model, optimizer, criterion
    # ------------------------------------------------------------------
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6,
    )
    criterion = nn.BCEWithLogitsLoss()

    ckpt_path = os.path.join(model_folder, "model.pt")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Model parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    best_val_auroc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    smooth_loss = None

    for epoch in range(1, 201):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            signal = batch["signal"].to(device)
            target = batch["target"].unsqueeze(1).to(device)

            # Gaussian noise augmentation
            signal = signal + 0.1 * torch.randn_like(signal)

            # Mixup augmentation
            lam = np.random.beta(0.2, 0.2)
            idx = torch.randperm(signal.size(0), device=device)
            signal = lam * signal + (1 - lam) * signal[idx]
            target = lam * target + (1 - lam) * target[idx]

            optimizer.zero_grad()
            pred = model(signal)
            loss = criterion(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            if smooth_loss is None:
                smooth_loss = loss_val
            else:
                smooth_loss = 0.95 * smooth_loss + 0.05 * loss_val

        avg_train_loss = np.mean(epoch_losses)
        scheduler.step()

        # --- Validation: epoch-level loss ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                signal = batch["signal"].to(device)
                target = batch["target"].unsqueeze(1).to(device)
                pred = model(signal)
                val_losses.append(criterion(pred, target).item())
        avg_val_loss = np.mean(val_losses)

        # --- Validation: patient-level AUROC ---
        val_metrics = evaluate(model, df_val, mmap, device)
        val_auroc = val_metrics["auroc"]

        # --- Early stopping on patient-level AUROC ---
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            epochs_no_improve = 0
            # Save BEST checkpoint immediately
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_auroc": val_auroc,
                    "threshold": val_metrics["threshold"],
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1

        if verbose:
            elapsed = time.time() - t_start
            print(
                f"epoch {epoch:03d} | loss: {avg_train_loss:.4f} | "
                f"val_loss: {avg_val_loss:.4f} | auroc: {val_auroc:.4f} | "
                f"bacc: {val_metrics['balanced_accuracy']:.4f} | "
                f"thr: {val_metrics['threshold']:.2f} | "
                f"best: {best_val_auroc:.4f} (ep{best_epoch}) | "
                f"patience: {EARLY_STOP_PATIENCE - epochs_no_improve} | "
                f"elapsed: {elapsed:.0f}s"
            )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            if verbose:
                print(f"Early stopping at epoch {epoch}.")
            break

        if smooth_loss is not None and (np.isnan(smooth_loss) or smooth_loss > 10):
            if verbose:
                print(f"Loss diverged ({smooth_loss}), stopping.")
            break

    if verbose:
        print(f"Training complete. Best AUROC: {best_val_auroc:.4f} at epoch {best_epoch}.")
        print("Done.\n")


def load_model(model_folder, verbose):
    """Load the trained SimpleCNN from model_folder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    ckpt = torch.load(
        os.path.join(model_folder, "model.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "device": device,
        "threshold": ckpt.get("threshold", 0.5),
    }


def run_model(model, record, data_folder, verbose):
    """
    Run the trained model on a single patient record.

    Steps:
      1. Preprocess the patient's physiological EDF into 30s epochs
      2. Identify REM epochs via CAISR algorithmic annotations
      3. Run CNN on REM epochs and aggregate to patient-level probability
      4. Return (binary_prediction, probability)
    """
    cnn = model["model"]
    device = model["device"]
    threshold = model["threshold"]

    patient_id = record[HEADERS["bids_folder"]]
    site_id = record[HEADERS["site_id"]]
    session_id = record[HEADERS["session_id"]]

    edf_path = os.path.join(
        data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id,
        f"{patient_id}_ses-{session_id}.edf",
    )
    caisr_path = os.path.join(
        data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id,
        f"{patient_id}_ses-{session_id}_caisr_annotations.edf",
    )

    # ------------------------------------------------------------------
    # Edge case: missing physiological EDF
    # ------------------------------------------------------------------
    if not os.path.exists(edf_path):
        return 0, 0.5

    try:
        epochs_tensor, _ = process_and_epoch_eeg(edf_path)
        n_epochs = epochs_tensor.shape[0]
    except Exception:
        return 0, 0.5

    # ------------------------------------------------------------------
    # Identify REM epochs; fall back to all epochs if CAISR missing
    # ------------------------------------------------------------------
    rem_epochs = None
    if os.path.exists(caisr_path):
        try:
            rem_idx = get_rem_epoch_indices(caisr_path, n_epochs)
            if len(rem_idx) > 0:
                rem_epochs = epochs_tensor[rem_idx]
        except Exception:
            pass

    if rem_epochs is None or len(rem_epochs) == 0:
        # Fallback: use all epochs
        rem_epochs = epochs_tensor

    # ------------------------------------------------------------------
    # Run model
    # ------------------------------------------------------------------
    probability = predict_patient(cnn, rem_epochs, device)
    binary = int(probability >= threshold)

    return binary, probability
