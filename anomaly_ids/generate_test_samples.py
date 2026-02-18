"""
Generate realistic test samples from a dataset CSV file.
Supports KDD Cup 1999 and UNSW-NB15 datasets (and any compatible dataset).

Usage:
    # KDD (default, two files)
    python generate_test_samples.py

    # KDD explicit
    python generate_test_samples.py --dataset kdd --train training/KDDTrain.csv --test training/KDDTest.csv

    # UNSW (single file, all attacks)
    python generate_test_samples.py --dataset unsw --csv training/UNSW-NB15_Dataset2.csv --output test_samples_unsw.json

    # Generic single CSV
    python generate_test_samples.py --csv training/mydata.csv --output test_samples_custom.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Known label column names (in priority order)
# ---------------------------------------------------------------------------
KNOWN_LABEL_COLS = ["attack_class", "attack_class_category", "label", "attack"]

# UNSW-NB15 feature columns.
# Two releases exist with slightly different names:
#   UNSW-NB15_Dataset2.csv : lowercase, 39 cols, no proto/state
#   UNSW-NB15.csv (full)   : mixed-case, 42 cols, includes proto/state
# This list covers the full dataset; detect_feature_cols() picks what's present.
UNSW_FEATURE_COLS = [
    # Categorical (full dataset only)
    "proto", "state",
    # Shared timing / volume
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    # Load – full dataset uses capitals
    "Sload", "Dload",
    # Packet counts – full dataset uses capitals
    "Spkts", "Dpkts",
    # Window / TCP
    "swin", "dwin", "stcpb", "dtcpb",
    # Mean packet size – full dataset uses smeansz/dmeansz
    "smeansz", "dmeansz",
    # HTTP
    "trans_depth", "res_bdy_len",
    # Jitter – full dataset uses capitals
    "Sjit", "Djit",
    # Inter-packet – full dataset uses Sintpkt/Dintpkt
    "Sintpkt", "Dintpkt",
    # TCP handshake
    "tcprtt", "synack", "ackdat",
    # Rate / misc
    "rate",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
    "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    # Note: 'ct_src_ ltm' (with space) in raw CSV is normalised to 'ct_src_ltm'
    "ct_src_ltm",
]

# KDD Cup 1999 feature columns (41 features, 3 categorical)
KDD_FEATURE_COLS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_label_col(df):
    """Return the first matching label column found in the DataFrame."""
    for col in KNOWN_LABEL_COLS:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not detect label column. Tried: {KNOWN_LABEL_COLS}. "
        f"Available columns: {list(df.columns)}"
    )


def detect_normal_label(df, label_col):
    """
    Return the value that represents *normal* traffic, or None if not found.
    Handles both string labels ('normal', 'benign') and integer labels (0 = normal).
    For UNSW-NB15_Dataset2.csv the dataset contains only attack traffic - no normal label.
    """
    # Integer 0/1 column (e.g., UNSW-NB15.csv full dataset)
    if pd.api.types.is_integer_dtype(df[label_col]):
        if 0 in df[label_col].values:
            return 0
        return None
    # String label column
    classes = set(df[label_col].astype(str).str.lower().unique())
    for candidate in ("normal", "benign", "legitimate", "0"):
        if candidate in classes:
            match = df[df[label_col].astype(str).str.lower() == candidate][label_col].iloc[0]
            return match
    return None


def detect_feature_cols(df):
    """Return the feature columns to include in the API payload."""
    if all(c in df.columns for c in ["sbytes", "dbytes", "spkts"]):
        return [c for c in UNSW_FEATURE_COLS if c in df.columns]
    if "protocol_type" in df.columns:
        if "is_host_login" not in df.columns:
            df["is_host_login"] = 0
        return [c for c in KDD_FEATURE_COLS if c in df.columns]
    # Fallback: use all non-label, non-id columns
    known_meta = set(KNOWN_LABEL_COLS) | {"id", "index"}
    return [c for c in df.columns if c not in known_meta]


def row_to_dict(row, feature_cols):
    """Convert a DataFrame row to a JSON-serialisable feature dict."""
    data = {}
    for col in feature_cols:
        val = row.get(col, 0)
        try:
            if hasattr(val, "item"):
                val = val.item()
            if isinstance(val, float):
                data[col] = float(val)
            elif isinstance(val, int):
                data[col] = int(val)
            else:
                data[col] = str(val)
        except Exception:
            data[col] = str(val)
    return data


# ---------------------------------------------------------------------------
# KDD-specific generator (keeps original behaviour + novel attacks)
# ---------------------------------------------------------------------------

def generate_kdd(train_csv, test_csv, n_normal, n_attack, seed):
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    df = pd.concat([df_train, df_test], ignore_index=True)

    label_col = detect_label_col(df)
    feature_cols = detect_feature_cols(df)

    test_cases = []

    # Normal samples - stratify by service
    normal_df = df[df[label_col] == "normal"]
    normal_services = [
        "http", "ftp_data", "smtp", "private", "telnet",
        "ftp", "ssh", "domain_u", "pop_3", "other",
    ]
    for i, svc in enumerate(normal_services):
        if "service" in normal_df.columns:
            subset = normal_df[normal_df["service"] == svc]
        else:
            subset = pd.DataFrame()
        if len(subset) == 0:
            subset = normal_df
        row = subset.sample(1, random_state=seed + i).iloc[0]
        test_cases.append({
            "name": f"KDD Normal {svc.upper()} Traffic #{i + 1}",
            "expected": "normal",
            "data": row_to_dict(row, feature_cols),
        })

    extra_normal = max(0, n_normal - len(normal_services))
    for i in range(extra_normal):
        row = normal_df.sample(1, random_state=100 + seed + i).iloc[0]
        test_cases.append({
            "name": f"KDD Normal Traffic (random) #{i + 1}",
            "expected": "normal",
            "data": row_to_dict(row, feature_cols),
        })

    # Attack samples
    attack_labels = {
        "neptune": "DoS SYN Flood (neptune)",
        "smurf": "DoS ICMP Flood (smurf)",
        "satan": "Probe - Satan Scanner",
        "ipsweep": "Probe - IP Sweep",
        "portsweep": "Probe - Port Sweep",
        "nmap": "Probe - Nmap Scan",
        "back": "DoS - Back (Apache)",
        "teardrop": "DoS - Teardrop",
        "pod": "DoS - Ping of Death",
        "warezclient": "R2L - Warez Client",
        "guess_passwd": "R2L - Password Guessing",
        "buffer_overflow": "U2R - Buffer Overflow",
        "rootkit": "U2R - Rootkit",
        "land": "DoS - Land Attack",
        "loadmodule": "U2R - Loadmodule",
        "ftp_write": "R2L - FTP Write",
        "imap": "R2L - IMAP Attack",
        "multihop": "R2L - Multihop",
        "warezmaster": "R2L - Warez Master",
    }
    for attack_name, friendly in attack_labels.items():
        subset = df[df[label_col] == attack_name]
        if len(subset) == 0:
            continue
        n_samples = min(3, len(subset))
        for j in range(n_samples):
            row = subset.sample(1, random_state=200 + seed + j).iloc[0]
            suffix = f" #{j + 1}" if n_samples > 1 else ""
            test_cases.append({
                "name": f"KDD Attack: {friendly}{suffix}",
                "expected": "intrusion",
                "data": row_to_dict(row, feature_cols),
            })

    # Novel attacks - present in test set only
    train_classes = set(df_train[label_col].unique())
    novel_df = df_test[~df_test[label_col].isin(train_classes)]
    for attack_name in novel_df[label_col].unique()[:8]:
        subset = novel_df[novel_df[label_col] == attack_name]
        row = subset.sample(1, random_state=300 + seed).iloc[0]
        test_cases.append({
            "name": f"KDD Novel Attack: {attack_name}",
            "expected": "intrusion",
            "data": row_to_dict(row, feature_cols),
        })

    return test_cases


# ---------------------------------------------------------------------------
# Generic / UNSW generator
# ---------------------------------------------------------------------------

def generate_generic(csv_path, n_normal, n_attack, seed, dataset_tag="UNSW"):
    df = pd.read_csv(csv_path)

    # Normalise column names: strip whitespace and fix known typo 'ct_src_ ltm'
    df.columns = df.columns.str.strip()
    if "ct_src_ ltm" in df.columns:
        df = df.rename(columns={"ct_src_ ltm": "ct_src_ltm"})

    label_col = detect_label_col(df)
    feature_cols = detect_feature_cols(df)
    normal_label = detect_normal_label(df, label_col)

    test_cases = []

    # Normal samples
    if normal_label is not None:
        normal_df = df[df[label_col] == normal_label]
        for i in range(min(n_normal, len(normal_df))):
            row = normal_df.sample(1, random_state=seed + i).iloc[0]
            test_cases.append({
                "name": f"{dataset_tag} Normal Traffic #{i + 1}",
                "expected": "normal",
                "data": row_to_dict(row, feature_cols),
            })
    else:
        print(
            f"[WARNING] No normal/benign class found in '{label_col}'. "
            f"All samples in this dataset appear to be attack traffic. "
            f"Skipping normal test cases.",
            file=sys.stderr,
        )

    # Attack samples - sample evenly across all attack classes
    attack_classes = df[label_col].unique().tolist()
    if normal_label is not None:
        # Compare correctly whether label is int or string
        attack_classes = [c for c in attack_classes if c != normal_label]

    # Label for display: integer 1 → "attack", otherwise use the value itself
    def attack_display_name(val):
        if isinstance(val, (int, float)) and val == 1:
            return "attack"
        return str(val)

    per_class = max(1, n_attack // max(len(attack_classes), 1))
    remainder = n_attack - per_class * len(attack_classes)

    for idx, attack_name in enumerate(attack_classes):
        subset = df[df[label_col] == attack_name]
        take = per_class + (1 if idx < remainder else 0)
        take = min(take, len(subset))
        sampled = subset.sample(min(take, len(subset)), random_state=seed + idx)
        for j, (_, row) in enumerate(sampled.iterrows()):
            test_cases.append({
                "name": f"{dataset_tag} Attack: {attack_display_name(attack_name)} #{j + 1}",
                "expected": "intrusion",
                "data": row_to_dict(row, feature_cols),
            })

    return test_cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate test_samples.json for the Hybrid IDS API."
    )
    parser.add_argument(
        "--dataset", choices=["kdd", "unsw", "generic"],
        default=None,
        help="Dataset type. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to a single CSV file (used for UNSW / generic datasets).",
    )
    parser.add_argument(
        "--train", default="training/KDDTrain.csv",
        help="Path to KDD training CSV (default: training/KDDTrain.csv).",
    )
    parser.add_argument(
        "--test", default="training/KDDTest.csv",
        help="Path to KDD test CSV (default: training/KDDTest.csv).",
    )
    parser.add_argument(
        "--output", default="test_samples.json",
        help="Output JSON file path (default: test_samples.json).",
    )
    parser.add_argument(
        "--normal", type=int, default=20,
        help="Number of normal samples to include (default: 20).",
    )
    parser.add_argument(
        "--attack", type=int, default=60,
        help="Number of attack samples to include (default: 60).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    # Auto-detect dataset type
    dataset = args.dataset
    if dataset is None:
        if args.csv is not None:
            csv_lower = args.csv.lower()
            if "unsw" in csv_lower:
                dataset = "unsw"
            else:
                dataset = "generic"
        else:
            dataset = "kdd"

    print(f"Dataset type : {dataset.upper()}")

    if dataset == "kdd":
        print(f"Train CSV    : {args.train}")
        print(f"Test CSV     : {args.test}")
        test_cases = generate_kdd(
            args.train, args.test,
            n_normal=args.normal,
            n_attack=args.attack,
            seed=args.seed,
        )
    else:
        if args.csv is None:
            parser.error("--csv is required for UNSW / generic datasets.")
        tag = "UNSW" if dataset == "unsw" else Path(args.csv).stem
        print(f"CSV          : {args.csv}")
        test_cases = generate_generic(
            args.csv,
            n_normal=args.normal,
            n_attack=args.attack,
            seed=args.seed,
            dataset_tag=tag,
        )

    output = {"test_cases": test_cases}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=4)

    n_normal = sum(1 for t in test_cases if t["expected"] == "normal")
    n_attack_out = sum(1 for t in test_cases if t["expected"] == "intrusion")
    print(f"\nOutput       : {out_path}")
    print(f"Total cases  : {len(test_cases)}")
    print(f"  Normal     : {n_normal}")
    print(f"  Intrusion  : {n_attack_out}")
    print("\nAttack types covered:")
    seen = set()
    for t in test_cases:
        if t["expected"] == "intrusion":
            name = t["name"]
            base = name.rsplit(" #", 1)[0]
            if base not in seen:
                seen.add(base)
                print(f"  - {base}")


if __name__ == "__main__":
    main()
