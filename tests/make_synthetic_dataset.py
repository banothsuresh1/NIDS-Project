"""Generates a small synthetic CIC-IDS2017-shaped dataset (5 daily CSVs) for
smoke-testing the pipeline without the real ~2.8M-row dataset. Not part of
the methodology -- purely a development/testing helper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

COLUMNS = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port",
    "Protocol", "Timestamp", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label",
]

DAY_LABEL_MIX = {
    "Monday": {"BENIGN": 1.0},
    "Tuesday": {"BENIGN": 0.7, "FTP-Patator": 0.15, "SSH-Patator": 0.15},
    "Wednesday": {"BENIGN": 0.6, "DoS Hulk": 0.2, "DoS GoldenEye": 0.1, "DoS slowloris": 0.05, "DoS Slowhttptest": 0.05},
    "Thursday": {"BENIGN": 0.85, "Web Attack - Brute Force": 0.05, "Web Attack - XSS": 0.03,
                 "Web Attack - Sql Injection": 0.02, "Infiltration": 0.03, "Heartbleed": 0.02},
    "Friday": {"BENIGN": 0.6, "PortScan": 0.2, "DDoS": 0.15, "Bot": 0.05},
}

DAY_DATE = {"Monday": "3/7/2017", "Tuesday": "4/7/2017", "Wednesday": "5/7/2017",
            "Thursday": "6/7/2017", "Friday": "7/7/2017"}


def _rand_ip(rng):
    return f"192.168.{rng.integers(1, 5)}.{rng.integers(1, 254)}"


def make_flow(rng, label: str, t0_seconds: int):
    is_attack = label != "BENIGN"
    src_ip = _rand_ip(rng)
    dst_ip = f"205.174.{rng.integers(1,255)}.{rng.integers(1,255)}" if is_attack else _rand_ip(rng)
    port_map = {"FTP-Patator": 21, "SSH-Patator": 22, "Web Attack - Brute Force": 80,
                "Web Attack - XSS": 80, "Web Attack - Sql Injection": 80, "Heartbleed": 443}
    dst_port = port_map.get(label, int(rng.integers(1, 65535)))
    proto = 6
    duration_us = int(rng.integers(1000, 5_000_000))
    fwd_pkts = int(rng.integers(1, 20)) if not is_attack else int(rng.integers(1, 250))
    bwd_pkts = int(rng.integers(0, 20))
    fwd_bytes = fwd_pkts * int(rng.integers(40, 1500))
    bwd_bytes = bwd_pkts * int(rng.integers(40, 1500))
    syn = 1 if rng.random() < 0.8 else 0
    ack = 1 if bwd_pkts > 0 else 0
    rst = 1 if (is_attack and rng.random() < 0.4) else 0
    fin = 1 if (not is_attack and rng.random() < 0.3) else 0
    pkt_rate = (fwd_pkts + bwd_pkts) / max(duration_us / 1e6, 1e-3)

    sec = t0_seconds % 86400
    hh, mm, ss = sec // 3600, (sec % 3600) // 60, sec % 60
    ts = f"{DAY_DATE_CURRENT} {hh:02d}:{mm:02d}:{ss:02d}"

    row = {
        "Flow ID": f"{src_ip}-{dst_ip}-{dst_port}", "Source IP": src_ip, "Source Port": int(rng.integers(1024, 65535)),
        "Destination IP": dst_ip, "Destination Port": dst_port, "Protocol": proto, "Timestamp": ts,
        "Flow Duration": duration_us, "Total Fwd Packets": fwd_pkts, "Total Backward Packets": bwd_pkts,
        "Total Length of Fwd Packets": fwd_bytes, "Total Length of Bwd Packets": bwd_bytes,
        "Fwd Packet Length Max": fwd_bytes, "Fwd Packet Length Min": 0, "Fwd Packet Length Mean": fwd_bytes / max(fwd_pkts, 1),
        "Fwd Packet Length Std": 1.0, "Bwd Packet Length Max": bwd_bytes, "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": bwd_bytes / max(bwd_pkts, 1), "Bwd Packet Length Std": 1.0,
        "Flow Bytes/s": (fwd_bytes + bwd_bytes) / max(duration_us / 1e6, 1e-3), "Flow Packets/s": pkt_rate,
        "Flow IAT Mean": 100.0, "Flow IAT Std": 10.0, "Flow IAT Max": 500.0, "Flow IAT Min": 1.0,
        "Fwd IAT Total": 100.0, "Fwd IAT Mean": 10.0, "Fwd IAT Std": 1.0, "Fwd IAT Max": 50.0, "Fwd IAT Min": 1.0,
        "Bwd IAT Total": 100.0, "Bwd IAT Mean": 10.0, "Bwd IAT Std": 1.0, "Bwd IAT Max": 50.0, "Bwd IAT Min": 1.0,
        "Fwd PSH Flags": 0, "Bwd PSH Flags": 0, "Fwd URG Flags": 0, "Bwd URG Flags": 0,
        "Fwd Header Length": 20 * fwd_pkts, "Bwd Header Length": 20 * bwd_pkts,
        "Fwd Packets/s": fwd_pkts / max(duration_us / 1e6, 1e-3), "Bwd Packets/s": bwd_pkts / max(duration_us / 1e6, 1e-3),
        "Min Packet Length": 0, "Max Packet Length": max(fwd_bytes, bwd_bytes, 1),
        "Packet Length Mean": (fwd_bytes + bwd_bytes) / max(fwd_pkts + bwd_pkts, 1),
        "Packet Length Std": 1.0, "Packet Length Variance": 1.0,
        "FIN Flag Count": fin, "SYN Flag Count": syn, "RST Flag Count": rst, "PSH Flag Count": int(rng.integers(0, 2)),
        "ACK Flag Count": ack, "URG Flag Count": 0, "CWE Flag Count": 0, "ECE Flag Count": 0,
        "Down/Up Ratio": bwd_pkts / max(fwd_pkts, 1), "Average Packet Size": (fwd_bytes + bwd_bytes) / max(fwd_pkts + bwd_pkts, 1),
        "Avg Fwd Segment Size": fwd_bytes / max(fwd_pkts, 1), "Avg Bwd Segment Size": bwd_bytes / max(bwd_pkts, 1),
        "Subflow Fwd Packets": fwd_pkts, "Subflow Fwd Bytes": fwd_bytes, "Subflow Bwd Packets": bwd_pkts, "Subflow Bwd Bytes": bwd_bytes,
        "Init_Win_bytes_forward": 8192, "Init_Win_bytes_backward": 8192, "act_data_pkt_fwd": fwd_pkts, "min_seg_size_forward": 20,
        "Active Mean": 1.0, "Active Std": 0.1, "Active Max": 2.0, "Active Min": 0.5,
        "Idle Mean": 1.0, "Idle Std": 0.1, "Idle Max": 2.0, "Idle Min": 0.5, "Label": label,
    }
    return row


BRUTE_FORCE_LABELS = {"FTP-Patator", "SSH-Patator", "Web Attack - Brute Force"}


def make_campaign(rng, label: str, t0_seconds: int) -> list:
    """A burst of 3-8 flows sharing ONE 5-tuple, close together in time --
    the multi-flow 'session' shape (SSH brute-force worked example) that a
    single random independent flow per label never produces."""
    src_ip = _rand_ip(rng)
    dst_ip = f"205.174.{rng.integers(1,255)}.{rng.integers(1,255)}"
    src_port = int(rng.integers(1024, 65535))
    dst_port = {"FTP-Patator": 21, "SSH-Patator": 22}.get(label, 80)
    n_flows = int(rng.integers(3, 9))
    rows = []
    t = t0_seconds
    for i in range(n_flows):
        is_last = i == n_flows - 1
        row = make_flow(rng, label, t)
        # pin the campaign to one shared 5-tuple, close in time (< tau=60s)
        row["Source IP"], row["Destination IP"] = src_ip, dst_ip
        row["Source Port"], row["Destination Port"] = src_port, dst_port
        row["Total Backward Packets"] = 0 if not is_last else int(rng.integers(1, 5))
        row["RST Flag Count"] = 0 if is_last else 1
        row["ACK Flag Count"] = 1 if is_last else 0
        rows.append(row)
        t += int(rng.integers(3, 8))
    return rows


def main(out_dir: str, rows_per_day: int = 400):
    global DAY_DATE_CURRENT
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for day, mix in DAY_LABEL_MIX.items():
        DAY_DATE_CURRENT = DAY_DATE[day]
        rng = np.random.default_rng(hash(day) % (2**32))
        labels = list(mix.keys())
        probs = np.array(list(mix.values()))
        probs = probs / probs.sum()
        chosen = rng.choice(labels, size=rows_per_day, p=probs)
        rows = []
        t = 28800  # 08:00
        for lbl in chosen:
            t += int(rng.integers(0, 3))
            if lbl in BRUTE_FORCE_LABELS and rng.random() < 0.5:
                rows.extend(make_campaign(rng, lbl, t))
                t += 20
            else:
                rows.append(make_flow(rng, lbl, t))
        df = pd.DataFrame(rows, columns=COLUMNS)
        fname = out / f"{day}-WorkingHours.pcap_ISCX.csv"
        df.to_csv(fname, index=False)
        print(f"wrote {fname} ({len(df)} rows)")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/synthetic_cicids2017"
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    main(out_dir, rows)
