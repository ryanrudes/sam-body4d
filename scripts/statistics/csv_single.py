import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


def plot_one_score_csv_by_metric(
    csv_path: str,
    out_dir: str = "plot_by_metric",
    fig_w: int = 16,
    row_h: float = 2.2,
):
    """
    One CSV, no header. Each row:
      col0 = seq name (string)
      col1 = metric name (string)
      col2 = metric mean (float)
      col3.. = per-frame scores (float)

    For each metric:
      - stack all sequences vertically
      - plot single curve
      - mark argmax index (ignoring NaN)
      - show argmax index + max value
    """

    def _to_float(x: str) -> float:
        x = (x or "").strip()
        if x == "":
            return float("nan")
        try:
            return float(x)
        except ValueError:
            return float("nan")

    def _finite(x: float) -> bool:
        return isinstance(x, (int, float)) and not math.isnan(x) and math.isfinite(x)

    def _read_csv(p: str):
        data = {}
        with open(p, "r", newline="") as f:
            reader = csv.reader(f)
            for r in reader:
                if not r or len(r) < 4:
                    continue

                seq = (r[0] or "").strip()
                metric = (r[1] or "").strip()
                mean_v = _to_float(r[2])

                vals = []
                for x in r[3:]:
                    vals.append(_to_float(x))

                if metric not in data:
                    data[metric] = {}
                data[metric][seq] = {"mean": mean_v, "vals": vals}
        return data

    def _sanitize_filename(name: str) -> str:
        name = name.strip()
        name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name or "metric"

    d = _read_csv(csv_path)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    metrics = sorted(d.keys())
    if not metrics:
        raise ValueError("No metric rows found in the CSV.")

    base = Path(csv_path).name

    for metric in metrics:
        s = d.get(metric, {})
        seqs = sorted(s.keys())
        if not seqs:
            continue

        nrows = len(seqs)
        fig_h = max(row_h * nrows, 3.0)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            figsize=(fig_w, fig_h),
            sharex=False
        )

        if nrows == 1:
            axes = [axes]

        for ax, seq in zip(axes, seqs):
            vals = s[seq]["vals"]
            n = len(vals)
            if n == 0:
                continue

            x = list(range(n))
            ax.plot(x, vals, color="black", linewidth=1.5, label=base)

            # -------- argmax (ignore NaN) --------
            best_idx = None
            best_val = -float("inf")

            for i, v in enumerate(vals):
                if _finite(v) and v > best_val:
                    best_val = v
                    best_idx = i

            if best_idx is not None:
                ax.axvline(
                    best_idx,
                    linestyle=":",
                    linewidth=1.2,
                    color="black",
                    alpha=0.6,
                )

                ax.text(
                    0.99,
                    0.92,
                    f"argmax={best_idx} ({best_val:.4f})",
                    transform=ax.transAxes,
                    fontsize=9,
                    horizontalalignment="right",
                    verticalalignment="top",
                )

                if best_idx > 0:
                    ax.set_xticks([0, best_idx])
                else:
                    ax.set_xticks([0])

            # -------- mean (if available) --------
            mean_v = s[seq].get("mean", float("nan"))
            if _finite(mean_v):
                ax.text(
                    0.01,
                    0.92,
                    f"mean={mean_v:.4f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                )

            ax.set_title(seq, fontsize=11)
            ax.grid(True, alpha=0.25)

        axes[0].legend(loc="upper right", fontsize=9)
        fig.suptitle(metric, fontsize=14, y=1.002)
        fig.tight_layout()

        out_path = out_dir_p / f"{_sanitize_filename(metric)}.jpg"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Done. Figures saved to: {out_dir_p.resolve()}")


# example
plot_one_score_csv_by_metric(
    "/root/projects/sam-body4d/rich-mask-kp-debug.csv",
    out_dir="/root/projects/sam-body4d/figs_onecsv",
)