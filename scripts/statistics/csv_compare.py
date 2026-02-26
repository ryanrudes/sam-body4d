import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


def compare_two_score_csvs_by_metric(
    csv1_path: str,
    csv2_path: str,
    out_dir: str = "compare_by_metric",
    fig_w: int = 16,
    row_h: float = 2.2,
    mark_max_gain: bool = False,
):
    """
    Two CSVs, no header. Each row:
      col0 = seq name (string)
      col1 = metric name (string)
      col2 = metric mean (float)     # optional/NaN tolerated
      col3.. = per-frame scores (float)

    Each seq appears in 4 rows (4 different metrics). (Not strictly enforced; we group by metric anyway.)

    We generate big figures (one per metric), each stacking all seqs vertically:
      csv1 = red line
      csv2 = blue line

    If mark_max_gain=True:
      For each subplot (seq), mark the index where (b - a) is maximized among finite pairs.

    Output filename: <metric_name>.jpg under out_dir.
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
        """
        return dict:
          data[metric_name][seq_name] = {"mean": float, "vals": [float, ...]}
        """
        data = {}
        with open(p, "r", newline="") as f:
            reader = csv.reader(f)
            for r in reader:
                if not r:
                    continue
                if len(r) < 4:
                    # must have at least seq, metric, mean, one score (or empty score)
                    continue

                seq = (r[0] or "").strip()
                metric = (r[1] or "").strip()
                mean_v = _to_float(r[2])

                vals = []
                for x in r[3:]:
                    x = (x or "").strip()
                    if x == "":
                        vals.append(float("nan"))
                    else:
                        try:
                            vals.append(float(x))
                        except ValueError:
                            vals.append(float("nan"))

                if metric not in data:
                    data[metric] = {}
                data[metric][seq] = {"mean": mean_v, "vals": vals}
        return data

    def _sanitize_filename(name: str) -> str:
        name = name.strip()
        name = re.sub(r"[\\/:*?\"<>|]+", "_", name)  # windows-illegal chars
        name = re.sub(r"\s+", " ", name).strip()
        return name or "metric"

    d1 = _read_csv(csv1_path)
    d2 = _read_csv(csv2_path)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # metrics to generate = intersection (safer for comparison)
    metrics = sorted(set(d1.keys()) & set(d2.keys()))
    if not metrics:
        raise ValueError("No overlapping metric names found between the two CSVs.")

    base1 = Path(csv1_path).name
    base2 = Path(csv2_path).name

    for metric in metrics:
        s1 = d1.get(metric, {})
        s2 = d2.get(metric, {})

        # seq intersection for this metric
        seqs = [k for k in s1.keys() if k in s2]
        if not seqs:
            continue
        seqs = sorted(seqs)

        nrows = len(seqs)
        fig_h = max(row_h * nrows, 3.0)
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(fig_w, fig_h), sharex=False)
        if nrows == 1:
            axes = [axes]

        for ax, seq in zip(axes, seqs):
            a = s1[seq]["vals"]
            b = s2[seq]["vals"]
            n = min(len(a), len(b))
            a = a[:n]
            b = b[:n]
            x = list(range(n))

            ax.plot(x, a, color="red", linewidth=1.5, label=base1)
            ax.plot(x, b, color="blue", linewidth=1.5, label=base2)

            # mark index where (b - a) is maximized
            if mark_max_gain:
                best_idx = None
                best_val = -float("inf")
                for i, (ai, bi) in enumerate(zip(a, b)):
                    if _finite(ai) and _finite(bi):
                        d = bi - ai
                        if d > best_val:
                            best_val = d
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
                        f"max(b-a)={best_val:.4f} @ {best_idx}",
                        transform=ax.transAxes,
                        fontsize=9,
                        horizontalalignment="right",
                        verticalalignment="top",
                    )

            # show means if available
            m1 = s1[seq].get("mean", float("nan"))
            m2 = s2[seq].get("mean", float("nan"))
            mean_txt = []
            if _finite(m1):
                mean_txt.append(f"{base1} mean={m1:.4f}")
            if _finite(m2):
                mean_txt.append(f"{base2} mean={m2:.4f}")
            if mean_txt:
                ax.text(
                    0.01,
                    0.92,
                    " | ".join(mean_txt),
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


# Example:
compare_two_score_csvs_by_metric(
    # "/home/hmq/projects/hmr/sam-body4d/csv_results/3DPW-box-kp-dp.csv", 
    "/home/hmq/projects/hmr/sam-body4d/csv_results/3DPW-box-kp-post.csv", 
    "/home/hmq/projects/hmr/sam-body4d/csv_results/3DPW-mask-kp-text-new.csv", 
    out_dir="/home/hmq/projects/hmr/sam-body4d/figs",
    mark_max_gain=True
)
