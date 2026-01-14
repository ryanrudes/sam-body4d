import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def compare_two_score_csvs(csv1_path: str, csv2_path: str, out_path: str = "compare.jpg"):
    """
    Two CSVs, no header. Each row:
      col0 = seq name (string)
      col1.. = per-frame scores (float)

    For each row (matched by seq name), plot:
      csv1 = red line
      csv2 = blue line
    Mark indices where the two curves "cross" (intersection) on the x-axis (frame index).
    Stack all rows vertically into one big figure and save to out_path.
    """

    def _read_csv(p: str):
        rows = []
        with open(p, "r", newline="") as f:
            reader = csv.reader(f)
            for r in reader:
                if not r:
                    continue
                name = r[0]
                vals = []
                for x in r[1:]:
                    x = (x or "").strip()
                    if x == "":
                        vals.append(float("nan"))
                    else:
                        try:
                            vals.append(float(x))
                        except ValueError:
                            vals.append(float("nan"))
                rows.append((name, vals))
        return rows

    def _cross_indices(a, b):
        """
        Return indices i (0-based) where a and b cross between i and i+1,
        i.e., sign(a-b) changes. Also handle exact hits (diff == 0).
        """
        n = min(len(a), len(b))
        idxs = set()

        def finite(x):
            return x is not None and isinstance(x, (int, float)) and not math.isnan(x) and math.isfinite(x)

        for i in range(n):
            if finite(a[i]) and finite(b[i]) and (a[i] - b[i]) == 0.0:
                idxs.add(i)

        for i in range(n - 1):
            if not (finite(a[i]) and finite(b[i]) and finite(a[i + 1]) and finite(b[i + 1])):
                continue
            d0 = a[i] - b[i]
            d1 = a[i + 1] - b[i + 1]
            # strict sign change
            if d0 == 0.0 or d1 == 0.0:
                continue
            if (d0 > 0 and d1 < 0) or (d0 < 0 and d1 > 0):
                idxs.add(i + 1)  # mark the later index for visibility
        return sorted(idxs)

    rows1 = _read_csv(csv1_path)
    rows2 = _read_csv(csv2_path)

    m2 = {name: vals for name, vals in rows2}
    names = [name for name, _ in rows1 if name in m2]

    if not names:
        raise ValueError("No overlapping seq names found between the two CSVs.")

    nrows = len(names)
    # big vertical stack; tune height per row
    fig_h = max(2.2 * nrows, 3.0)
    fig_w = 16
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(fig_w, fig_h), sharex=False)
    if nrows == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        a = dict(rows1).get(name)
        b = m2.get(name)
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]

        x = list(range(n))
        ax.plot(x, a, color="red", linewidth=1.5, label=Path(csv1_path).name)
        ax.plot(x, b, color="blue", linewidth=1.5, label=Path(csv2_path).name)

        crosses = _cross_indices(a, b)
        # mark crossings on the plot (use mid y for a simple marker line)
        if crosses:
            for ci in crosses:
                ax.axvline(ci, linestyle="--", linewidth=1.0, alpha=0.6)
            # also annotate once with indices (avoid clutter)
            ax.text(
                0.01, 0.92,
                f"cross idx: {crosses[:30]}{' ...' if len(crosses) > 30 else ''}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
            )

        ax.set_title(name, fontsize=11)
        ax.grid(True, alpha=0.25)

    # legend only once (top subplot)
    axes[0].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Example:

