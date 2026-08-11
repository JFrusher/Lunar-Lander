"""Generate the 1280x640 GitHub social-preview card.

A raw chart doesn't work here: social cards render ~600px wide in Slack,
LinkedIn and X, where axis labels are illegible. This pairs large text with
the chart as a supporting visual.

    python scripts/make_social_preview.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import image as mpimg

# Same palette as the charts (see time_penalty.py).
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_MUTED = "#575652"
_ACCENT = "#eb6834"

OUTPUT = Path("docs/media/social_preview.png")
CHART = Path("docs/media/time_penalty_tradeoff.png")


def main() -> None:
    # 1280x640 at 100 dpi — GitHub's recommended social-preview size.
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=_SURFACE)

    fig.text(0.06, 0.87, "Lunar Lander Lab", size=44, weight="bold",
             color=_INK_PRIMARY, va="top")
    fig.text(0.06, 0.755,
             "Reward shaping on LunarLander-v3 — and what it took to trust the result",
             size=19, color=_INK_MUTED, va="top")

    # Stat and caption on separate baselines — side-by-side collides, since
    # the "22%" glyph box is wider than its nominal extent.
    fig.text(0.06, 0.585, "22% faster landings", size=33, weight="bold",
             color=_ACCENT, va="top")
    fig.text(0.06, 0.445, "at no cost to reward or success rate",
             size=18, color=_INK_PRIMARY, va="top")

    fig.text(0.06, 0.275,
             "PPO · classical control · Latin Hypercube search\n"
             "multi-seed error bars · held-out validation",
             size=15, color=_INK_MUTED, va="top", linespacing=1.6)

    fig.text(0.06, 0.08, "github.com/JFrusher/Lunar-Lander",
             size=14, color=_INK_MUTED, va="bottom")

    # Right panel: crop off the chart's own suptitle/legend, which are
    # illegible at card size and duplicate the headline above.
    chart = mpimg.imread(CHART)
    chart = chart[int(chart.shape[0] * 0.22):, :]
    ax = fig.add_axes([0.50, 0.08, 0.47, 0.80])
    ax.imshow(chart)
    ax.axis("off")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=100, facecolor=_SURFACE)
    plt.close(fig)

    size_kb = OUTPUT.stat().st_size / 1024
    print(f"{OUTPUT} written ({size_kb:.0f} KB) — GitHub limit is 1 MB")


if __name__ == "__main__":
    main()
