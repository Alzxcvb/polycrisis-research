"""
Simple, explainable visualizations for non-technical audiences.

Complements the network maps in 04_network_viz.py. Network graphs are hard
to read cold — these four charts each answer ONE question at a glance:

    1. simple_timeline.png     — How fast is each field growing?
    2. simple_top_topics.png   — What does each field talk about?
    3. simple_venn.png         — How much do the two fields overlap?
    4. simple_wordclouds.png   — Visual gist of each field

All outputs are single-page PNGs, 200 DPI, suitable for Substack or slides.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn2
from wordcloud import WordCloud

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# Generic OpenAlex concept tags + biology-of-growth noise that obscures
# the actual research content. These are high-level discipline labels or
# noise from cell/plant growth papers that slipped into the LtG corpus.
STOPWORDS = {
    # generic discipline / meta tags
    "political science", "sociology", "computer science", "mathematics",
    "physics", "psychology", "engineering", "chemistry", "biology",
    "art", "humanities", "geography", "history", "philosophy", "medicine",
    "law", "materials science", "environmental science", "business",
    "public relations", "library science", "social science",
    # biology-of-growth noise (not Meadows LtG)
    "botany", "agronomy", "horticulture", "cell biology", "biochemistry",
    "cancer research", "gene", "genetics", "nutrient", "microbiology",
    "biophysics", "biotechnology", "molecular biology", "endocrinology",
    # meta / disambiguation tags
    "context (archaeology)", "content (measure theory)",
    "action (physics)", "quantum entanglement",
    "resilience (materials science)", "government (linguistics)",
    "crisis response", "epistemology",
}

POLY_COLOR = "#e85d4c"   # warm red
LTG_COLOR = "#4a90d9"    # blue
SHARED_COLOR = "#9b6bb7"  # purple (overlap)


def load_clean_freqs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["term"])
    df["term"] = df["term"].astype(str).str.strip().str.lower()
    df = df[df["term"] != ""]
    df = df[~df["term"].isin(STOPWORDS)]
    df = df[~df["term"].str.contains(r"\(", regex=True)]  # drop "term (disambig)"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Publication timeline — how fast each field is growing
# ---------------------------------------------------------------------------
def make_timeline() -> None:
    poly = pd.read_csv(DATA / "polycrisis_papers.csv")
    ltg_path = DATA / "ltg_refined_papers.csv"
    if not ltg_path.exists():
        ltg_path = DATA / "ltg_papers.csv"
    ltg = pd.read_csv(ltg_path)

    def yearly(df: pd.DataFrame) -> pd.Series:
        s = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
        s = s[(s >= 2000) & (s <= 2025)]
        return s.value_counts().sort_index()

    poly_yr, ltg_yr = yearly(poly), yearly(ltg)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(poly_yr.index, poly_yr.values, marker="o", linewidth=2.5,
            color=POLY_COLOR, label=f"Polycrisis papers (n={len(poly):,})")
    ax.plot(ltg_yr.index, ltg_yr.values, marker="s", linewidth=2.5,
            color=LTG_COLOR, label=f"Limits to Growth papers (n={len(ltg):,})")

    ax.set_title("Two fields, very different growth curves",
                 fontsize=17, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Papers published", fontsize=12)
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate(
        "Polycrisis explodes\nafter 2022 (Tooze, WEF)",
        xy=(2024, poly_yr.get(2024, 300)),
        xytext=(2015, max(poly_yr.max(), ltg_yr.max()) * 0.7),
        fontsize=10, color=POLY_COLOR,
        arrowprops=dict(arrowstyle="->", color=POLY_COLOR, alpha=0.6),
    )

    fig.tight_layout()
    out = FIGS / "simple_timeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 2. Top topics — side-by-side bar chart, what each field actually talks about
# ---------------------------------------------------------------------------
def make_top_topics(n: int = 15) -> None:
    poly = load_clean_freqs(DATA / "polycrisis_keyword_cooccurrence_frequencies.csv").head(n)
    ltg = load_clean_freqs(DATA / "ltg_keyword_cooccurrence_frequencies.csv").head(n)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 8))

    for ax, df, color, title in [
        (ax_l, poly, POLY_COLOR, "What POLYCRISIS research talks about"),
        (ax_r, ltg, LTG_COLOR, "What LIMITS TO GROWTH research talks about"),
    ]:
        df = df[::-1]  # so biggest sits on top
        ax.barh(df["term"], df["count"], color=color, alpha=0.85)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Number of papers mentioning this topic", fontsize=10)
        ax.tick_params(axis="y", labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Top 15 topics in each field",
                 fontsize=16, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = FIGS / "simple_top_topics.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 3. Venn diagram — how much do the two fields overlap?
# ---------------------------------------------------------------------------
def make_venn(top_n: int = 50) -> None:
    poly = set(load_clean_freqs(DATA / "polycrisis_keyword_cooccurrence_frequencies.csv")
               .head(top_n)["term"])
    ltg = set(load_clean_freqs(DATA / "ltg_keyword_cooccurrence_frequencies.csv")
              .head(top_n)["term"])
    shared = poly & ltg
    only_poly = poly - ltg
    only_ltg = ltg - poly

    fig, ax = plt.subplots(figsize=(11, 8))
    v = venn2(
        subsets=(len(only_poly), len(only_ltg), len(shared)),
        set_labels=("Polycrisis", "Limits to Growth"),
        set_colors=(POLY_COLOR, LTG_COLOR),
        alpha=0.55,
        ax=ax,
    )
    for lbl in v.set_labels:
        if lbl:
            lbl.set_fontsize(16)
            lbl.set_fontweight("bold")
    for lbl in v.subset_labels:
        if lbl:
            lbl.set_fontsize(14)

    # Show the actual shared words below the diagram
    shared_sorted = sorted(shared)
    shared_str = ", ".join(shared_sorted) if shared_sorted else "(none)"
    caption = (
        f"Of the top {top_n} topics in each field, "
        f"{len(shared)} appear in both.\n\n"
        f"Shared: {shared_str}"
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=11,
             wrap=True, style="italic", color="#444")

    ax.set_title("Do these two research fields talk about the same things?",
                 fontsize=16, fontweight="bold", pad=15)
    fig.tight_layout()
    out = FIGS / "simple_venn.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT)}  (shared={len(shared)})")


# ---------------------------------------------------------------------------
# 4. Word clouds — instant visual gist of each field
# ---------------------------------------------------------------------------
def make_wordclouds(top_n: int = 100) -> None:
    poly = load_clean_freqs(DATA / "polycrisis_keyword_cooccurrence_frequencies.csv").head(top_n)
    ltg = load_clean_freqs(DATA / "ltg_keyword_cooccurrence_frequencies.csv").head(top_n)

    def freqs(df: pd.DataFrame) -> dict[str, int]:
        return dict(zip(df["term"], df["count"]))

    def cloud(frequencies: dict, color: str) -> WordCloud:
        return WordCloud(
            width=1000, height=700, background_color="white",
            colormap=None, color_func=lambda *a, **kw: color,
            prefer_horizontal=0.9, relative_scaling=0.55,
            max_words=top_n, min_font_size=10,
        ).generate_from_frequencies(frequencies)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7))
    ax_l.imshow(cloud(freqs(poly), POLY_COLOR), interpolation="bilinear")
    ax_l.set_title("Polycrisis", fontsize=16, fontweight="bold",
                   color=POLY_COLOR, pad=10)
    ax_l.axis("off")
    ax_r.imshow(cloud(freqs(ltg), LTG_COLOR), interpolation="bilinear")
    ax_r.set_title("Limits to Growth", fontsize=16, fontweight="bold",
                   color=LTG_COLOR, pad=10)
    ax_r.axis("off")

    fig.suptitle("Word size = how often the topic appears in the field",
                 fontsize=13, y=0.98, style="italic", color="#555")
    fig.tight_layout()
    out = FIGS / "simple_wordclouds.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT)}")


def main() -> None:
    print("Building simple visualizations...")
    make_timeline()
    make_top_topics()
    make_venn()
    make_wordclouds()
    print("\nDone. Open the four simple_*.png files in figures/.")


if __name__ == "__main__":
    main()
