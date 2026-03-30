"""
Bibliometric Analysis — Co-occurrence, Citation Networks, Trend Mapping
Project: Polycrisis & Limits to Growth Research
Author: Alex Coffman

What this does:
    Takes the CSVs from 01_fetch_openalex.py and produces:
    1. Publication trend charts (papers per year)
    2. Keyword co-occurrence data (for VOSviewer import)
    3. Top author/institution rankings
    4. Journal distribution
    5. Citation analysis (most influential papers)

Run AFTER 01_fetch_openalex.py:
    pip install pandas matplotlib seaborn plotly
    python 02_bibliometric_analysis.py

Output:
    data/keyword_cooccurrence.csv  (import into VOSviewer)
    data/author_network.csv
    figures/ (PNG charts)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ── Load Data ─────────────────────────────────────────────────────────────────

def load_data():
    poly = pd.read_csv(DATA_DIR / "polycrisis_papers.csv")
    ltg = pd.read_csv(DATA_DIR / "ltg_papers.csv")
    poly["query"] = "polycrisis"
    ltg["query"] = "limits_to_growth"
    return poly, ltg


# ── 1. Publication Trend Chart ────────────────────────────────────────────────

def plot_publication_trends(poly: pd.DataFrame, ltg: pd.DataFrame):
    """
    The core trend chart: papers per year for both search terms.
    This is Figure 1 of the eventual paper.
    """
    poly_by_year = poly.groupby("year").size()
    ltg_by_year = ltg.groupby("year").size()

    # Focus on 2000-present for clarity
    years = range(2000, 2026)
    poly_counts = [poly_by_year.get(y, 0) for y in years]
    ltg_counts = [ltg_by_year.get(y, 0) for y in years]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Academic Publication Trends: Polycrisis & Limits to Growth",
                 fontsize=14, fontweight="bold")

    # Polycrisis
    ax1.bar(years, poly_counts, color="#e74c3c", alpha=0.8, label="Polycrisis")
    ax1.set_ylabel("Papers per Year")
    ax1.set_title("'Polycrisis' in Title/Abstract (OpenAlex)")
    ax1.axvline(x=2022, color="#333", linestyle="--", alpha=0.5, label="Tooze + Cascade (2022)")
    ax1.legend()
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # LtG
    ax2.bar(years, ltg_counts, color="#2980b9", alpha=0.8, label="Limits to Growth")
    ax2.set_ylabel("Papers per Year")
    ax2.set_xlabel("Year")
    ax2.set_title("'Limits to Growth' (exact phrase) in Title/Abstract (OpenAlex)")
    ax2.axvline(x=2021, color="#333", linestyle="--", alpha=0.5, label="Herrington KPMG paper (2021)")
    ax2.legend()
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out = FIG_DIR / "01_publication_trends.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── 2. Keyword Co-occurrence (for VOSviewer) ──────────────────────────────────

def build_keyword_cooccurrence(df: pd.DataFrame, output_path: Path, min_count: int = 3):
    """
    Build a keyword co-occurrence matrix.
    Import the output CSV into VOSviewer to generate a visual network map.

    VOSviewer: https://www.vosviewer.com/ (free download)
    Import via: Create map → Based on bibliographic data → VOSviewer format
    """
    # Parse keywords per paper
    cooccurrence = defaultdict(int)
    keyword_counts = Counter()

    for _, row in df.iterrows():
        keywords_raw = str(row.get("keywords", "") or "")
        concepts_raw = str(row.get("concepts", "") or "")

        # Combine keywords and concepts
        terms = set()
        for kw in keywords_raw.split(";"):
            kw = kw.strip().lower()
            if kw and len(kw) > 2:
                terms.add(kw)
        for concept in concepts_raw.split(";"):
            concept = concept.strip().lower()
            if concept and len(concept) > 2:
                terms.add(concept)

        # Count individual terms
        for term in terms:
            keyword_counts[term] += 1

        # Count pairs (co-occurrence)
        for t1, t2 in combinations(sorted(terms), 2):
            cooccurrence[(t1, t2)] += 1

    # Filter to terms appearing >= min_count times
    valid_terms = {term for term, count in keyword_counts.items() if count >= min_count}

    # Build dataframe
    rows = []
    for (t1, t2), count in cooccurrence.items():
        if t1 in valid_terms and t2 in valid_terms and count >= 2:
            rows.append({"term1": t1, "term2": t2, "cooccurrence": count})

    co_df = pd.DataFrame(rows).sort_values("cooccurrence", ascending=False)
    co_df.to_csv(output_path, index=False)
    print(f"  Keyword co-occurrence saved: {output_path} ({len(co_df):,} pairs)")

    # Also save term frequency
    freq_df = pd.DataFrame(keyword_counts.most_common(100), columns=["term", "count"])
    freq_path = output_path.parent / (output_path.stem + "_frequencies.csv")
    freq_df.to_csv(freq_path, index=False)
    print(f"  Term frequencies saved: {freq_path}")

    return co_df


# ── 3. Top Authors & Institutions ─────────────────────────────────────────────

def analyze_authors_institutions(df: pd.DataFrame, label: str):
    print(f"\n── {label} ──")

    # Top authors (rough — OpenAlex format is "Name; Name; ...")
    all_authors = []
    for authors_str in df["authors"].dropna():
        for author in authors_str.split(";"):
            author = author.strip().split("(")[0].strip()  # remove affiliation if inline
            if author and "et al" not in author:
                all_authors.append(author)

    top_authors = Counter(all_authors).most_common(15)
    print("\nTop 15 Authors:")
    for author, count in top_authors:
        print(f"  {count:4d}  {author}")

    # Top institutions
    all_institutions = []
    for inst_str in df["institutions"].dropna():
        for inst in inst_str.split(";"):
            inst = inst.strip()
            if inst:
                all_institutions.append(inst)

    top_institutions = Counter(all_institutions).most_common(15)
    print("\nTop 15 Institutions:")
    for inst, count in top_institutions:
        print(f"  {count:4d}  {inst}")


# ── 4. Most Cited Papers ──────────────────────────────────────────────────────

def top_cited_papers(df: pd.DataFrame, label: str, n: int = 20):
    print(f"\n── Most Cited Papers: {label} ──")
    top = df.nlargest(n, "cited_by_count")[["year", "title", "authors", "journal", "cited_by_count", "doi"]]
    for _, row in top.iterrows():
        print(f"\n  [{row['year']}] {row['title'][:80]}")
        print(f"    Authors: {str(row['authors'])[:60]}")
        print(f"    Journal: {row['journal']}")
        print(f"    Cited: {row['cited_by_count']:,}  DOI: {row['doi']}")


# ── 5. Journal Distribution ───────────────────────────────────────────────────

def plot_journal_distribution(df: pd.DataFrame, label: str, filename: str):
    top_journals = (
        df[df["journal"].notna() & (df["journal"] != "")]
        .groupby("journal")
        .size()
        .sort_values(ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    top_journals.sort_values().plot(kind="barh", ax=ax, color="#2ecc71", alpha=0.8)
    ax.set_title(f"Top 15 Journals — {label}", fontweight="bold")
    ax.set_xlabel("Number of Papers")
    plt.tight_layout()
    out = FIG_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    try:
        poly, ltg = load_data()
    except FileNotFoundError:
        print("ERROR: Run 01_fetch_openalex.py first to download the data.")
        return

    print(f"  Polycrisis papers: {len(poly):,}")
    print(f"  LtG papers: {len(ltg):,}")

    print("\n1. Publication trend charts...")
    plot_publication_trends(poly, ltg)

    print("\n2. Keyword co-occurrence networks...")
    build_keyword_cooccurrence(poly, DATA_DIR / "polycrisis_keyword_cooccurrence.csv")
    build_keyword_cooccurrence(ltg, DATA_DIR / "ltg_keyword_cooccurrence.csv")

    print("\n3. Author & institution analysis...")
    analyze_authors_institutions(poly, "Polycrisis")
    analyze_authors_institutions(ltg, "Limits to Growth")

    print("\n4. Most cited papers...")
    top_cited_papers(poly, "Polycrisis")
    top_cited_papers(ltg, "Limits to Growth")

    print("\n5. Journal distribution charts...")
    plot_journal_distribution(poly, "Polycrisis", "02_polycrisis_journals.png")
    plot_journal_distribution(ltg, "Limits to Growth", "03_ltg_journals.png")

    print("\nDone. Next steps:")
    print("  1. Open VOSviewer → Create map → Based on text data → import polycrisis_keyword_cooccurrence.csv")
    print("  2. Review figures/ folder for charts")
    print("  3. Run 03_vosviewer_export.py to format data for VOSviewer's native format")


if __name__ == "__main__":
    main()
