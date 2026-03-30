"""
VOSviewer Export — Format OpenAlex data for visual network mapping
Project: Polycrisis & Limits to Growth Research

What this does:
    VOSviewer (vosviewer.com) is free software that draws beautiful network maps
    of academic literature. It needs data in a specific format.
    This script exports your OpenAlex data into that format.

    The maps it produces look like this:
    https://app.vosviewer.com/#map&url=...

    Types of maps you can make:
    - Keyword co-occurrence (which topics cluster together?)
    - Author co-authorship (who collaborates with whom?)
    - Citation network (who cites whom?)
    - Bibliographic coupling (which papers address the same questions?)

Run:
    python 03_vosviewer_export.py

Output:
    data/vosviewer_works.csv  — import this into VOSviewer directly
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def export_for_vosviewer(df: pd.DataFrame, output_path: Path):
    """
    VOSviewer can import a CSV with specific column names.
    Required: title, abstract (for text analysis)
    Optional: authors, year, citations, source, doi
    """
    vos_df = pd.DataFrame({
        "title": df["title"],
        "abstract": df["abstract"],
        "author": df["authors"],
        "year": df["year"],
        "source": df["journal"],
        "doi": df["doi"],
        "citations": df["cited_by_count"],
    })

    vos_df.to_csv(output_path, index=False)
    print(f"VOSviewer export saved: {output_path} ({len(vos_df):,} rows)")
    print("\nTo use in VOSviewer:")
    print("  1. Download VOSviewer free at https://www.vosviewer.com/")
    print("  2. Open VOSviewer → Create → Based on bibliographic data")
    print("  3. Choose 'Read data from CSV files'")
    print(f"  4. Select: {output_path}")
    print("  5. Choose map type: Keyword co-occurrence, Author co-authorship, etc.")


def main():
    try:
        poly = pd.read_csv(DATA_DIR / "polycrisis_papers.csv")
        ltg = pd.read_csv(DATA_DIR / "ltg_papers.csv")
    except FileNotFoundError:
        print("Run 01_fetch_openalex.py first.")
        return

    export_for_vosviewer(poly, DATA_DIR / "vosviewer_polycrisis.csv")
    export_for_vosviewer(ltg, DATA_DIR / "vosviewer_ltg.csv")


if __name__ == "__main__":
    main()
