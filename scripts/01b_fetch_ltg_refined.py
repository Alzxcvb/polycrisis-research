"""
Refined Limits to Growth Query — Excludes Biology Papers
Project: Polycrisis & Limits to Growth Research

Problem:
    The broad "limits to growth" query returns 13,000+ papers, many from
    biology/ecology ("iron limits to growth of phytoplankton," cancer cell
    growth limits, plant nutrition). Only ~700-1,000 are about the Meadows
    model and Club of Rome concept.

Solution:
    Use OpenAlex concept filtering + keyword co-occurrence to isolate papers
    actually about the Meadows/Club of Rome "Limits to Growth."

    Strategy 1: Search for "limits to growth" AND require co-occurrence with
    at least one LtG-specific term (meadows, club of rome, world3, overshoot,
    planetary boundaries, system dynamics, etc.)

    Strategy 2: Use OpenAlex concept IDs to filter to relevant domains
    (sustainability, environmental science, economics) and exclude biology.

Run:
    python 01b_fetch_ltg_refined.py

Output:
    data/ltg_refined_papers.csv
    data/ltg_refined_papers_raw.json
"""

import requests
import json
import time
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path

EMAIL = "your@email.com"
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
BASE_URL = "https://api.openalex.org/works"

# LtG-specific co-occurring terms — if a paper mentions "limits to growth"
# AND one of these, it's almost certainly about the Meadows concept
LTG_MARKERS = {
    "meadows", "club of rome", "world3", "world 3", "overshoot", "collapse",
    "planetary boundaries", "system dynamics", "carrying capacity",
    "ecological footprint", "steady state", "degrowth", "post-growth",
    "resource depletion", "population growth", "exponential growth",
    "sustainability transition", "ecological economics", "donella",
    "jorgen randers", "herrington", "turner", "nebel", "polycrisis",
    "forrester", "systems thinking", "feedback loop", "nonrenewable",
    "industrial output", "pollution", "finite world", "overshoot and collapse",
}


def reconstruct_abstract(inverted_index) -> str:
    if not inverted_index:
        return ""
    positions = []
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions.append((pos, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def get_journal_name(primary_location) -> str:
    if not primary_location:
        return ""
    source = primary_location.get("source") or {}
    return source.get("display_name", "")


def get_authors(authorships) -> str:
    names = []
    for a in authorships[:5]:
        author = a.get("author") or {}
        names.append(author.get("display_name", ""))
    result = "; ".join(n for n in names if n)
    if len(authorships) > 5:
        result += f" et al. (+{len(authorships) - 5})"
    return result


def get_institutions(authorships) -> str:
    institutions = set()
    for a in authorships:
        for inst in a.get("institutions", []):
            name = inst.get("display_name", "")
            if name:
                institutions.add(name)
    return "; ".join(sorted(institutions)[:5])


def get_top_concepts(concepts) -> str:
    sorted_concepts = sorted(concepts, key=lambda x: x.get("score", 0), reverse=True)
    return "; ".join(c.get("display_name", "") for c in sorted_concepts[:5])


def is_ltg_relevant(work) -> bool:
    """
    Check if a paper is actually about the Meadows 'Limits to Growth' concept
    by looking for co-occurring LtG-specific terms in title, abstract, and keywords.
    """
    text_parts = []

    # Title
    title = (work.get("title") or "").lower()
    text_parts.append(title)

    # Abstract
    abstract = reconstruct_abstract(work.get("abstract_inverted_index")).lower()
    text_parts.append(abstract)

    # Keywords
    for kw in work.get("keywords", []):
        text_parts.append((kw.get("display_name") or "").lower())

    # Concepts
    for c in work.get("concepts", []):
        text_parts.append((c.get("display_name") or "").lower())

    combined = " ".join(text_parts)

    # Check for LtG markers
    for marker in LTG_MARKERS:
        if marker in combined:
            return True

    # Also accept if the title itself says "limits to growth" with quotes-like emphasis
    # (i.e., it's the subject, not just a phrase used in passing)
    if re.search(r'\blimits to growth\b', title) and len(title) < 120:
        # Short title with exact phrase = likely about the concept
        # Check it's not biology
        bio_markers = {"cancer", "tumor", "cell", "phytoplankton", "plant",
                       "bacterial", "microbial", "algae", "fungal", "yeast",
                       "zeolite", "enzyme", "protein", "gene", "iron",
                       "nitrogen", "phosphorus", "nutrient"}
        if not any(bm in combined for bm in bio_markers):
            return True

    return False


def fetch_all_works(filter_str, per_page=200):
    all_works = []
    cursor = "*"

    params = {
        "filter": filter_str,
        "per_page": 1,
        "mailto": EMAIL,
    }
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    total = r.json()["meta"]["count"]
    print(f"  Total papers in OpenAlex: {total:,}")

    select_fields = [
        "id", "title", "publication_year", "publication_date", "type",
        "cited_by_count", "doi", "primary_location", "authorships",
        "concepts", "keywords", "referenced_works", "related_works",
        "abstract_inverted_index",
    ]

    with tqdm(total=total, desc="  Fetching") as pbar:
        while cursor:
            params = {
                "filter": filter_str,
                "per_page": per_page,
                "cursor": cursor,
                "select": ",".join(select_fields),
                "mailto": EMAIL,
            }
            r = requests.get(BASE_URL, params=params)
            r.raise_for_status()
            data = r.json()
            works = data.get("results", [])
            all_works.extend(works)
            pbar.update(len(works))
            cursor = data.get("meta", {}).get("next_cursor")
            time.sleep(0.1)

    return all_works


def works_to_dataframe(works):
    rows = []
    for w in works:
        rows.append({
            "openalex_id": w.get("id", "").replace("https://openalex.org/", ""),
            "title": w.get("title", ""),
            "year": w.get("publication_year"),
            "date": w.get("publication_date", ""),
            "type": w.get("type", ""),
            "doi": w.get("doi", ""),
            "journal": get_journal_name(w.get("primary_location")),
            "authors": get_authors(w.get("authorships", [])),
            "institutions": get_institutions(w.get("authorships", [])),
            "cited_by_count": w.get("cited_by_count", 0),
            "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
            "concepts": get_top_concepts(w.get("concepts", [])),
            "keywords": "; ".join(
                kw.get("display_name", "") for kw in w.get("keywords", [])[:10]
            ),
            "reference_count": len(w.get("referenced_works", [])),
            "related_count": len(w.get("related_works", [])),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("REFINED Limits to Growth Query")
    print("Fetching all 'limits to growth' papers, then filtering")
    print("to only Meadows/Club of Rome related work.")
    print("=" * 60)

    filter_str = 'title_and_abstract.search:"limits to growth"'
    works = fetch_all_works(filter_str)

    # Filter for relevance
    print(f"\n  Filtering {len(works):,} papers for LtG relevance...")
    relevant = [w for w in works if is_ltg_relevant(w)]
    rejected = len(works) - len(relevant)
    print(f"  Kept: {len(relevant):,} papers")
    print(f"  Rejected: {rejected:,} papers (biology, off-topic)")

    # Save raw JSON
    json_path = DATA_DIR / "ltg_refined_papers_raw.json"
    with open(json_path, "w") as f:
        json.dump(relevant, f, indent=2)
    print(f"  Raw JSON saved: {json_path}")

    # Convert and save CSV
    df = works_to_dataframe(relevant)
    csv_path = DATA_DIR / "ltg_refined_papers.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path} ({len(df):,} rows)")

    # Quick summary
    print(f"\n  Papers by year:")
    year_counts = df.groupby("year").size().sort_index()
    for year, count in year_counts.tail(15).items():
        bar = "█" * (count // 3 + 1)
        print(f"    {year}: {count:4d} {bar}")

    top_journals = df.groupby("journal").size().sort_values(ascending=False).head(10)
    print(f"\n  Top 10 journals (refined):")
    for journal, count in top_journals.items():
        if journal:
            print(f"    {count:4d}  {journal}")

    # Most cited
    print(f"\n  Top 10 most-cited papers (refined):")
    top = df.nlargest(10, "cited_by_count")
    for _, row in top.iterrows():
        print(f"    [{int(row['year'])}] {str(row['title'])[:70]}... (cited: {row['cited_by_count']:,})")


if __name__ == "__main__":
    main()
