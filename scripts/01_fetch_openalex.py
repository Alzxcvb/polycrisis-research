"""
Bibliometric Data Collection — OpenAlex API
Project: Polycrisis & Limits to Growth Research
Author: Alex Coffman
Date: 2026-03-27

What this does:
    Pulls all academic papers mentioning "polycrisis" or "limits to growth"
    from the OpenAlex API (free, no key needed). Saves raw data as JSON and
    a cleaned CSV for analysis in VOSviewer or Python.

Run:
    pip install requests pandas tqdm
    python 01_fetch_openalex.py

Output:
    data/polycrisis_papers.csv
    data/ltg_papers.csv
    data/polycrisis_papers_raw.json
"""

import requests
import json
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

EMAIL = "your@email.com"  # Add your email — gets you into the "polite pool" (higher rate limits)
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://api.openalex.org/works"

QUERIES = {
    "polycrisis": {
        "filter": "title_and_abstract.search:polycrisis",
        "output_csv": DATA_DIR / "polycrisis_papers.csv",
        "output_json": DATA_DIR / "polycrisis_papers_raw.json",
    },
    "limits_to_growth": {
        "filter": 'title_and_abstract.search:"limits to growth"',
        "output_csv": DATA_DIR / "ltg_papers.csv",
        "output_json": DATA_DIR / "ltg_papers_raw.json",
    },
}

# Fields to extract per paper
FIELDS = [
    "id",
    "title",
    "publication_year",
    "publication_date",
    "type",
    "cited_by_count",
    "doi",
    "primary_location",
    "authorships",
    "concepts",
    "keywords",
    "referenced_works",
    "related_works",
    "abstract_inverted_index",  # OpenAlex stores abstracts inverted; we'll reconstruct
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [position, ...]}. Reconstruct the text."""
    if not inverted_index:
        return ""
    positions = []
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions.append((pos, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def get_journal_name(primary_location: dict | None) -> str:
    if not primary_location:
        return ""
    source = primary_location.get("source") or {}
    return source.get("display_name", "")


def get_authors(authorships: list) -> str:
    names = []
    for a in authorships[:5]:  # cap at 5 authors
        author = a.get("author") or {}
        names.append(author.get("display_name", ""))
    result = "; ".join(n for n in names if n)
    if len(authorships) > 5:
        result += f" et al. (+{len(authorships) - 5})"
    return result


def get_institutions(authorships: list) -> str:
    institutions = set()
    for a in authorships:
        for inst in a.get("institutions", []):
            name = inst.get("display_name", "")
            if name:
                institutions.add(name)
    return "; ".join(sorted(institutions)[:5])


def get_top_concepts(concepts: list) -> str:
    """Return top 5 concepts by score."""
    sorted_concepts = sorted(concepts, key=lambda x: x.get("score", 0), reverse=True)
    return "; ".join(c.get("display_name", "") for c in sorted_concepts[:5])


def fetch_all_works(filter_str: str, per_page: int = 200) -> list[dict]:
    """
    Paginate through all OpenAlex results for a given filter.
    Uses cursor-based pagination (handles >10k results).
    """
    all_works = []
    cursor = "*"
    page_num = 0

    # Get total count first
    params = {
        "filter": filter_str,
        "per_page": 1,
        "mailto": EMAIL,
    }
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    total = r.json()["meta"]["count"]
    print(f"  Total papers found: {total:,}")

    with tqdm(total=total, desc="  Fetching") as pbar:
        while cursor:
            params = {
                "filter": filter_str,
                "per_page": per_page,
                "cursor": cursor,
                "select": ",".join(FIELDS),
                "mailto": EMAIL,
            }
            r = requests.get(BASE_URL, params=params)
            r.raise_for_status()
            data = r.json()

            works = data.get("results", [])
            all_works.extend(works)
            pbar.update(len(works))

            # Get next cursor
            cursor = data.get("meta", {}).get("next_cursor")
            page_num += 1

            # Polite delay
            time.sleep(0.1)

    return all_works


def works_to_dataframe(works: list[dict]) -> pd.DataFrame:
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for query_name, config in QUERIES.items():
        print(f"\n{'='*60}")
        print(f"Query: {query_name}")
        print(f"Filter: {config['filter']}")

        # Fetch
        works = fetch_all_works(config["filter"])

        # Save raw JSON
        with open(config["output_json"], "w") as f:
            json.dump(works, f, indent=2)
        print(f"  Raw JSON saved: {config['output_json']}")

        # Convert and save CSV
        df = works_to_dataframe(works)
        df.to_csv(config["output_csv"], index=False)
        print(f"  CSV saved: {config['output_csv']} ({len(df):,} rows)")

        # Quick summary
        print(f"\n  Papers by year:")
        year_counts = df.groupby("year").size().sort_index()
        for year, count in year_counts.tail(10).items():
            bar = "█" * (count // 5 + 1)
            print(f"    {year}: {count:4d} {bar}")

        top_journals = df.groupby("journal").size().sort_values(ascending=False).head(10)
        print(f"\n  Top 10 journals:")
        for journal, count in top_journals.items():
            if journal:
                print(f"    {count:4d}  {journal}")


if __name__ == "__main__":
    main()
