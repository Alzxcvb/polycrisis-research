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

# Tiered markers for isolating the Meadows/Club of Rome "Limits to Growth"
# concept from the biology/ecology noise (cell growth, plant growth, etc.).
#
# STRONG = names/terms that are essentially unique to the Meadows literature.
# A paper mentioning any of these is accepted even if biology terms appear.
STRONG_LTG_MARKERS = {
    "meadows", "donella", "jorgen randers", "forrester", "club of rome",
    "world3", "world 3", "herrington", "nebel", "turner",
    "overshoot and collapse", "finite world",
    "polycrisis", "degrowth", "post-growth",
}

# MEDIUM = LtG-adjacent concepts that are also used in ecology/biology.
# Accept only when NO biology/ecology markers appear.
MEDIUM_LTG_MARKERS = {
    "planetary boundaries", "system dynamics", "systems thinking",
    "ecological footprint", "carrying capacity", "steady state economy",
    "ecological economics", "sustainability transition",
    "industrial output", "nonrenewable resources",
    "overshoot",
}

# Biology / ecology / medicine / chemistry noise. If any of these appear
# AND no STRONG marker is present, reject the paper as off-topic.
BIO_ECO_MARKERS = {
    # ecology / environmental bio
    "ecology", "ecological niche", "ecosystem function",
    "phytoplankton", "cyanobacteria", "algae", "algal", "fungal", "fungi",
    "yeast", "bacterial", "bacteria", "microbial", "microbe",
    "biofilm", "rhizosphere", "mycorrhiz",
    # cell / cancer biology
    "cancer", "tumor", "tumour", "carcinoma", "metastasis", "oncology",
    "cell proliferation", "cell cycle", "cell culture", "cell line",
    "in vitro", "in vivo", "clinical trial", "patient outcomes",
    # plant / agriculture
    "seedling", "germination", "leaf area", "crop yield", "agronomic",
    "horticulture", "cultivar", "rootstock", "photosynthesis",
    # animal biology
    "broiler", "livestock", "juvenile fish", "larvae", "larval",
    "insect", "rodent", "mammal", "avian", "poultry",
    # molecular bio
    "enzyme", "genome", "chromosome", "dna", "rna transcription",
    "protein expression", "gene expression",
    # elemental / nutrient bio
    "iron limitation", "nitrogen limitation", "phosphorus limitation",
    "nutrient limitation", "phosphate", "nitrate",
    # medicine
    "diabetes", "cardiovascular", "patient", "dosage", "therapy",
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


def is_ltg_relevant(work) -> tuple[bool, str]:
    """
    Tiered relevance check. Returns (keep, reason) so we can audit the filter.

    Rules:
      1. STRONG marker in title/abstract/keywords -> keep (reason: strong)
      2. Any bio/ecology marker -> reject (reason: bio_noise)
      3. MEDIUM marker -> keep (reason: medium)
      4. Short title with exact 'limits to growth' and no bio noise -> keep
      5. Otherwise -> reject
    """
    title = (work.get("title") or "").lower()
    abstract = reconstruct_abstract(work.get("abstract_inverted_index")).lower()
    kw_text = " ".join(
        (kw.get("display_name") or "").lower() for kw in work.get("keywords", [])
    )
    concept_text = " ".join(
        (c.get("display_name") or "").lower() for c in work.get("concepts", [])
    )
    combined = " ".join([title, abstract, kw_text, concept_text])

    # Rule 1: strong markers override everything
    for m in STRONG_LTG_MARKERS:
        if m in combined:
            return True, "strong"

    # Rule 2: biology/ecology noise -> reject
    for m in BIO_ECO_MARKERS:
        if m in combined:
            return False, f"bio_noise:{m}"

    # Rule 3: medium markers with no bio noise -> keep
    for m in MEDIUM_LTG_MARKERS:
        if m in combined:
            return True, "medium"

    # Rule 4: exact-phrase short title, clean of bio noise
    if re.search(r"\blimits to growth\b", title) and len(title) < 120:
        return True, "exact_title"

    return False, "no_marker"


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
    import sys

    print("=" * 60)
    print("REFINED Limits to Growth Query")
    print("=" * 60)

    # Two modes:
    #   default        -> fetch from OpenAlex (slow, hits the network)
    #   --refilter     -> reapply filter to cached ltg_refined_papers_raw.json
    refilter = "--refilter" in sys.argv
    raw_path = DATA_DIR / "ltg_refined_papers_raw.json"

    if refilter and raw_path.exists():
        print(f"  Re-filtering cached data from {raw_path.name}")
        with open(raw_path) as f:
            works = json.load(f)
        print(f"  Loaded {len(works):,} cached works")
    else:
        filter_str = 'title_and_abstract.search:"limits to growth"'
        works = fetch_all_works(filter_str)

    # Filter for relevance with reason tracking
    print(f"\n  Filtering {len(works):,} papers for LtG relevance...")
    relevant = []
    reason_counts: dict[str, int] = {}
    for w in works:
        keep, reason = is_ltg_relevant(w)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if keep:
            relevant.append(w)
    rejected = len(works) - len(relevant)
    print(f"  Kept: {len(relevant):,} papers")
    print(f"  Rejected: {rejected:,} papers")
    print("  Decision breakdown:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {count:>5,}  {reason}")

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
