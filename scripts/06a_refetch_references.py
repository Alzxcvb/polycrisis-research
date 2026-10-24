"""
Refetch reference lists from OpenAlex for the citation bridge analysis
Project: Polycrisis & Limits to Growth Research
Author: Alex Coffman

What this does:
    The saved raw pulls have an empty referenced_works list for many papers
    (691 of 1,500 polycrisis rows and 466 of 1,154 LtG rows have one).
    This script asks OpenAlex again for the reference lists of EVERY unique
    corpus paper (all ids, not only the empty ones, so stale lists update;
    it is only about 53 calls). It does not touch the raw files.

    This script only fetches and saves. 06_citation_bridge.py then takes the
    union of the refetched list and the original list from the raw file for
    each paper (it also reports how often the two lists differ).

    Network: api.openalex.org only. No email or mailto is sent.

Run (from the repo root):
    .venv/bin/python scripts/06a_refetch_references.py

Output:
    data/references_refetch_2026-10-24.json
        {openalex_id: {"referenced_works": [...], "type": ..., "publication_year": ..., "fetched": ISO time}}
    Prints reference coverage before and after, by corpus and by type.
"""

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = DATA_DIR / "references_refetch_2026-10-24.json"
API = "https://api.openalex.org/works"
BATCH = 50
SLEEP = 0.2


# ── Load corpora ──────────────────────────────────────────────────────────────

def short_id(x):
    return str(x).rsplit("/", 1)[-1]


def load_corpora():
    """Return {corpus_name: set(ids)} and {id: best raw record}."""
    corpora, raw_best = {}, {}
    for name, csv, raw in [
        ("polycrisis", "polycrisis_papers.csv", "polycrisis_papers_raw.json"),
        ("ltg", "ltg_refined_papers.csv", "ltg_refined_papers_raw.json"),
    ]:
        ids = set(pd.read_csv(DATA_DIR / csv)["openalex_id"].map(short_id))
        corpora[name] = ids
        for w in json.load(open(DATA_DIR / raw)):
            wid = short_id(w["id"])
            refs = w.get("referenced_works") or []
            cur = raw_best.get(wid)
            if cur is None or len(refs) > len(cur.get("referenced_works") or []):
                raw_best[wid] = w
    return corpora, raw_best


# ── Fetch ─────────────────────────────────────────────────────────────────────

def get_with_retry(params, tries=3):
    for attempt in range(tries + 1):
        try:
            r = requests.get(API, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            if attempt == tries:
                raise
            print(f"    retry after error: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("OpenAlex request failed after retries")


def refetch(ids):
    out = {}
    ids = sorted(ids)
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        params = {
            "filter": "openalex_id:" + "|".join(chunk),
            "select": "id,referenced_works,publication_year,type",
            "per-page": BATCH,
        }
        data = get_with_retry(params)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for w in data.get("results", []):
            out[short_id(w["id"])] = {
                "referenced_works": [short_id(r) for r in (w.get("referenced_works") or [])],
                "type": w.get("type"),
                "publication_year": w.get("publication_year"),
                "fetched": now,
            }
        print(f"  batch {i // BATCH + 1}/{-(-len(ids) // BATCH)}: got {len(data.get('results', []))} of {len(chunk)}")
        time.sleep(SLEEP)
    return out


# ── Coverage report ───────────────────────────────────────────────────────────

def coverage(label, corpora, raw_best, refetched):
    print(f"\n── Reference coverage ({label}) ──")
    rows = []
    for cname, ids in corpora.items():
        for state in ("before", "after"):
            n = nonempty = 0
            by_type = defaultdict(lambda: [0, 0])
            for wid in ids:
                orig = [short_id(r) for r in (raw_best.get(wid, {}).get("referenced_works") or [])]
                new = refetched.get(wid, {}).get("referenced_works") or []
                refs = orig if state == "before" else (new or orig)
                t = (refetched.get(wid, {}).get("type") or raw_best.get(wid, {}).get("type") or "unknown")
                n += 1
                by_type[t][0] += 1
                if refs:
                    nonempty += 1
                    by_type[t][1] += 1
            rows.append((cname, state, n, nonempty, by_type))
    for cname, state, n, ne, by_type in rows:
        print(f"{cname:11s} {state:6s}: {ne:5d} of {n:5d} papers have references ({100 * ne / n:.1f}%)")
    print("\nBy type (non-empty / total), after refetch:")
    for cname, state, n, ne, by_type in rows:
        if state != "after":
            continue
        parts = [f"{t}: {v[1]}/{v[0]}" for t, v in sorted(by_type.items(), key=lambda kv: -kv[1][0])]
        print(f"  {cname}: " + "; ".join(parts))
    print("By type before refetch:")
    for cname, state, n, ne, by_type in rows:
        if state != "before":
            continue
        parts = [f"{t}: {v[1]}/{v[0]}" for t, v in sorted(by_type.items(), key=lambda kv: -kv[1][0])]
        print(f"  {cname}: " + "; ".join(parts))


def main():
    corpora, raw_best = load_corpora()
    all_ids = set().union(*corpora.values())
    print(f"Unique corpus ids: {len(all_ids)} "
          f"(polycrisis {len(corpora['polycrisis'])}, ltg {len(corpora['ltg'])}, "
          f"in both {len(corpora['polycrisis'] & corpora['ltg'])})")
    missing_raw = [i for i in all_ids if i not in raw_best]
    print(f"Ids in CSV with no raw record: {len(missing_raw)}")
    print(f"Refetching all {len(all_ids)} ids in batches of {BATCH}")
    refetched = refetch(all_ids)
    not_returned = all_ids - set(refetched)
    print(f"Returned {len(refetched)} of {len(all_ids)}; not returned: {len(not_returned)}")
    with open(OUT_PATH, "w") as f:
        json.dump(refetched, f)
    print(f"Saved: {OUT_PATH}")
    coverage("after: refetched list if non-empty, else saved list (06_citation_bridge.py uses the union of both lists)", corpora, raw_best, refetched)


if __name__ == "__main__":
    main()
