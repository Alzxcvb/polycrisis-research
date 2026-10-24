"""
Temporal Keyword Convergence: is the vocabulary of the two fields converging?
Project: Polycrisis & Limits to Growth Research
Author: Alex Coffman

What this does:
    Splits both corpora into time periods and, for each period, compares the
    top 50 keywords of polycrisis papers with the top 50 of Limits to Growth
    (LtG) papers: overlap count, Jaccard of the two top 50 sets, weighted
    Jaccard over all keywords, and cosine similarity of per paper normalised
    keyword vectors. Also lists which keywords newly enter each top 50.

    Keyword source and cleaning are the same as the "21 of the top 50"
    headline in 05_simple_viz.py, except for one deliberate change: repeated
    OpenAlex ids in the paper CSVs are kept once (the script computes and
    prints how many term counts this changes against 05's saved files). The `keywords` plus `concepts` columns of
    the paper CSVs (as in 02_bibliometric_analysis.py: lowercase, longer than
    2 characters, one count per paper), then 05's STOPWORDS and its rule that
    drops terms containing "(". The full period number is recomputed first as
    a consistency check and compared with the 05 route (the saved
    *_keyword_cooccurrence_frequencies.csv files).

Run (from the repo root):
    .venv/bin/python scripts/07_temporal_keywords.py

Output:
    data/temporal_keyword_overlap.csv
    data/temporal_top_keywords.csv
    figures/temporal_keyword_convergence.png
    docs/RESULTS_temporal_keywords.md
No network access.
"""

import ast
import json
import re
import unicodedata
from collections import Counter
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
DOC_DIR = ROOT / "docs"
FIG_DIR.mkdir(exist_ok=True)
DOC_DIR.mkdir(exist_ok=True)

TOP_N = 50
# Bins chosen from the year counts printed at the start of a run. The polycrisis
# corpus has only 22 papers up to 2019 and 84 up to 2022, so everything up to
# 2022 is one bin. LtG has 50 papers in 2023 (47 with tags) and 47 papers in 2024 (39 with
# tags), under 50 once papers with no tags are dropped, so those two years are merged for the main
# analysis and shown singly for reference only. Counts after dropping papers with no
# tags are printed and checked against MIN_PAPERS.
BINS = [("up to 2022", None, 2022), ("2023 to 2024", 2023, 2024), ("2025 to 2026", 2025, 2026)]
# Single years shown for reference only: LtG has fewer than 50 tagged papers in each.
REFERENCE_ONLY = [("2023", 2023, 2023), ("2024", 2024, 2024)]
MIN_PAPERS = 45            # about 50; a bin below this would be merged
N_SUBSAMPLES = 300
CONTROL_N = 60     # papers drawn from each corpus in the size matched control (below the smallest main period corpus, 75)
TIE_DRAWS = 1000
SEED = 20261024

# Copied from 05_simple_viz.py (that file imports plotting libraries that are not
# needed here). A runtime check below asserts this equals 05's set.
STOPWORDS = {
    "political science", "sociology", "computer science", "mathematics",
    "physics", "psychology", "engineering", "chemistry", "biology",
    "art", "humanities", "geography", "history", "philosophy", "medicine",
    "law", "materials science", "environmental science", "business",
    "public relations", "library science", "social science",
    "botany", "agronomy", "horticulture", "cell biology", "biochemistry",
    "cancer research", "gene", "genetics", "nutrient", "microbiology",
    "biophysics", "biotechnology", "molecular biology", "endocrinology",
    "context (archaeology)", "content (measure theory)",
    "action (physics)", "quantum entanglement",
    "resilience (materials science)", "government (linguistics)",
    "crisis response", "epistemology",
}


def _check_stopwords_against_05():
    """05 cannot be imported here (it loads plotting libraries and builds figures), so read its STOPWORDS
    set literal from the source and fail loudly if this copy has drifted."""
    src = (Path(__file__).parent / "05_simple_viz.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "STOPWORDS" for t in node.targets):
            assert set(ast.literal_eval(node.value)) == STOPWORDS, "STOPWORDS differs from scripts/05_simple_viz.py"
            return
    raise AssertionError("STOPWORDS not found in scripts/05_simple_viz.py")


_check_stopwords_against_05()


# ── Keywords ──────────────────────────────────────────────────────────────────

def paper_terms(row):
    """Set of cleaned terms for one paper (02 extraction, then 05 filtering)."""
    terms = set()
    for col in ("keywords", "concepts"):
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue          # empty cell (02 turns these into the text "nan"; not done here)
        for t in str(val).split(";"):
            t = t.strip().lower()
            if t and len(t) > 2:
                terms.add(t)
    return {t for t in terms if t not in STOPWORDS and "(" not in t}


def load():
    poly = pd.read_csv(DATA_DIR / "polycrisis_papers.csv")
    ltg = pd.read_csv(DATA_DIR / "ltg_refined_papers.csv")
    # the CSVs hold repeated OpenAlex ids; keep one row per id within each corpus
    n_raw = (len(poly), len(ltg))
    raw_full = (poly.copy(), ltg.copy())     # all rows, repeated ids kept, for the 05 comparison
    poly = poly.drop_duplicates("openalex_id").reset_index(drop=True)
    ltg = ltg.drop_duplicates("openalex_id").reset_index(drop=True)
    print(f"Rows in CSVs {n_raw}; unique ids kept: polycrisis {len(poly)}, ltg {len(ltg)}")
    for df in (poly, ltg):
        df["terms"] = [paper_terms(r) for r in df.to_dict("records")]
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return poly, ltg, raw_full, n_raw


def top_terms(papers, n=TOP_N):
    """papers: list of term sets. Returns [(term, count)] top n (ties by term)."""
    c = Counter()
    for s in papers:
        c.update(s)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n], c


def similarities(papers_p, papers_l):
    cp, cl = Counter(), Counter()
    for s in papers_p:
        cp.update(s)
    for s in papers_l:
        cl.update(s)
    tp, tl = ({t for t, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_N]} for c in (cp, cl))
    ov = tp & tl
    jac = len(ov) / len(tp | tl) if tp | tl else float("nan")
    # weighted Jaccard on share of papers containing each term
    vocab = set(cp) | set(cl)
    sp = {t: cp[t] / len(papers_p) for t in vocab}
    sl = {t: cl[t] / len(papers_l) for t in vocab}
    num = sum(min(sp[t], sl[t]) for t in vocab)
    den = sum(max(sp[t], sl[t]) for t in vocab)
    wj = num / den if den else float("nan")
    # cosine of vectors where each paper contributes weight 1 spread over its terms
    def vec(papers):
        v = Counter()
        for s in papers:
            for t in s:
                v[t] += 1.0 / len(s)
        return v
    vp, vl = vec([s for s in papers_p if s]), vec([s for s in papers_l if s])
    dot = sum(vp[t] * vl[t] for t in vp if t in vl)
    norm = sqrt(sum(x * x for x in vp.values())) * sqrt(sum(x * x for x in vl.values()))
    cos = dot / norm if norm else float("nan")
    return len(ov), jac, wj, cos, tp, tl


def tie_parts(counter):
    """Split a keyword counter at the top N cutoff. Returns (cutoff count, strict set
    = terms with count above the cutoff, sorted tied terms, open slots)."""
    srt = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    cut = srt[TOP_N - 1][1] if len(srt) >= TOP_N else 0
    strict = {t for t, c in srt if c > cut}
    tied = sorted(t for t, c in srt if c == cut)
    return cut, strict, tied, TOP_N - len(strict)


def random_top_set(parts, rng):
    _cut, strict, tied, slots = parts
    if slots >= len(tied):
        return strict | set(tied)
    pick = rng.choice(len(tied), slots, replace=False)
    return strict | {tied[i] for i in pick}


def count_terms(papers):
    c = Counter()
    for s in papers:
        c.update(s)
    return c


def tie_robust(papers_p, papers_l, rng, draws):
    """Overlap of the two top N lists under three tie rules: strict (only terms above
    the cutoff count), alphabetical, and random tie breaking (mean and 5th to 95th percentile)."""
    pp, pl = tie_parts(count_terms(papers_p)), tie_parts(count_terms(papers_l))
    strict = len(pp[1] & pl[1])
    ov = [len(random_top_set(pp, rng) & random_top_set(pl, rng)) for _ in range(draws)]
    return {"strict": strict, "mean": float(np.mean(ov)), "p5": float(np.percentile(ov, 5)),
            "p95": float(np.percentile(ov, 95)), "parts_p": pp, "parts_l": pl,
            "strict_p": pp[1], "strict_l": pl[1]}


def random_overlap(papers_p, papers_l, rng):
    pp, pl = tie_parts(count_terms(papers_p)), tie_parts(count_terms(papers_l))
    return len(random_top_set(pp, rng) & random_top_set(pl, rng))


def _norm(t):
    t = unicodedata.normalize("NFKD", str(t or "")).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t)).strip()


def merge_records(df):
    """Keep one record per work within a corpus: same normalised title (12+ characters) and same
    first author surname, same rule as 06's paper merge. The record kept is chosen by an explicit,
    order independent rule: highest cited_by_count, then most distinct terms, then lowest OpenAlex id.
    Returns (merged frame, number of merged groups, number of groups whose top two records tie on cited_by_count)."""
    keys = {}
    for i, r in df.iterrows():
        nt = _norm(r["title"])
        a = str(r["authors"]) if isinstance(r["authors"], str) else ""
        first = a.split(";")[0].split("(")[0].strip().split()
        sn = _norm(first[-1]) if first else ""
        keys[i] = "|".join((nt, sn)) if len(nt) >= 12 and sn else "unique|" + str(i)
    t = df.assign(_k=pd.Series(keys), _nt=[len(x) for x in df["terms"]])
    t = t.sort_values(["_k", "cited_by_count", "_nt", "openalex_id"], ascending=[True, False, False, True], kind="mergesort")
    n_groups = n_tied = 0
    for _k, g in t.groupby("_k", sort=True):
        if len(g) > 1:
            n_groups += 1
            n_tied += int(g["cited_by_count"].iloc[0] == g["cited_by_count"].iloc[1])
    return t.drop_duplicates("_k").drop(columns=["_k", "_nt"]).sort_index(), n_groups, n_tied


def in_bin(df, lo, hi):
    y = df["year"]
    m = y.notna()
    if lo is not None:
        m &= y >= lo
    if hi is not None:
        m &= y <= hi
    return df[m]


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join([":--"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    return "\n".join(out)


def main():
    poly, ltg, raw_full, (n_rows_poly, n_rows_ltg) = load()
    n_both = len(set(poly["openalex_id"]) & set(ltg["openalex_id"]))
    def kw_len(df):
        return int(max((len([t for t in str(v).split(";") if t.strip()]) for v in df["keywords"].dropna()), default=0))
    max_kw = max(kw_len(poly), kw_len(ltg))
    print(f"Papers in both corpora: {n_both}; max keywords in one paper's keywords cell: {max_kw}")

    # truncation facts from the raw API records
    raw_kw = {}
    raw_co = {}
    for f in ("polycrisis_papers_raw.json", "ltg_refined_papers_raw.json"):
        for w in json.load(open(DATA_DIR / f)):
            raw_kw[w["id"]] = len(w.get("keywords") or [])
            raw_co[w["id"]] = len(w.get("concepts") or [])
    n_raw_papers = len(raw_kw)
    n_kw_cut = sum(1 for v in raw_kw.values() if v > 10)
    n_kw_max = max(raw_kw.values())
    n_co_cut = sum(1 for v in raw_co.values() if v > 5)
    print(f"Raw records {n_raw_papers}: with more than 10 keywords {n_kw_cut} (max {n_kw_max}); with more than 5 concepts {n_co_cut}")

    # year counts
    yc = pd.DataFrame({"polycrisis": poly["year"].value_counts(), "ltg": ltg["year"].value_counts()}).fillna(0).astype(int).sort_index()
    print("Papers per year (2010 onward; earlier years are LtG only):")
    print(yc[yc.index >= 2010].T.to_string())
    print(f"Papers with no year: polycrisis {int(poly['year'].isna().sum())}, ltg {int(ltg['year'].isna().sum())}")
    print(f"Papers with no usable terms: polycrisis {sum(1 for s in poly['terms'] if not s)}, ltg {sum(1 for s in ltg['terms'] if not s)}")

    # consistency check against the 21
    # full period uses tagged papers only (same rule as the period rows), with or without a year
    poly_t = poly[[bool(t) for t in poly["terms"]]]
    ltg_t = ltg[[bool(t) for t in ltg["terms"]]]
    n_full_p, n_full_l = len(poly_t), len(ltg_t)
    n_full_noyear_p, n_full_noyear_l = int(poly_t["year"].isna().sum()), int(ltg_t["year"].isna().sum())
    print(f"Full period tagged papers: polycrisis {n_full_p} ({n_full_noyear_p} without a year), ltg {n_full_l} ({n_full_noyear_l} without a year)")
    ov_full, jac_full, wj_full, cos_full, tp_full, tl_full = similarities(list(poly_t["terms"]), list(ltg_t["terms"]))
    print(f"\nCONSISTENCY CHECK, full period from paper CSVs: overlap of top {TOP_N} = {ov_full}, "
          f"Jaccard {jac_full:.3f}, weighted Jaccard {wj_full:.3f}, cosine {cos_full:.3f}")
    # 05 route: saved frequency files
    def route05(path):
        df = pd.read_csv(path).dropna(subset=["term"])
        df["term"] = df["term"].astype(str).str.strip().str.lower()
        df = df[(df["term"] != "") & ~df["term"].isin(STOPWORDS) & ~df["term"].str.contains(r"\(", regex=True)]
        return set(df.head(TOP_N)["term"])
    r05p = route05(DATA_DIR / "polycrisis_keyword_cooccurrence_frequencies.csv")
    r05l = route05(DATA_DIR / "ltg_keyword_cooccurrence_frequencies.csv")
    ov05 = len(r05p & r05l)

    # why do counts differ from 05's saved files? compare with and without the id dedupe
    def saved_counts(path):
        df = pd.read_csv(path).dropna(subset=["term"])
        df["term"] = df["term"].astype(str).str.strip().str.lower()
        df = df[(df["term"] != "") & ~df["term"].isin(STOPWORDS) & ~df["term"].str.contains(r"\(", regex=True)]
        return [(r.term, int(r.count)) for r in df.head(TOP_N).itertuples()]
    def mismatches(saved, term_lists, k):
        c = Counter()
        for t in term_lists:
            c.update(t)
        return sum(1 for term, cnt in saved[:k] if c.get(term, 0) != cnt)
    sv_p = saved_counts(DATA_DIR / "polycrisis_keyword_cooccurrence_frequencies.csv")
    sv_l = saved_counts(DATA_DIR / "ltg_keyword_cooccurrence_frequencies.csv")
    full_p = [paper_terms(r) for r in raw_full[0].to_dict("records")]
    full_l = [paper_terms(r) for r in raw_full[1].to_dict("records")]
    cmm = {}
    for k in (45, TOP_N):
        cmm[k] = {"all rows polycrisis": mismatches(sv_p, full_p, k), "all rows ltg": mismatches(sv_l, full_l, k),
                 "deduped polycrisis": mismatches(sv_p, list(poly["terms"]), k), "deduped ltg": mismatches(sv_l, list(ltg["terms"]), k)}
        print(f"Count mismatches against 05's saved counts, first {k} terms: {cmm[k]}")
    print(f"05 route (saved frequency CSVs): overlap = {ov05}")
    print(f"Same sets as 05 route? polycrisis {tp_full == r05p}, ltg {tl_full == r05l}")
    print(f"Terms in my top {TOP_N} but not 05's: polycrisis {sorted(tp_full - r05p)}, ltg {sorted(tl_full - r05l)}")
    print(f"Terms in 05's top {TOP_N} but not mine: polycrisis {sorted(r05p - tp_full)}, ltg {sorted(r05l - tl_full)}")

    def tie_info(papers):
        c = Counter()
        for t in papers:
            c.update(t)
        srt = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        cut = srt[TOP_N - 1][1]
        return cut, sum(1 for _, v in srt if v == cut), sum(1 for _, v in srt[TOP_N:] if v == cut)
    tie_p, tie_l = tie_info(list(poly_t["terms"])), tie_info(list(ltg_t["terms"]))
    print(f"Tie check at rank {TOP_N}: polycrisis (count at cutoff, terms with that count, tied terms left out) {tie_p}; ltg {tie_l}")
    rng = np.random.default_rng(SEED)
    full_tr = tie_robust(list(poly_t["terms"]), list(ltg_t["terms"]), rng, TIE_DRAWS)
    # sensitivity: also merge repeated records of the same work (06's paper rule)
    poly_m, grp_p, tied_p = merge_records(poly)
    ltg_m, grp_l, tied_l = merge_records(ltg)
    n_merged_p, n_merged_l = len(poly) - len(poly_m), len(ltg) - len(ltg_m)
    n_tagged_merged_p = int(sum(1 for i in poly.index.difference(poly_m.index) if poly.loc[i, "terms"]))
    n_tagged_merged_l = int(sum(1 for i in ltg.index.difference(ltg_m.index) if ltg.loc[i, "terms"]))
    print(f"Extra records of the same work removed by the title and first author merge: polycrisis {n_merged_p} ({n_tagged_merged_p} tagged), ltg {n_merged_l} ({n_tagged_merged_l} tagged)")
    rng_m = np.random.default_rng(SEED + 7)
    def tagged(df):
        return df[[bool(t) for t in df["terms"]]]
    fm = tie_robust(list(tagged(poly_m)["terms"]), list(tagged(ltg_m)["terms"]), rng_m, TIE_DRAWS)
    print(f"Merged-record sensitivity, full period: strict {fm['strict']}, random mean {fm['mean']:.2f} ({fm['p5']:.0f} to {fm['p95']:.0f})")
    print(f"Full period tie robust: alphabetical {ov_full}, strict {full_tr['strict']}, random mean {full_tr['mean']:.2f} (5th to 95th {full_tr['p5']:.0f} to {full_tr['p95']:.0f})")
    print(f"Full period ties: polycrisis cutoff {full_tr['parts_p'][0]}, {len(full_tr['parts_p'][2])} tied, {full_tr['parts_p'][3]} slots; ltg cutoff {full_tr['parts_l'][0]}, {len(full_tr['parts_l'][2])} tied, {full_tr['parts_l'][3]} slots")
    # bins
    rows, topkw = [], []
    for tier, lo_hi_list in (("main", BINS), ("reference_only", REFERENCE_ONLY)):
        prev = None
        for label, lo, hi in lo_hi_list:
            bp_all, bl_all = in_bin(poly, lo, hi), in_bin(ltg, lo, hi)
            # papers with no keyword or concept tags are left out of every measure
            bp = bp_all[[bool(t) for t in bp_all["terms"]]]
            bl = bl_all[[bool(t) for t in bl_all["terms"]]]
            n_p, n_l = len(bp), len(bl)
            print(f"  papers in period before dropping untagged: polycrisis {len(bp_all)}, ltg {len(bl_all)}")
            flag = "" if min(n_p, n_l) >= 50 else " (below 50)"
            print(f"\nPeriod {label}: polycrisis n={n_p}, ltg n={n_l}{flag}")
            assert tier != "main" or min(n_p, n_l) >= MIN_PAPERS, f"bin {label} too small"
            tp_list, cp = top_terms(list(bp["terms"]))
            tl_list, cl = top_terms(list(bl["terms"]))
            ov, jac, wj, cos, sp_, sl_ = similarities(list(bp["terms"]), list(bl["terms"]))
            tr = tie_robust(list(bp["terms"]), list(bl["terms"]), rng, TIE_DRAWS)
            bpm, blm = tagged(in_bin(poly_m, lo, hi)), tagged(in_bin(ltg_m, lo, hi))
            trm = tie_robust(list(bpm["terms"]), list(blm["terms"]), rng_m, TIE_DRAWS)
            strict_sh = tr["strict_p"] & tr["strict_l"]
            # size matched control: the same number of papers from each corpus, same size in every period
            k = min(CONTROL_N, n_p, n_l)
            pl, ll = list(bp["terms"]), list(bl["terms"])
            sub, subwj, subcos = [], [], []
            for _ in range(N_SUBSAMPLES):
                ip = rng.choice(n_p, k, replace=False)
                il = rng.choice(n_l, k, replace=False)
                A_, B_ = [pl[i] for i in ip], [ll[i] for i in il]
                sub.append(random_overlap(A_, B_, rng))
                _o, _j, w_, c_, _a, _b = similarities(A_, B_)
                subwj.append(w_)
                subcos.append(c_)
            for corpus, tl_, n in (("polycrisis", tp_list, n_p), ("ltg", tl_list, n_l)):
                for rank, (t, c) in enumerate(tl_, 1):
                    topkw.append({"tier": tier, "period": label, "corpus": corpus, "rank": rank, "keyword": t,
                                  "papers": c, "share_of_papers": round(c / n, 4)})
            row = {"tier": tier, "period": label, "n_polycrisis_all": len(bp_all), "n_ltg_all": len(bl_all),
                   "n_polycrisis": n_p, "n_ltg": n_l, "overlap_top50": ov,
                   "jaccard_top50": jac, "weighted_jaccard_share": wj,
                   "cosine_per_paper": cos,
                   "matched_overlap_mean": float(np.mean(sub)),
                   "matched_overlap_sd": float(np.std(sub)),
                   "matched_overlap_p5": float(np.percentile(sub, 5)), "matched_overlap_p95": float(np.percentile(sub, 95)),
                   "matched_wj_mean": float(np.mean(subwj)), "matched_wj_p5": float(np.percentile(subwj, 5)), "matched_wj_p95": float(np.percentile(subwj, 95)),
                   "matched_cos_mean": float(np.mean(subcos)), "matched_cos_p5": float(np.percentile(subcos, 5)), "matched_cos_p95": float(np.percentile(subcos, 95)),
                   "matched_n_per_corpus": k,
                   "mean_terms_polycrisis": float(np.mean([len(t) for t in bp["terms"]])),
                   "mean_terms_ltg": float(np.mean([len(t) for t in bl["terms"]])),
                   "untagged_share_polycrisis": 1 - n_p / len(bp_all), "untagged_share_ltg": 1 - n_l / len(bl_all),
                   "merged_n_polycrisis": len(bpm), "merged_n_ltg": len(blm), "merged_overlap_strict": trm["strict"],
                   "merged_overlap_random_mean": trm["mean"], "merged_overlap_random_p5": trm["p5"], "merged_overlap_random_p95": trm["p95"],
                   "overlap_strict": tr["strict"], "overlap_random_mean": tr["mean"],
                   "overlap_random_p5": tr["p5"], "overlap_random_p95": tr["p95"],
                   "cutoff_count_polycrisis": tr["parts_p"][0], "tied_polycrisis": len(tr["parts_p"][2]), "slots_polycrisis": tr["parts_p"][3],
                   "cutoff_count_ltg": tr["parts_l"][0], "tied_ltg": len(tr["parts_l"][2]), "slots_ltg": tr["parts_l"][3],
                   "shared_keywords": "; ".join(sorted(sp_ & sl_)),
                   "shared_strict": "; ".join(sorted(strict_sh))}
            if prev:
                row["entered_polycrisis_top50"] = "; ".join(sorted(sp_ - prev[0]))
                row["entered_ltg_top50"] = "; ".join(sorted(sl_ - prev[1]))
                row["newly_shared"] = "; ".join(sorted((sp_ & sl_) - (prev[0] & prev[1])))
                # entered = in this period's strict set and not among the previous period's top 50 candidates
                # (previous strict set plus every term tied at the previous cutoff)
                ent_p = tr["strict_p"] - prev[2]
                ent_l = tr["strict_l"] - prev[3]
                new_sh = strict_sh - (prev[2] & prev[3])
                row["entered_polycrisis_strict"] = "; ".join(sorted(ent_p))
                row["entered_ltg_strict"] = "; ".join(sorted(ent_l))
                row["newly_shared_strict"] = "; ".join(sorted(new_sh))
                old_p, old_l, old_sh = len(tr["strict_p"] - prev[5]), len(tr["strict_l"] - prev[6]), len(strict_sh - prev[4])
                print(f"  entered terms removed by the tie candidate rule in {label}: polycrisis {old_p - len(ent_p)} (of {old_p}), ltg {old_l - len(ent_l)} (of {old_l}), newly shared {old_sh - len(new_sh)} (of {old_sh})")
            else:
                row["entered_polycrisis_top50"] = row["entered_ltg_top50"] = row["newly_shared"] = ""
                row["entered_polycrisis_strict"] = row["entered_ltg_strict"] = row["newly_shared_strict"] = ""
            prev = (sp_, sl_, tr["strict_p"] | set(tr["parts_p"][2]), tr["strict_l"] | set(tr["parts_l"][2]), strict_sh, tr["strict_p"], tr["strict_l"])
            rows.append(row)
            print({k_: v for k_, v in row.items() if not isinstance(v, str)})
    full_row = {"tier": "full_period", "period": "full period (tagged papers, any year)",
                "n_polycrisis_all": len(poly), "n_ltg_all": len(ltg), "n_polycrisis": n_full_p, "n_ltg": n_full_l,
                "overlap_top50": ov_full, "jaccard_top50": jac_full, "weighted_jaccard_share": wj_full, "cosine_per_paper": cos_full,
                "overlap_strict": full_tr["strict"], "overlap_random_mean": full_tr["mean"],
                "overlap_random_p5": full_tr["p5"], "overlap_random_p95": full_tr["p95"],
                "cutoff_count_polycrisis": full_tr["parts_p"][0], "tied_polycrisis": len(full_tr["parts_p"][2]), "slots_polycrisis": full_tr["parts_p"][3],
                "cutoff_count_ltg": full_tr["parts_l"][0], "tied_ltg": len(full_tr["parts_l"][2]), "slots_ltg": full_tr["parts_l"][3],
                "merged_overlap_strict": fm["strict"], "merged_overlap_random_mean": fm["mean"],
                "merged_overlap_random_p5": fm["p5"], "merged_overlap_random_p95": fm["p95"],
                "shared_keywords": "; ".join(sorted(tp_full & tl_full)),
                "shared_strict": "; ".join(sorted(full_tr["strict_p"] & full_tr["strict_l"]))}
    odf_all = pd.DataFrame(rows + [full_row])
    odf_all.round(4).to_csv(DATA_DIR / "temporal_keyword_overlap.csv", index=False)
    odf = odf_all[odf_all.tier == "main"].reset_index(drop=True)
    oref = odf_all[odf_all.tier == "reference_only"].reset_index(drop=True)
    pd.DataFrame(topkw).to_csv(DATA_DIR / "temporal_top_keywords.csv", index=False)
    n_in_bins_p = int(odf.n_polycrisis.sum()); n_in_bins_l = int(odf.n_ltg.sum())
    n_placed_p = int(odf.n_polycrisis_all.sum()); n_placed_l = int(odf.n_ltg_all.sum())
    latest = pd.to_datetime(pd.concat([poly["date"], ltg["date"]]).dropna().max())
    latest_txt = f"{latest.strftime('%B')} {latest.day}, {latest.year}"
    print(f"\nPapers placed in a period: polycrisis {n_in_bins_p} of {len(poly)}, ltg {n_in_bins_l} of {len(ltg)}")

    # figure
    x = np.arange(len(odf))
    labels = [f"{r.period}\n(P={r.n_polycrisis}, L={r.n_ltg})" for r in odf.itertuples()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.errorbar(x, odf.overlap_random_mean, yerr=[odf.overlap_random_mean - odf.overlap_random_p5, odf.overlap_random_p95 - odf.overlap_random_mean],
                marker="o", color="#9b6bb7", linewidth=2, capsize=3, label="Overlap, random tie breaking (mean, 5th to 95th)")
    a1.plot(x, odf.overlap_strict, marker="v", color="#e85d4c", linewidth=1.5, label="Overlap, strict set (above cutoff count only)")
    a1.plot(x, odf.overlap_top50, marker="x", color="#aaaaaa", linestyle=":", label="Overlap, alphabetical ties (not used for reading)")
    a1.errorbar(x, odf.matched_overlap_mean, yerr=[odf.matched_overlap_mean - odf.matched_overlap_p5, odf.matched_overlap_p95 - odf.matched_overlap_mean],
                marker="s", color="#777777", linestyle="--", capsize=3, label=f"Size matched ({CONTROL_N} papers per corpus), random ties (mean, 5th to 95th)")
    a1.set_ylabel(f"Keywords shared by both top {TOP_N} lists")
    a1.set_title("Top 50 keyword overlap")
    a1.set_ylim(0, TOP_N)
    a2.plot(x, odf.cosine_per_paper, marker="o", color="#222222", linewidth=2, label="Cosine (per paper normalised)")
    a2.plot(x, odf.weighted_jaccard_share, marker="s", color="#e85d4c", linewidth=2, label="Weighted Jaccard (share of papers)")
    a2.set_ylabel("Similarity (0 to 1)")
    a2.set_title("Similarity of the two vocabularies (no tie cutoff)")
    a2.set_ylim(0, 1)
    for a in (a1, a2):
        a.set_xticks(x)
        a.set_xticklabels(labels, fontsize=8)
        a.legend(frameon=False, fontsize=7, loc="best")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "temporal_keyword_convergence.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── Results note ──────────────────────────────────────────────────────────
    rm = list(odf.overlap_random_mean); cosv = list(odf.cosine_per_paper)
    steps_up = all(b > a for a, b in zip(rm, rm[1:]))
    sep = all(odf.overlap_random_p5.iloc[k + 1] > odf.overlap_random_p95.iloc[k] for k in range(len(odf) - 1))
    cos_up = all(b > a for a, b in zip(cosv, cosv[1:]))
    mm = list(odf.matched_overlap_mean)
    msep = all(odf.matched_overlap_p5.iloc[k + 1] > odf.matched_overlap_p95.iloc[k] for k in range(len(odf) - 1))
    m_up = all(b > a for a, b in zip(mm, mm[1:])) and msep
    mc = list(odf.matched_cos_mean)
    mcsep = all(odf.matched_cos_p5.iloc[k + 1] > odf.matched_cos_p95.iloc[k] for k in range(len(odf) - 1))
    mc_up = all(b > a for a, b in zip(mc, mc[1:])) and mcsep
    if steps_up and sep and cos_up and m_up and mc_up:
        reading = "The tie robust overlap, the size matched overlap and the cosine similarity (observed and size matched) all rise at every step, and the 5th to 95th ranges of consecutive periods do not overlap."
    else:
        reading = ("The tie robust overlap, the size matched overlap and the cosine similarity do not all rise at every step with separated ranges, so the data do not show a steady rise. "
                   "Size matched cosine means: " + ", ".join(f"{r.period} {r.matched_cos_mean:.3f}" for r in odf.itertuples()) + "; size matched weighted Jaccard means: " + ", ".join(f"{r.period} {r.matched_wj_mean:.3f}" for r in odf.itertuples()) + ". "
                   "Tie robust means: " + ", ".join(f"{r.period} {r.overlap_random_mean:.1f}" for r in odf.itertuples())
                   + ". Size matched means: " + ", ".join(f"{r.period} {r.matched_overlap_mean:.1f}" for r in odf.itertuples()) + ".")
    first = odf.iloc[0]
    ctrl_rows = [(r.period, f"{r.overlap_random_mean:.1f}", f"{r.matched_overlap_mean:.1f}", f"{r.matched_overlap_p5:.0f} to {r.matched_overlap_p95:.0f}", f"{r.overlap_random_mean - r.matched_overlap_mean:+.1f}", r.matched_n_per_corpus)
                 for r in odf.itertuples()]
    equal_note = (f"Size matched control: each row draws the same number of papers from each corpus, without replacement ({N_SUBSAMPLES} draws, random tie breaking): {', '.join(f'{int(r.matched_n_per_corpus)} for {r.period}' for r in odf.itertuples())}"
                  + (f"; for the single year rows {', '.join(f'{int(r.matched_n_per_corpus)} for {r.period}' for r in oref.itertuples())}" if len(oref) else "") + f" (the smaller of {CONTROL_N} and the tagged papers available in each corpus). "
                  + "Overlap at that size: " + "; ".join(f"{r.period} {r.matched_overlap_mean:.1f} (5th to 95th {r.matched_overlap_p5:.0f} to {r.matched_overlap_p95:.0f})" for r in odf.itertuples())
                  + ". Size matched cosine: " + "; ".join(f"{r.period} {r.matched_cos_mean:.3f} (5th to 95th {r.matched_cos_p5:.3f} to {r.matched_cos_p95:.3f}) against observed {r.cosine_per_paper:.3f}" for r in odf.itertuples())
                  + ". Size matched weighted Jaccard: " + "; ".join(f"{r.period} {r.matched_wj_mean:.3f} (5th to 95th {r.matched_wj_p5:.3f} to {r.matched_wj_p95:.3f}) against observed {r.weighted_jaccard_share:.3f}" for r in odf.itertuples())
                  + ". Because the size is the same in every period, this series can be compared across periods without the unequal size problem; the observed numbers cannot. "
                  "It is not a correction of the observed numbers. The gap between observed and size matched overlap mixes two things, using all papers instead of "
                  f"{CONTROL_N} and the unequal sizes of the two corpora, and the two cannot be separated here. "
                  "The spread is understated where a draw takes a large share of its corpus, because draws then overlap heavily and the range reflects the draw, not full sampling uncertainty. "
                  "Draws that take at least half of their corpus: " + ("; ".join(f"{int(r.matched_n_per_corpus)} of {int(nn)} {cname} papers in {r.period}" for r in odf.itertuples()
                                                                       for cname, nn in (("polycrisis", r.n_polycrisis), ("LtG", r.n_ltg)) if r.matched_n_per_corpus / nn >= 0.5) or "none") + ".")
    md = []
    md.append("# Results: Temporal Keyword Convergence\n")
    md.append("Generated by `scripts/07_temporal_keywords.py`. Every number below was computed by the script.\n")
    md.append("## Question\n\nIs the vocabulary of the polycrisis literature and the Limits to Growth (LtG) literature converging over time?\n")
    md.append("## Method\n")
    md.append(
        f"* **Keyword source.** The same as the {ov05} of {TOP_N} headline in `scripts/05_simple_viz.py`, except that repeated OpenAlex ids are kept once (counts differ from 05's saved counts for {cmm[45]['deduped polycrisis']} polycrisis and {cmm[45]['deduped ltg']} LtG terms in the first 45, see the consistency check): the `keywords` and `concepts` columns of "
        "`data/polycrisis_papers.csv` and `data/ltg_refined_papers.csv`, lowercased, longer than 2 characters, counted once per paper, "
        "then the stopword list and the drop of any term containing a bracket from 05. These are the OpenAlex keyword and concept tags saved in the CSVs, not author keywords.\n"
        f"* **Consistency check.** For the full period, the top {TOP_N} lists of the two corpora share {ov_full} terms with alphabetical tie breaking, recomputed from the paper CSVs. "
        f"Using the saved frequency files exactly as 05 does gives {ov05}. "
        + ("These agree, and the top 50 sets are identical. " if (ov_full == ov05 and tp_full == r05p and tl_full == r05l) else
           f"These share the count but not the terms: the top 50 sets differ by {len(tp_full - r05p)} polycrisis and {len(tl_full - r05l)} LtG terms per side. " if ov_full == ov05 else
           "These do not agree, and this note reports the paper CSV number for the full period without forcing a match. ")
        + f"Tie sensitivity of this headline: counting only terms above the cutoff count gives {full_tr['strict']}; random tie breaking over {TIE_DRAWS} draws gives a mean of {full_tr['mean']:.1f} "
        f"(5th to 95th percentile {full_tr['p5']:.0f} to {full_tr['p95']:.0f}). So the {ov05} can reasonably be read as roughly {full_tr['p5']:.0f} to {full_tr['p95']:.0f}, not as an exact count. "
        f"Full period ties: polycrisis cutoff count {full_tr['parts_p'][0]} shared by {len(full_tr['parts_p'][2])} terms for {full_tr['parts_p'][3]} open slots; "
        f"LtG cutoff count {full_tr['parts_l'][0]} shared by {len(full_tr['parts_l'][2])} terms for {full_tr['parts_l'][3]} open slots. "
        f"Why this differs from 05's saved counts (checked in code): counting every CSV row reproduces 05's saved counts with {cmm[45]['all rows polycrisis']} polycrisis and {cmm[45]['all rows ltg']} LtG mismatches in the first 45 terms ({cmm[TOP_N]['all rows polycrisis']} and {cmm[TOP_N]['all rows ltg']} in the first {TOP_N}). "
        f"Keeping each OpenAlex id once, as this script does, changes {cmm[45]['deduped polycrisis']} polycrisis and {cmm[45]['deduped ltg']} LtG term counts in the first 45 ({cmm[TOP_N]['deduped polycrisis']} and {cmm[TOP_N]['deduped ltg']} in the first {TOP_N}). "
        f"The cause is the repeated ids in the CSVs ({n_rows_poly} polycrisis rows for {len(poly)} unique ids, {n_rows_ltg} LtG rows for {len(ltg)}), which 05's route counts more than once and this script counts once. "
        f"As a result this script's top 50 lists have {len(tp_full - r05p)} polycrisis and {len(tl_full - r05l)} LtG terms that 05's lists lack, and the reverse; ties at the cutoff add to this. "
        "02 also stores empty keyword cells as the text nan, which 05 drops by accident and this script treats as empty.\n"
        f"* **Periods.** Chosen after seeing the year counts, so every period has about 50 or more papers in both corpora (the script stops if a period has fewer than {MIN_PAPERS}). "
        f"Papers without a year ({int(poly['year'].isna().sum())} polycrisis, {int(ltg['year'].isna().sum())} LtG) are not placed in any period.\n"
        f"* **Measures per period.** Top {TOP_N} overlap is reported three ways: with alphabetical tie breaking (this script's tie rule; 05 breaks ties by first seen order, so the full period count of {ov_full} matches 05's {ov05} only as a count, and the top 50 sets differ by {len(tp_full - r05p)} polycrisis and {len(tl_full - r05l)} LtG terms per side; shown for comparison only), with the strict set (only terms whose count is above the cutoff count), "
        f"and with random tie breaking ({TIE_DRAWS} seeded draws, mean and 5th to 95th percentile). Readings use the random tie breaking numbers. "
        "Weighted Jaccard uses all keywords by share of papers (sum of the smaller share divided by sum of the larger share), and cosine similarity uses keyword vectors in which each paper adds a total weight of 1 spread over its keywords; neither depends on a top 50 cutoff.\n"
        f"* **Size matched control.** Because the corpora differ in size, the script also draws, without replacement, the smaller of {CONTROL_N} and the tagged papers available from each corpus in each row ({N_SUBSAMPLES} draws) and measures the top {TOP_N} overlap with random tie breaking. {CONTROL_N} is below the smallest main period corpus ({int(min(odf.n_polycrisis.min(), odf.n_ltg.min()))} papers). Seed {SEED}.\n"
        f"* **Full period row.** Uses the same rule as the period rows: tagged papers only. That is {n_full_p} polycrisis and {n_full_l} LtG papers, including {n_full_noyear_p} and {n_full_noyear_l} with no year, which the period rows cannot include.\n")
    md.append("## Papers per period\n")
    md.append(md_table(["Period", "Polycrisis papers", "Polycrisis with tags (used)", "LtG papers", "LtG with tags (used)"],
                       [(r.period, r.n_polycrisis_all, r.n_polycrisis, r.n_ltg_all, r.n_ltg) for r in odf.itertuples()]))
    md.append(f"\nPapers placed in a period (have a year, before dropping untagged papers): {n_placed_p} of {len(poly)} polycrisis papers and {n_placed_l} of {len(ltg)} LtG papers. "
              f"Of these, the tagged papers actually used: {n_in_bins_p} polycrisis and {n_in_bins_l} LtG. "
              f"Polycrisis papers up to 2019: {int(((poly.year <= 2019)).sum())}, up to 2022: {int(((poly.year <= 2022)).sum())}, which is why the first period is wide. "
              f"The single years 2023 and 2024 have {', '.join(str(int(r.n_ltg)) + ' tagged LtG papers in ' + r.period for r in oref.itertuples())}, both under 50, so they are shown only for reference. 2026 is a partial year: the latest publication date in either corpus is {latest_txt}, so the 2025 to 2026 period holds a full 2025 and only part of 2026.\n")
    md.append("## Results\n")
    md.append(md_table(["Period", f"Overlap, random ties: mean (5th to 95th)", "Overlap, strict set", "Overlap, alphabetical ties", "Weighted Jaccard", "Cosine"],
                       [(r.period, f"{r.overlap_random_mean:.1f} ({r.overlap_random_p5:.0f} to {r.overlap_random_p95:.0f})", r.overlap_strict, r.overlap_top50,
                         f"{r.weighted_jaccard_share:.3f}", f"{r.cosine_per_paper:.3f}") for r in odf.itertuples()]
                       + [("Full period", f"{full_tr['mean']:.1f} ({full_tr['p5']:.0f} to {full_tr['p95']:.0f})", full_tr["strict"], ov_full, f"{wj_full:.3f}", f"{cos_full:.3f}")]))
    md.append("\n### Ties at the top 50 cutoff\n")
    md.append(md_table(["Period", "Polycrisis: cutoff count", "tied terms", "open slots", "left out", "LtG: cutoff count", "tied terms", "open slots", "left out"],
                       [(r.period, r.cutoff_count_polycrisis, r.tied_polycrisis, r.slots_polycrisis, r.tied_polycrisis - r.slots_polycrisis,
                         r.cutoff_count_ltg, r.tied_ltg, r.slots_ltg, r.tied_ltg - r.slots_ltg) for r in odf.itertuples()]
                       + [("Full period", full_tr["parts_p"][0], len(full_tr["parts_p"][2]), full_tr["parts_p"][3], len(full_tr["parts_p"][2]) - full_tr["parts_p"][3],
                           full_tr["parts_l"][0], len(full_tr["parts_l"][2]), full_tr["parts_l"][3], len(full_tr["parts_l"][2]) - full_tr["parts_l"][3])]))
    md.append("\nWhere many terms share the cutoff count, which of them make the top 50 is arbitrary, so the alphabetical overlap can be too high or too low. The strict and random numbers show how much this matters.\n")
    md.append("### Size matched control\n")
    md.append(md_table(["Period", "Observed overlap (random ties)", "Size matched mean", "Size matched 5th to 95th", "Gap (observed minus matched)", "Papers per corpus in control"], ctrl_rows))
    if len(oref):
        md.append(f"\nSingle years for reference only, not used for the main reading (LtG tagged papers used: {', '.join(str(int(r.n_ltg)) + ' in ' + r.period for r in oref.itertuples())}, under 50, so treat as indicative only):\n")
        md.append(md_table(["Period", "Polycrisis used", "LtG used", "Overlap, random ties: mean (5th to 95th)", "Overlap, strict set", "Weighted Jaccard", "Cosine", "Size matched mean"],
                           [(r.period, r.n_polycrisis, r.n_ltg, f"{r.overlap_random_mean:.1f} ({r.overlap_random_p5:.0f} to {r.overlap_random_p95:.0f})", r.overlap_strict,
                             f"{r.weighted_jaccard_share:.3f}", f"{r.cosine_per_paper:.3f}", f"{r.matched_overlap_mean:.1f} (n = {int(r.matched_n_per_corpus)} per corpus)") for r in oref.itertuples()]))
    md.append("\n### New and shared keywords by period\n")
    md.append("These lists use the strict sets only: terms whose count is above the cutoff count, so tied terms are left out. Terms at the cutoff could move in or out, so a keyword missing here may still be near the edge. The tie table above gives how many terms are tied in each period. "
              "Definition: a term has entered when it is in this period's strict set and was not among the previous period's top 50 candidates, meaning the previous strict set plus every term tied at the previous cutoff. "
              "A term newly shared is in this period's strict shared set and was not in the previous candidates of both corpora. This stops a term that merely sat at the previous cutoff from showing as new.\n")
    kw_rows = []
    for r in odf.itertuples():
        kw_rows.append((r.period, r.shared_strict, r.newly_shared_strict, r.entered_polycrisis_strict, r.entered_ltg_strict))
    md.append(md_table(["Period", "Shared keywords (strict)", "Newly shared since previous period (strict)", "Entered polycrisis top 50 (strict)", "Entered LtG top 50 (strict)"], kw_rows))
    md.append("\nThe full top 50 lists (alphabetical ties) for every period and corpus are in `data/temporal_top_keywords.csv`. The CSV `data/temporal_keyword_overlap.csv` has one row per period and one full period row. The columns shared_keywords, entered_polycrisis_top50, entered_ltg_top50 and newly_shared list terms with alphabetical tie breaking, and the columns shared_strict, entered_polycrisis_strict, entered_ltg_strict and newly_shared_strict list the strict versions used in the table above.\n")
    rmx = int(np.argmax(rm)); rmn = int(np.argmin(rm))
    mrg = list(odf.merged_overlap_random_mean)
    mrg_last = mrg[-1]
    sy = [(r.period, r.overlap_random_mean) for r in oref.itertuples()]      # single year rows, reference only
    mx_i = int(np.argmax(mm))
    sy_txt = ", ".join(f"{pname} {v:.1f}" for pname, v in sy)
    single_vals = [v for _, v in sy] + [rm[-1]]
    plain = (f"In plain terms: the highest tie robust overlap of the two fields' top {TOP_N} keywords is in {odf.period.iloc[rmx]} ({rm[rmx]:.1f} shared keywords), and the lowest is in {odf.period.iloc[rmn]} ({rm[rmn]:.1f}). "
             f"The single year rows give {sy_txt}, and {odf.period.iloc[-1]} gives {rm[-1]:.1f}. "
             f"The highest single year or last period value is {max(single_vals):.1f}, against the pooled {rm[rmx]:.1f}. "
             f"With duplicate records merged by title and first author the period series is {', '.join(f'{x:.1f}' for x in mrg)} instead of {', '.join(f'{x:.1f}' for x in rm)}. "
             f"The single year and last period values differ by at most {max(single_vals) - min(single_vals):.1f} keywords, and the duplicate merge moves the last period by {mrg_last - rm[-1]:+.1f}, so this note makes no claim of a trend in either direction. "
             f"{n_both} papers are in both corpora; each is counted in both corpora's periods and they are not tested separately. "
             "Periods differ a lot in paper counts, so read the details below with the size and tie caveats.")
    md.append("## Plain reading\n")
    md.append(plain + "\n")
    md.append(f"Duplicate record sensitivity: this script keeps one row per OpenAlex id, while 06 finds more records of the same work. Applying the same title and first author rule here removes {n_merged_p} more polycrisis and {n_merged_l} more LtG records "
              f"({n_tagged_merged_p} and {n_tagged_merged_l} of them tagged). Tie robust overlap with those merged: " + "; ".join(f"{r.period} {r.merged_overlap_random_mean:.1f} ({r.merged_overlap_random_p5:.0f} to {r.merged_overlap_random_p95:.0f}) against {r.overlap_random_mean:.1f} unmerged" for r in odf.itertuples())
              + f"; full period {fm['mean']:.1f} ({fm['p5']:.0f} to {fm['p95']:.0f}) against {full_tr['mean']:.1f}. The main tables use the id only rule. "
              f"Which record of a merged work is kept: highest cited_by_count, then most distinct terms, then lowest OpenAlex id. Merged groups: {grp_p} polycrisis and {grp_l} LtG, of which {tied_p} and {tied_l} have the top two records tied on cited_by_count, so the second and third parts of the rule decided them.\n")
    md.append(f"Top {TOP_N} overlap with random tie breaking: " + ", ".join(f"{r.period} {r.overlap_random_mean:.1f} ({r.overlap_random_p5:.0f} to {r.overlap_random_p95:.0f})" for r in odf.itertuples()) + ". "
              f"Strict set overlap: " + ", ".join(f"{r.period} {r.overlap_strict}" for r in odf.itertuples()) + ". "
              f"Cosine: " + ", ".join(f"{r.period} {r.cosine_per_paper:.3f}" for r in odf.itertuples()) + ". "
              + reading + "\n")
    md.append(equal_note + "\n")
    md.append(f"The first period has {int(first.n_polycrisis)} polycrisis papers and its tie robust overlap range is {odf.overlap_random_p5.iloc[0]:.0f} to {odf.overlap_random_p95.iloc[0]:.0f}. "
              f"The LtG side is small in the later periods: " + " and ".join(f"{r.n_ltg} tagged papers in {r.period}" for r in odf.itertuples() if r.n_ltg < 100) + ", "
              f"compared with {int(odf.n_ltg.iloc[0])} in {odf.period.iloc[0]}. The LtG 2023 to 2026 slice is {int(odf.n_ltg.iloc[1:].sum())} tagged papers, "
              f"which is {100 * odf.n_ltg.iloc[1:].sum() / odf.n_ltg.iloc[0]:.1f}% of the earlier LtG bin. This is the main confound when comparing periods, because the size matched ranges show how far a {CONTROL_N} paper sample moves the overlap. "
              f"The widest tie robust 5th to 95th range in any period is {int((odf.overlap_random_p95 - odf.overlap_random_p5).max())} keywords and the widest size matched range is {int((odf.matched_overlap_p95 - odf.matched_overlap_p5).max())}. "
              f"The duplicate record merge also moves the last period: {mrg_last:.1f} instead of {rm[-1]:.1f} ({mrg_last - rm[-1]:+.1f}) in {odf.period.iloc[-1]}, so that period is sensitive to the record rule.\n")
    md.append("## Limitations\n")
    md.append(
        "* **Tag counts differ by period.** The overlap is computed from tags taken as given from the OpenAlex records. In the data, the mean number of distinct terms per tagged paper is "
        + "; ".join(f"{r.period} {r.mean_terms_polycrisis:.1f} polycrisis and {r.mean_terms_ltg:.1f} LtG" for r in odf.itertuples())
        + ", and the share of papers with no tags is " + "; ".join(f"{r.period} {100 * r.untagged_share_polycrisis:.1f}% polycrisis and {100 * r.untagged_share_ltg:.1f}% LtG" for r in odf.itertuples())
        + ". If tagging differs between periods or corpora, a change in overlap may come from tagging as well as from what authors write. This note does not test how the tags were produced.\n"
        f"* **Broad tags.** Shared top keywords can be broad labels instead of shared ideas. The full period strict shared set is: {'; '.join(sorted(full_tr['strict_p'] & full_tr['strict_l']))}. Whether these reflect shared ideas is not tested here.\n"
        f"* **Partial 2026.** The latest publication date in the data is {latest_txt}, so the last period is not a complete two years.\n"
        f"* **Small early bins.** Polycrisis papers: {int((poly.year <= 2019).sum())} up to 2019 and {int((poly.year <= 2022).sum())} up to 2022, out of {len(poly)}, so the earliest polycrisis period is small and merged. "
        f"The first period holds {int(odf.n_ltg_all.iloc[0])} LtG papers published {int(ltg.year[ltg.year <= 2022].min())} to 2022 (median year {ltg.year[ltg.year <= 2022].median():.1f}; {100 * (ltg.year[ltg.year <= 2022] < 2000).mean():.1f}% before 2000), so it mixes many years.\n"
        "* **Unequal sizes.** The two corpora differ in size within each period (see the papers per period table). The size matched control gives a like for like series across periods but does not correct the observed numbers.\n"
        f"* **Ties.** The most terms left out at the cutoff in any period and corpus is {int(max((odf.tied_polycrisis - odf.slots_polycrisis).max(), (odf.tied_ltg - odf.slots_ltg).max()))} (see the tie table), so the alphabetical top 50 overlap is a single draw from a range. Readings here use the random tie breaking range.\n"
        f"* **Corpus scope.** The corpora are the results of keyword queries, so overlap between them describes these retrieved sets, not the whole of either field. {n_both} papers are in both corpora.\n"
        f"* **Truncated tag lists.** Checked against the raw API records ({n_raw_papers} papers): {n_kw_cut} have more than 10 OpenAlex keywords (the most is {n_kw_max}), and the fetch script keeps only the first 10 in the CSV, so those papers lose tags. "
        f"The fetch script also keeps only the 5 highest scoring concepts; {n_co_cut} raw records have more than 5 concepts.\n"
        "* **Untagged papers.** Papers with no keyword or concept tags are left out, and the share of such papers may differ by period and corpus.\n"
        f"* **Duplicate records.** Repeated records of the same work are removed by OpenAlex id only in the main tables. The title and first author rule would remove {n_merged_p} more polycrisis and {n_merged_l} more LtG records; the sensitivity above shows the effect.\n"
        "* **Bin choice.** The period bins were chosen after looking at the year counts, to keep enough papers in both corpora.\n")
    text = "\n".join(md)
    (DOC_DIR / "RESULTS_temporal_keywords.md").write_text(text)
    bad = [l for l in text.splitlines() if not l.startswith("|") and any(ch in re.sub(r"`[^`]*`", "", l) for ch in "-–—")]
    print(f"\nProse lines containing a dash character: {len(bad)}")
    for l in bad:
        print("  DASH:", l[:160])
    print("Banned stem 'explor' in note:", "explor" in text.lower())


if __name__ == "__main__":
    main()
