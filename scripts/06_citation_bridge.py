"""
Citation Bridge: which papers draw on BOTH intellectual traditions?
Project: Polycrisis & Limits to Growth Research
Author: Alex Coffman

What this does:
    1. Builds a "canon" for each corpus from the reference lists of its papers:
       the N works that the most corpus papers cite (N = 50 primary, 25 and 100
       as sensitivity). Works in BOTH top N lists ("shared canon") are removed
       from both canons, so a bridge paper must draw on each tradition's
       distinct base.
    2. Variant 2 builds the canon differently: the N most cited papers inside
       each corpus by cited_by_count (same shared canon exclusion).
    3. A bridge paper is a corpus paper with at least one reference in Canon_P
       AND at least one in Canon_L. The denominator is papers that have a
       reference list, never all papers.
    4. Writes the data files, one figure and docs/RESULTS_citation_bridge.md.
       Every number in the note is computed and printed here.

Reference lists: 06a only fetches the refetched lists. This script takes the
UNION of the refetched list and the list saved in the original raw file for each
paper (it also reports how often the two differ and what other rules would give).

Run AFTER 06a_refetch_references.py (from the repo root):
    .venv/bin/python scripts/06_citation_bridge.py

Output:
    data/canon_polycrisis_N50.csv, data/canon_ltg_N50.csv
    data/bridge_papers.csv, data/bridge_sensitivity.csv, data/bridge_tie_sensitivity.csv
    data/canon_metadata_cache.json (titles, years and types of cited works; written on the
    first network run, then read offline)
    figures/bridge_papers_by_year.png
    docs/RESULTS_citation_bridge.md
Network: one batched call group to api.openalex.org (titles of canon works).
No email or mailto is sent.
"""

import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
DOC_DIR = ROOT / "docs"
FIG_DIR.mkdir(exist_ok=True)
DOC_DIR.mkdir(exist_ok=True)

PRIMARY_N = 50
SENS_NS = [25, 50, 100]
FIG_START_YEAR = 2000      # figure and trend table start here (see note)
MIN_YEAR_DENOM = 10        # years with fewer referencing papers are drawn hollow
API = "https://api.openalex.org/works"


def short_id(x):
    return str(x).rsplit("/", 1)[-1]


# ── Load ──────────────────────────────────────────────────────────────────────

def load():
    """Return meta (DataFrame by id), corpus membership sets, union refs dict, old rule refs dict, list difference counts."""
    frames, members = [], {}
    for name, csv in [("polycrisis", "polycrisis_papers.csv"), ("ltg", "ltg_refined_papers.csv")]:
        df = pd.read_csv(DATA_DIR / csv)
        df["openalex_id"] = df["openalex_id"].map(short_id)
        members[name] = set(df["openalex_id"])
        frames.append(df[["openalex_id", "title", "year", "date", "cited_by_count", "type", "authors"]])
    meta = pd.concat(frames).drop_duplicates("openalex_id").set_index("openalex_id")

    orig = {}
    for f in ["polycrisis_papers_raw.json", "ltg_refined_papers_raw.json"]:
        for w in json.load(open(DATA_DIR / f)):
            wid = short_id(w["id"])
            refs = [short_id(r) for r in (w.get("referenced_works") or [])]
            if len(refs) > len(orig.get(wid, [])):
                orig[wid] = refs
            orig.setdefault(wid, refs)
    new = json.load(open(DATA_DIR / "references_refetch_2026-10-24.json"))
    refs, old_refs, saved_only, longer = {}, {}, {}, {}
    st = Counter()
    for wid in meta.index:
        r = (new.get(wid, {}).get("referenced_works") or [])
        o = orig.get(wid, [])
        refs[wid] = list(dict.fromkeys(r + o))          # union of refetched and saved lists
        old_refs[wid] = list(dict.fromkeys(r or o))     # previous rule: refetched if nonempty, else saved
        saved_only[wid] = list(dict.fromkeys(o))
        longer[wid] = list(dict.fromkeys(max([r, o], key=len)))
        if r and o:
            sr, so = set(r), set(o)
            st["both lists nonempty"] += 1
            if sr == so:
                st["same set"] += 1
            elif sr < so:
                st["refetched shorter (subset of saved)"] += 1
            elif so < sr:
                st["refetched longer (superset of saved)"] += 1
            else:
                st["different (neither contains the other)"] += 1
        elif r:
            st["only refetched nonempty"] += 1
        elif o:
            st["only saved nonempty"] += 1
        else:
            st["both empty"] += 1
    return meta, members, refs, {"previous rule (refetched list if nonempty, else saved list)": old_refs, "saved list only": saved_only, "longer of the two lists": longer}, st


# ── Canon builders ────────────────────────────────────────────────────────────

def top_n(counter, n):
    """Top n by count, ties broken by id for a stable result. Also return the
    count at the cutoff and how many works tie at that count outside the list."""
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    top = ranked[:n]
    cut = top[-1][1] if top else 0
    extra_ties = sum(1 for _, c in ranked[n:] if c == cut)
    return top, cut, extra_ties


def ref_counts(ids, refs):
    c = Counter()
    for wid in ids:
        c.update(refs[wid])
    return c


def build_canons(variant, n, meta, members, refs):
    if variant == "refs":
        cp, cl = ref_counts(members["polycrisis"], refs), ref_counts(members["ltg"], refs)
    else:  # cited
        cp = {w: int(meta.loc[w, "cited_by_count"]) for w in members["polycrisis"]}
        cl = {w: int(meta.loc[w, "cited_by_count"]) for w in members["ltg"]}
    tp, cutp, tiep = top_n(cp, n)
    tl, cutl, tiel = top_n(cl, n)
    setp, setl = {w for w, _ in tp}, {w for w, _ in tl}
    shared = setp & setl
    return {
        "top_p": tp, "top_l": tl, "shared": shared,
        "eff_p": setp - shared, "eff_l": setl - shared,
        "cut": (cutp, cutl), "ties": (tiep, tiel),
        "counts_p": dict(cp), "counts_l": dict(cl),
    }


def find_bridges(canon, refs, ids_with_refs, min_refs=1):
    rows = []
    for wid in ids_with_refs:
        r = set(refs[wid])
        a, b = len(r & canon["eff_p"]), len(r & canon["eff_l"])
        if a >= min_refs and b >= min_refs:
            rows.append((wid, a, b))
    return rows


# ── OpenAlex title lookup ─────────────────────────────────────────────────────

CACHE_PATH = DATA_DIR / "canon_metadata_cache.json"


def resolve_titles(ids):
    """Title, year and type for canon works. Reads data/canon_metadata_cache.json
    first and only calls OpenAlex for ids missing from it, then updates the cache,
    so a rerun with a full cache needs no network."""
    cache = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}
    need = sorted(i for i in ids if i not in cache)
    print(f"Canon metadata: {len(ids) - len(need)} of {len(ids)} ids from cache, {len(need)} to fetch")
    for i in range(0, len(need), 50):
        chunk = need[i:i + 50]
        params = {"filter": "openalex_id:" + "|".join(chunk),
                  "select": "id,title,publication_year,type", "per-page": 50}
        for attempt in range(4):
            r = requests.get(API, params=params, timeout=60)
            if r.status_code == 200:
                break
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
        else:
            raise RuntimeError("title lookup failed")
        got = {short_id(w["id"]): [w.get("title"), w.get("publication_year"), w.get("type")]
               for w in r.json().get("results", [])}
        for wid in chunk:
            cache[wid] = got.get(wid, [None, None, None])   # unresolved ids cached as empty
        time.sleep(0.2)
    if need:
        json.dump(cache, open(CACHE_PATH, "w"))
    return {i: tuple(cache[i]) for i in ids}


# ── Markdown helpers (no dashes in prose; table markup is the only exception) ─

def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join([":--"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    return "\n".join(out)


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


# ── Record merge (same work under several OpenAlex ids) ───────────────────────

MIN_TITLE = 12     # normalised titles shorter than this are never merged
MIN_POOL_COUNT = 3  # works cited by fewer corpus papers than this are not checked for duplicates
NULL_DRAWS = 1000
TIE_BRIDGE_DRAWS = 500
SEED = 20261024


def norm_title(t):
    t = unicodedata.normalize("NFKD", str(t or "")).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def first_surname(a):
    if not isinstance(a, str) or not a.strip():
        return ""
    first = a.split(";")[0].split("(")[0].strip().split()
    return norm_title(first[-1]) if first else ""


def build_gmap(raw_meta, pool_info, ucount):
    """Union ids that are the same work. Returns gmap {id: representative id},
    comps {rep: sorted members}, rules {rep: set of rule names}."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    rule_of = defaultdict(set)   # pair tag collected per id, resolved to rep below
    # (a) papers: same normalised title and same first author surname
    keyp = defaultdict(list)
    for wid, row in raw_meta.iterrows():
        nt, sn = norm_title(row["title"]), first_surname(row["authors"])
        if len(nt) >= MIN_TITLE and sn:
            keyp[(nt, sn)].append(wid)
    for ids in keyp.values():
        for other in ids[1:]:
            union(ids[0], other)
            rule_of[other].add("paper title and first author")
            rule_of[ids[0]].add("paper title and first author")
    # (b) cited works: same normalised full title
    keyw = defaultdict(list)
    stems = defaultdict(set)
    for wid in sorted(pool_info):
        t = pool_info[wid][0]
        nt = norm_title(t)
        if len(nt) >= MIN_TITLE:
            keyw[nt].append(wid)
            if ":" in str(t):
                st = norm_title(str(t).split(":", 1)[0])
                if len(st) >= MIN_TITLE:
                    stems[st].add(nt)
    for ids in keyw.values():
        for other in ids[1:]:
            union(ids[0], other)
            rule_of[other].add("same normalised title")
            rule_of[ids[0]].add("same normalised title")
    # (c) a bare title equal to the part before the colon of longer titles joins the most cited such group
    for nt, ids in sorted(keyw.items()):
        if nt in stems:   # a title equal to the part before the colon of longer titles
            cands = sorted(stems[nt], key=lambda k: (-sum(ucount.get(i, 0) for i in keyw[k]), k))
            if cands and cands[0] != nt:
                union(ids[0], keyw[cands[0]][0])
                for i in ids + keyw[cands[0]][:1]:
                    rule_of[i].add(f"bare title joined the most cited of {len(cands)} longer title group(s)")
    comp = defaultdict(list)
    for x in list(parent):
        comp[find(x)].append(x)
    gmap, comps, rules = {}, {}, {}
    for ms in comp.values():
        ms = sorted(ms)
        if len(ms) < 2:
            continue
        rep = sorted(ms, key=lambda i: (-ucount.get(i, 0), i))[0]
        comps[rep] = ms
        rules[rep] = set().union(*[rule_of.get(i, set()) for i in ms])
        for i in ms:
            gmap[i] = rep
    return gmap, comps, rules


def make_context(raw_meta, raw_members, raw_refs, gmap):
    g = lambda x: gmap.get(x, x)
    unit_members = defaultdict(list)
    for p in raw_meta.index:
        unit_members[g(p)].append(p)
    members = {c: {g(p) for p in ids} for c, ids in raw_members.items()}
    refs, rows = {}, {}
    for u in sorted(unit_members):
        ps = sorted(unit_members[u], key=lambda p: (-int(raw_meta.loc[p, "cited_by_count"]), p))
        allr = [g(x) for p in ps for x in raw_refs[p]]
        refs[u] = [x for x in dict.fromkeys(allr) if x != u]
        b = raw_meta.loc[ps[0]]
        rows[u] = {"title": b["title"], "year": b["year"], "date": b["date"], "cited_by_count": int(b["cited_by_count"]),
                   "type": b["type"], "paper_id": ps[0], "merged_ids": "; ".join(sorted(ps))}
    meta = pd.DataFrame.from_dict(rows, orient="index")
    return meta, members, refs


def null_bridge(refs, groups, canon, rng, draws):
    """groups: list of (target units, pool units). Each target keeps its own number of references
    and draws that many DISTINCT references at random from its pool's references, weighted by how
    many pool papers cite each work (repeats are redrawn). Returns bridge-count statistics."""
    allu = sorted({u for t, p in groups for u in list(t) + list(p)})
    codes = {r: i for i, r in enumerate(sorted({r for u in allu for r in refs[u]}))}
    C = len(codes)
    isP = np.zeros(C, dtype=int)
    isL = np.zeros(C, dtype=int)
    for w in canon["eff_p"]:
        if w in codes:
            isP[codes[w]] = 1
    for w in canon["eff_l"]:
        if w in codes:
            isL[codes[w]] = 1
    prep = []
    for targets, pool_units in groups:
        targets = sorted(targets)
        if not targets:
            continue
        pool = np.array([codes[r] for u in sorted(pool_units) for r in refs[u]])
        lens = np.array([len(refs[u]) for u in targets])
        offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
        prep.append((pool, offs, np.repeat(np.arange(len(targets)), lens)))
    res = {1: [], 2: []}
    for _ in range(draws):
        tot = {1: 0, 2: 0}
        for pool, offs, pidx in prep:
            d = pool[rng.integers(0, len(pool), size=len(pidx))]
            while True:
                _, first = np.unique(pidx * C + d, return_index=True)
                dup = np.ones(len(d), dtype=bool)
                dup[first] = False
                nd = int(dup.sum())
                if nd == 0:
                    break
                d[dup] = pool[rng.integers(0, len(pool), size=nd)]
            a, b = np.add.reduceat(isP[d], offs), np.add.reduceat(isL[d], offs)
            for m in (1, 2):
                tot[m] += int(((a >= m) & (b >= m)).sum())
        for m in (1, 2):
            res[m].append(tot[m])
    return {m: (float(np.mean(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95)), int(max(v))) for m, v in res.items()}


def tie_bridge(canon, n, refs, with_refs, rng, draws):
    """Bridge counts when the canon cutoff ties are handled (a) strictly (only works above the
    cutoff count) and (b) by random tie breaking, instead of by id order."""
    parts = {}
    for side in ("p", "l"):
        cnt = canon["counts_" + side]
        srt = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
        cut = srt[n - 1][1]
        strict = {w for w, c in srt if c > cut}
        tied = sorted(w for w, c in srt if c == cut)
        parts[side] = (cut, strict, tied, n - len(strict))
    def counts(setp, setl):
        sh = setp & setl
        ep, el = setp - sh, setl - sh
        res = {1: 0, 2: 0}
        for wid in sorted(with_refs):
            r = set(refs[wid])
            a, b = len(r & ep), len(r & el)
            for m in (1, 2):
                if a >= m and b >= m:
                    res[m] += 1
        return res
    st = counts(parts["p"][1], parts["l"][1])
    draws_res = {1: [], 2: []}
    for _ in range(draws):
        sets = []
        for side in ("p", "l"):
            cut, strict, tied, slots = parts[side]
            pick = rng.choice(len(tied), slots, replace=False)
            sets.append(strict | {tied[i] for i in pick})
        rr = counts(sets[0], sets[1])
        for m in (1, 2):
            draws_res[m].append(rr[m])
    return parts, st, {m: (float(np.mean(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95))) for m, v in draws_res.items()}


def _yr(y):
    return "" if y is None or (isinstance(y, float) and np.isnan(y)) else str(int(y))


def sens_grid(meta, members, refs, with_refs, records):
    rows, canons = [], {}
    for variant, label in [("refs", "most referenced works"), ("cited", "most cited papers in corpus")]:
        for n in SENS_NS:
            c = build_canons(variant, n, meta, members, refs)
            canons[(variant, n)] = c
            variants = [(label, c)]
            if variant == "refs":
                kc = dict(c, eff_p={w for w, _ in c["top_p"]}, eff_l={w for w, _ in c["top_l"]})   # shared works stay in both canons
                canons[("refs_keep", n)] = kc
                variants.append((label + ", shared canon kept", kc))
            for lab_, cc in variants:
                for mr in (1, 2):
                    br = find_bridges(cc, refs, with_refs, mr)
                    rows.append({"records": records, "variant": lab_, "min_refs_each_canon": mr, "N": n,
                                 "canon_P_size": len(cc["eff_p"]), "canon_L_size": len(cc["eff_l"]),
                                 "shared_canon_size": len(c["shared"]), "bridge_count": len(br),
                                 "denominator": len(with_refs), "share": round(len(br) / len(with_refs), 4)})
    return rows, canons


def headline(raw_meta, raw_members, raw_refs, merged=True):
    """Bridge counts at the primary N for a given set of reference lists (unmerged and merged records)."""
    out = {}
    all0 = set(raw_meta.index)
    with0 = {w for w in all0 if raw_refs[w]}
    c0 = build_canons("refs", PRIMARY_N, raw_meta, raw_members, raw_refs)
    for m in (1, 2):
        out[("unmerged", m)] = len(find_bridges(c0, raw_refs, with0, m))
    out[("unmerged", "den")] = len(with0)
    if not merged:
        return out
    pc, lc = ref_counts(raw_members["polycrisis"], raw_refs), ref_counts(raw_members["ltg"], raw_refs)
    uc = Counter({w: pc[w] + lc[w] for w in set(pc) | set(lc)})
    pool = sorted(w for w in set(pc) | set(lc) if pc[w] >= MIN_POOL_COUNT or lc[w] >= MIN_POOL_COUNT)
    gm, _, _ = build_gmap(raw_meta, resolve_titles(pool), uc)
    mt, mm, mr = make_context(raw_meta, raw_members, raw_refs, gm)
    wr = {w for w in mt.index if mr[w]}
    cm = build_canons("refs", PRIMARY_N, mt, mm, mr)
    for m in (1, 2):
        out[("merged", m)] = len(find_bridges(cm, mr, wr, m))
    out[("merged", "den")] = len(wr)
    return out


def main():
    raw_meta, raw_members, raw_refs, alt_refs, list_stats = load()
    old_refs_raw = alt_refs["previous rule (refetched list if nonempty, else saved list)"]
    print('Reference list comparison (saved vs refetched):', dict(list_stats))
    _ld = pd.to_datetime(str(raw_meta["date"].dropna().max()))
    latest_date = f"{_ld.strftime('%B')} {_ld.day}, {_ld.year}"
    # ── unmerged records (reference run) ──
    all0 = set(raw_meta.index)
    with0 = {w for w in all0 if raw_refs[w]}
    rows0, canons0 = sens_grid(raw_meta, raw_members, raw_refs, with0, "unmerged")
    ucount = Counter()
    pcount = ref_counts(raw_members["polycrisis"], raw_refs)
    lcount = ref_counts(raw_members["ltg"], raw_refs)
    for w in set(pcount) | set(lcount):
        ucount[w] = pcount[w] + lcount[w]
    pool_ids = sorted({w for w in set(pcount) | set(lcount) if pcount[w] >= MIN_POOL_COUNT or lcount[w] >= MIN_POOL_COUNT})
    print(f"Works cited by at least {MIN_POOL_COUNT} papers in either corpus (checked for duplicate records): {len(pool_ids)}")
    pool_info = resolve_titles(pool_ids)
    gmap, comps, rules = build_gmap(raw_meta, pool_info, ucount)
    print(f"Merged groups: {len(comps)} (ids merged: {len(gmap)})")

    # ── merged records (primary) ──
    meta, members, refs = make_context(raw_meta, raw_members, raw_refs, gmap)
    both = members["polycrisis"] & members["ltg"]
    all_ids = set(meta.index)
    with_refs = {w for w in all_ids if refs[w]}
    print(f"Units after merge: {len(all_ids)} (unmerged ids {len(all0)}); polycrisis {len(members['polycrisis'])}, "
          f"ltg {len(members['ltg'])}, in both {len(both)}")
    print(f"Papers with a reference list (denominator): merged {len(with_refs)}, unmerged {len(with0)}")

    # coverage table (merged units)
    cov_rows = []
    for name in ["polycrisis", "ltg"]:
        ids = members[name]
        wr = sorted(w for w in ids if refs[w])
        tot = sum(len(refs[w]) for w in wr)
        uniq = len({r for w in wr for r in refs[w]})
        cov_rows.append((name, len(ids), len(wr), pct(len(wr), len(ids)), tot, f"{tot / len(wr):.1f}", uniq))
    cov_rows.append(("either (deduped)", len(all_ids), len(with_refs), pct(len(with_refs), len(all_ids)),
                     sum(len(refs[w]) for w in with_refs), f"{sum(len(refs[w]) for w in with_refs) / len(with_refs):.1f}",
                     len({r for w in with_refs for r in refs[w]})))
    for r in cov_rows:
        print("coverage", r)

    rows1, canons = sens_grid(meta, members, refs, with_refs, "merged")
    sens_df = pd.DataFrame(rows1 + rows0)
    sens_df.to_csv(DATA_DIR / "bridge_sensitivity.csv", index=False)
    old_head = headline(raw_meta, raw_members, old_refs_raw)
    print("Old reference rule headline:", old_head)
    alt_heads = {k: headline(raw_meta, raw_members, v, merged=False) for k, v in alt_refs.items() if not k.startswith("previous")}
    print("Alternative reference rules (unmerged):", alt_heads)
    n_changed_lists = sum(1 for w in all0 if set(old_refs_raw[w]) != set(raw_refs[w]))
    print(f"Papers whose reference list changes under the union rule: {n_changed_lists}")
    for r in rows1 + rows0:
        print("sens", r["records"], r["variant"], r["min_refs_each_canon"], r["N"], "bridges", r["bridge_count"], "of", r["denominator"], "shared", r["shared_canon_size"])

    # primary (merged)
    prim = canons[("refs", PRIMARY_N)]
    prim0 = canons0[("refs", PRIMARY_N)]
    bridges = find_bridges(prim, refs, with_refs)
    bridges0 = find_bridges(prim0, raw_refs, with0)

    # null model
    rng = np.random.default_rng(SEED)
    null_rows, null_rows_pooled = [], []
    wp = with_refs & members["polycrisis"]
    wl = with_refs & members["ltg"]
    grp_within = [(wp - wl, wp), (wl - wp, wl), (wp & wl, wp | wl)]
    grp_pooled = [(with_refs, with_refs)]
    print(f"Within corpus null groups: polycrisis only {len(wp - wl)}, ltg only {len(wl - wp)}, in both {len(wp & wl)}; tested papers in neither (should be 0): {len(with_refs - wp - wl)}")
    for n in SENS_NS:
        cn = canons[("refs", n)]
        nw = null_bridge(refs, grp_within, cn, rng, NULL_DRAWS)
        npool = null_bridge(refs, grp_pooled, cn, rng, NULL_DRAWS)
        for m in (1, 2):
            obs = len(find_bridges(cn, refs, with_refs, m))
            null_rows.append((n, m, obs, nw[m][0], nw[m][1], nw[m][2], nw[m][3]))
            null_rows_pooled.append((n, m, obs, npool[m][0], npool[m][1], npool[m][2], npool[m][3]))
            print(f"null N={n} min_refs={m}: observed {obs}; within corpus mean {nw[m][0]:.1f} (5th to 95th {nw[m][1]:.0f} to {nw[m][2]:.0f}), max {nw[m][3]}; pooled mean {npool[m][0]:.1f} ({npool[m][1]:.0f} to {npool[m][2]:.0f})")

    # within corpus null for the variant that keeps the shared canon (own rng so earlier numbers do not move)
    rng_k = np.random.default_rng(SEED + 3)
    null_keep = []
    for n in SENS_NS:
        ck = canons[("refs_keep", n)]
        nk = null_bridge(refs, grp_within, ck, rng_k, NULL_DRAWS)
        for m in (1, 2):
            obs = len(find_bridges(ck, refs, with_refs, m))
            null_keep.append((n, m, obs, nk[m][0], nk[m][1], nk[m][2], nk[m][3]))
            print(f"null (shared canon kept) N={n} min_refs={m}: observed {obs}; within corpus mean {nk[m][0]:.1f} ({nk[m][1]:.0f} to {nk[m][2]:.0f})")

    # canon cutoff ties: strict canon and random tie breaking (merged primary)
    tie_rows = []
    rng_t = np.random.default_rng(SEED + 1)
    for n in SENS_NS:
        cn = canons[("refs", n)]
        parts, st, rnd = tie_bridge(cn, n, refs, with_refs, rng_t, TIE_BRIDGE_DRAWS)
        for m in (1, 2):
            obs = len(find_bridges(cn, refs, with_refs, m))
            tie_rows.append({"N": n, "min_refs_each_canon": m, "bridges_id_order": obs, "bridges_strict_canon": st[m],
                             "bridges_random_ties_mean": rnd[m][0], "bridges_random_ties_p5": rnd[m][1], "bridges_random_ties_p95": rnd[m][2],
                             "cutoff_count_polycrisis": parts["p"][0], "tied_polycrisis": len(parts["p"][2]), "open_slots_polycrisis": parts["p"][3],
                             "cutoff_count_ltg": parts["l"][0], "tied_ltg": len(parts["l"][2]), "open_slots_ltg": parts["l"][3], "denominator": len(with_refs)})
            print("tie bridge", tie_rows[-1])
    pd.DataFrame(tie_rows).round(4).to_csv(DATA_DIR / "bridge_tie_sensitivity.csv", index=False)

    # titles for canon works
    canon_ids = sorted({w for w, _ in prim["top_p"]} | {w for w, _ in prim["top_l"]})
    info_rep = resolve_titles(canon_ids)

    def _disp(w):
        # a merged work is shown with the earliest year among its merged records and the type of that earliest record
        cands = []
        for m in comps.get(w, [w]):
            rec = info_rep.get(m) or pool_info.get(m)
            if rec and rec[1]:
                cands.append((rec[1], m, rec[2]))
        if not cands:
            return info_rep[w]
        y, _m, ty = min(cands)
        return (info_rep[w][0], y, ty)
    info = {w: _disp(w) for w in canon_ids}
    n_title_missing = sum(1 for w in canon_ids if not info.get(w, (None,))[0])
    print(f"Resolved titles for {len(canon_ids) - n_title_missing} of {len(canon_ids)} canon works")

    def yrs(ws):
        return sorted(info[w][1] for w in ws if info.get(w, (None, None))[1])
    yp, yl = yrs(prim["eff_p"]), yrs(prim["eff_l"])
    ypool = sorted(yp + yl)

    def med(v):
        return float(np.median(v)) if v else float("nan")
    print(f"Canon_P work years: min {min(yp)}, median {med(yp):.1f}, max {max(yp)} (n={len(yp)} of {len(prim['eff_p'])} with a year)")
    print(f"Canon_L work years: min {min(yl)}, median {med(yl):.1f}, max {max(yl)} (n={len(yl)} of {len(prim['eff_l'])} with a year)")
    Y_MED = int(np.ceil(med(ypool)))
    Y_MAX = max(ypool)
    print(f"Pooled canon work years: median {med(ypool):.1f}, newest {Y_MAX}")
    canon_types = sorted(Counter((info[w][2] or "unresolved") for w in canon_ids).items(), key=lambda kv: (-kv[1], kv[0]))
    print("Canon work types (top lists before shared removal):", canon_types)

    def members_of(w):
        return comps.get(w, [w])

    def canon_df(top, eff):
        rows = []
        for rank, (w, c) in enumerate(top, 1):
            t, y, ty = info.get(w, (None, None, None))
            rows.append({"rank": rank, "work_id": w, "citing_corpus_papers": c, "title": t,
                         "year": y, "type": ty, "representative_year": info_rep[w][1], "representative_type": info_rep[w][2], "shared_canon": w in prim["shared"],
                         "in_bridge_canon": w in eff, "merged_record_ids": "; ".join(members_of(w))})
        return pd.DataFrame(rows)

    canon_df(prim["top_p"], prim["eff_p"]).to_csv(DATA_DIR / "canon_polycrisis_N50.csv", index=False)
    canon_df(prim["top_l"], prim["eff_l"]).to_csv(DATA_DIR / "canon_ltg_N50.csv", index=False)

    # bridge papers
    brows = []
    for wid, a, b in bridges:
        m = meta.loc[wid]
        memb = "both" if wid in both else ("polycrisis" if wid in members["polycrisis"] else "ltg")
        brows.append({"openalex_id": wid, "paper_id": m["paper_id"], "merged_ids": m["merged_ids"], "title": m["title"], "year": m["year"],
                      "corpus": memb, "n_refs_canon_polycrisis": a, "n_refs_canon_ltg": b, "cited_by_count": int(m["cited_by_count"])})
    bdf = pd.DataFrame(brows).sort_values(["cited_by_count", "openalex_id"], ascending=[False, True])
    bdf.to_csv(DATA_DIR / "bridge_papers.csv", index=False)
    nb_corp = sorted(bdf["corpus"].value_counts().to_dict().items())
    print(f"Primary bridge papers (merged): {len(bdf)}; unmerged: {len(bridges0)}; by corpus: {nb_corp}")

    # per-year trend
    yrs_all = meta["year"].dropna().astype(int)
    bridge_ids = set(bdf["openalex_id"])
    trend = []
    for y in range(FIG_START_YEAR, int(yrs_all.max()) + 1):
        ids_y = [w for w in sorted(with_refs) if pd.notna(meta.loc[w, "year"]) and int(meta.loc[w, "year"]) == y]
        nb = sum(1 for w in ids_y if w in bridge_ids)
        trend.append({"year": y, "referencing_papers": len(ids_y), "bridge_papers": nb,
                      "share": nb / len(ids_y) if ids_y else float("nan")})
    tdf = pd.DataFrame(trend)
    print(tdf.to_string(index=False))
    pre = [w for w in with_refs if pd.notna(meta.loc[w, "year"]) and int(meta.loc[w, "year"]) < FIG_START_YEAR]
    pre_b = [w for w in pre if w in bridge_ids]
    noyear = [w for w in with_refs if pd.isna(meta.loc[w, "year"])]
    print(f"Before {FIG_START_YEAR}: {len(pre)} referencing papers, {len(pre_b)} bridges; no year: {len(noyear)}; latest publication date in data: {latest_date}")

    periods = [("2000 to 2014", 2000, 2014), ("2015 to 2019", 2015, 2019), ("2020 to 2022", 2020, 2022),
               ("2023 to 2024", 2023, 2024), ("2025 to 2026", 2025, 2026)]
    prow = []
    for lab, a, b in periods:
        sub = tdf[(tdf.year >= a) & (tdf.year <= b)]
        d, n_ = int(sub.referencing_papers.sum()), int(sub.bridge_papers.sum())
        prow.append((lab, d, n_, pct(n_, d)))
        print("period", prow[-1])

    def restricted(y0):
        sub = tdf[tdf.year >= y0]
        d, n_ = int(sub.referencing_papers.sum()), int(sub.bridge_papers.sum())
        return y0, d, n_, pct(n_, d)
    Y_MID = int(max(med(yp), med(yl)))
    r_med, r_max, r_mid = restricted(Y_MED), restricted(Y_MAX), restricted(Y_MID)
    print("restricted:", r_med, r_mid, r_max)
    rsub = tdf[tdf.year >= Y_MED]
    rq = rsub[rsub.referencing_papers >= MIN_YEAR_DENOM]
    rshares = [float(x) for x in rq.share]
    r_rise = len(rshares) >= 3 and all(b >= a for a, b in zip(rshares, rshares[1:]))
    tcov = []
    for t, g in meta.groupby(meta["type"].fillna("unknown")):
        tcov.append((t, len(g), int(sum(1 for w in g.index if refs[w]))))
    tcov.sort(key=lambda r: (-r[1], r[0]))
    canon_is_bridge = bridge_ids & (prim["eff_p"] | prim["eff_l"])
    canon_shared_bridge = bridge_ids & prim["shared"]
    print(f"Bridge papers that are themselves canon works: {len(canon_is_bridge)} (shared canon: {len(canon_shared_bridge)})")

    # figure
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(tdf.year, tdf.bridge_papers, color="#9b6bb7", alpha=0.85, label="Bridge papers (count)")
    ax.set_xlabel(f"Publication year (2026 is partial, latest date in data {latest_date})")
    ax.set_ylabel("Bridge papers")
    ax2 = ax.twinx()
    solid = tdf[tdf.referencing_papers >= MIN_YEAR_DENOM]
    thin = tdf[tdf.referencing_papers < MIN_YEAR_DENOM]
    ax2.plot(solid.year, solid.share * 100, color="#222222", marker="o", linewidth=1.8, label="Share of papers with references (%)")
    ax2.scatter(thin.year, thin.share * 100, facecolors="none", edgecolors="#222222",
                label=f"Share, fewer than {MIN_YEAR_DENOM} such papers that year")
    ax2.set_ylabel("Share of referencing papers (%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9)
    ax.set_title(f"Bridge papers per year (N={PRIMARY_N} canon, merged records, {FIG_START_YEAR} onward)")
    for a_ in (ax, ax2):
        a_.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bridge_papers_by_year.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── Results note ──────────────────────────────────────────────────────────
    top15 = bdf.head(15)
    cut15 = int(bdf.cited_by_count.iloc[min(14, len(bdf) - 1)])
    top15 = bdf[bdf.cited_by_count >= cut15]       # includes every row tied with rank 15
    n_tied15 = int((bdf.cited_by_count == cut15).sum())
    shared_rows = []
    for w in sorted(prim["shared"], key=lambda w: (-(prim["counts_p"].get(w, 0) + prim["counts_l"].get(w, 0)), w)):
        t, y, ty = info.get(w, (None, None, None))
        shared_rows.append((w, t, y, prim["counts_p"].get(w, 0), prim["counts_l"].get(w, 0), "; ".join(members_of(w))))
    c50 = prim
    t50_1 = [r for r in tie_rows if r["N"] == PRIMARY_N and r["min_refs_each_canon"] == 1][0]
    sens_rows = [(r.records, r.variant, r.min_refs_each_canon, r.N, r.shared_canon_size, r.canon_P_size, r.canon_L_size,
                  r.bridge_count, r.denominator, pct(r.bridge_count, r.denominator))
                 for r in sens_df.assign(_o=sens_df.variant.map({"most referenced works": 0, "most referenced works, shared canon kept": 1, "most cited papers in corpus": 2}))
                 .sort_values(["records", "_o", "min_refs_each_canon", "N"]).itertuples()]
    tc = {t: (n, k) for t, n, k in tcov}
    bk, ch = tc.get("book", (0, 0)), tc.get("book-chapter", (0, 0))
    books_chap_text = (f"Books themselves rarely have reference lists: `book` {bk[1]} of {bk[0]} papers have one ({pct(bk[1], bk[0])}), and books are outside the denominator for the most part. "
                       f"Chapters are more often covered: `book-chapter` {ch[1]} of {ch[0]} ({pct(ch[1], ch[0])}). Even so, {ch[0] - ch[1]} chapters have no list and cannot be tested.")
    two50 = sens_df[(sens_df.records == "merged") & (sens_df.variant == "most referenced works") & (sens_df.N == PRIMARY_N) & (sens_df.min_refs_each_canon == 2)].bridge_count.iloc[0]
    two50_0 = sens_df[(sens_df.records == "unmerged") & (sens_df.variant == "most referenced works") & (sens_df.N == PRIMARY_N) & (sens_df.min_refs_each_canon == 2)].bridge_count.iloc[0]
    one50 = len(bdf)
    first_bridge_year = int(bdf["year"].min()) if len(bdf) else None
    peak = max(prow, key=lambda r: (int(r[2]) / r[1]) if r[1] else 0)
    refetch = json.load(open(DATA_DIR / "references_refetch_2026-10-24.json"))
    n_not_refetched = sum(1 for w in all0 if w not in refetch)
    _fd = sorted({v["fetched"][:10] for v in refetch.values()})
    _fmt = lambda d: (lambda t: f"{t.strftime('%B')} {t.day}, {t.year}")(pd.to_datetime(d))
    refetch_dates = _fmt(_fd[0]) if len(_fd) == 1 else f"{_fmt(_fd[0])} to {_fmt(_fd[-1])}"
    # merged groups relevant to the canon and to bridge papers
    canon_group_rows = []
    for rep in sorted(set(canon_ids)):
        if rep in comps:
            ms = comps[rep]
            canon_group_rows.append((rep, len(ms), "; ".join(f"{m} ({_yr((info_rep.get(m) or pool_info.get(m) or (None, None, None))[1])})" for m in ms),
                                     (info_rep.get(rep, (None,))[0] or ""), "; ".join(sorted(rules[rep]))))
    paper_groups = [(r, ms) for r, ms in comps.items() if any(m in raw_meta.index for m in ms)]
    n_paper_groups_merged = sum(1 for r, ms in paper_groups if sum(1 for m in ms if m in raw_meta.index) > 1)
    n_work_only = len(comps) - n_paper_groups_merged
    bridge_merge_rows = [(r.openalex_id, r.merged_ids, r.title, _yr(r.year)) for r in bdf.itertuples() if ";" in r.merged_ids]
    n_in_merged_bridge = len(bridge_merge_rows)
    n_ambiguous = sum(1 for r in comps if any(re.search(r"of ([2-9]|\d\d+) longer", x) for x in rules[r]))
    n_canon_ambiguous = sum(1 for row in canon_group_rows if re.search(r"of ([2-9]|\d\d+) longer", row[4]))

    ly = pd.Series([meta.loc[u, "year"] for u in members["ltg"] if pd.notna(meta.loc[u, "year"])])
    py = pd.Series([meta.loc[u, "year"] for u in members["polycrisis"] if pd.notna(meta.loc[u, "year"])])
    ltg_year_text = (f"In the LtG corpus the median publication year is {ly.median():.1f}, {pct(int((ly <= 2022).sum()), len(ly))} of papers with a year are from 2022 or earlier, and {pct(int((ly < 2010).sum()), len(ly))} are from before 2010. "
                     f"In the polycrisis corpus the median year is {py.median():.1f} and {pct(int((py >= 2024).sum()), len(py))} of papers with a year are from 2024 or later.")
    n_pool_untitled = sum(1 for w in pool_ids if not (pool_info.get(w) or (None,))[0])
    merged_year_diff = 0
    for r in bdf.itertuples():
        if ";" in r.merged_ids:
            ys = {raw_meta.loc[m, "year"] for m in r.merged_ids.split("; ") if m in raw_meta.index and pd.notna(raw_meta.loc[m, "year"])}
            merged_year_diff += int(len(ys) > 1)
    md = []
    md.append("# Results: Citation Bridge Between Polycrisis and Limits to Growth\n")
    md.append("Generated by `scripts/06_citation_bridge.py`. Every number below was computed by the script. "
              f"The reference lists were refetched from OpenAlex on {refetch_dates}.\n")
    md.append("## Question\n\nWhich papers draw on both intellectual traditions? The test is whether a paper's "
              "reference list contains works from each tradition's own base of most cited sources.\n")
    md.append("## Definitions\n")
    md.append(
        "* **Corpora.** Polycrisis: `data/polycrisis_papers.csv`. Limits to Growth (LtG): `data/ltg_refined_papers.csv` "
        "(the refined pull, not the noisy `ltg_papers.csv`). Papers are deduplicated by OpenAlex id across corpora.\n"
        "* **Reference list.** The union of the refetched list and the list saved in the original raw file. Both are OpenAlex records of the same work's references, and the two differ for some papers (see Reference coverage).\n"
        f"* **Record merge (primary).** OpenAlex often holds several records of the same work, for example the Limits to Growth book. Before anything is counted, records are merged into one work when they have the same normalised title (lowercase, punctuation removed, spaces collapsed, at least {MIN_TITLE} characters). "
        "For corpus papers the first author surname must also match. A record whose whole title equals the part before the colon of longer titles joins the most cited of those title groups. "
        f"Only works cited by at least {MIN_POOL_COUNT} corpus papers in either corpus are checked for duplicate records, because titles were looked up for those only. "
        "A merged cited work is shown with the earliest year among its merged records and the type of that earliest record, and canon work years, their medians and the canon type counts use that earliest year and type (the representative record's own year and type are kept in the canon CSVs as representative_year and representative_type). A merged work is counted once per citing paper, and merged corpus papers are one paper whose reference list is the union of their lists. "
        "A merged corpus paper takes the publication year and the cited_by_count of its most cited member record (it is not summed). In the trend table it is therefore placed in that member's year, and in the top 15 ranking it is ranked by that member's count. "
        f"Among the {len(bdf)} merged bridge papers, {merged_year_diff} have member records with different publication years, so their placement in the yearly table depends on this rule. "
        "Results are shown with merged records (primary) and with unmerged records.\n"
        f"* **Canon (primary).** For each corpus, count how many of its papers cite each work. Canon_P is the top {PRIMARY_N} works "
        f"cited by polycrisis papers. Canon_L is the top {PRIMARY_N} works cited by LtG papers. Ties at the cutoff are broken by OpenAlex id. "
        "No works were picked by hand.\n"
        "* **Shared canon.** Works that appear in both top N lists. They are removed from both canons for the bridge test, "
        "so a bridge paper must cite something distinctive to each tradition, not only the works both fields share.\n"
        "* **Sensitivity.** The same test at N = 25 and N = 100. Variant 2 builds each canon from the N most cited papers inside "
        "each corpus by `cited_by_count` (OpenAlex citations from anywhere), again removing papers that land in both lists.\n"
        "* **Bridge paper.** A corpus paper with at least one reference in Canon_P and at least one in Canon_L. "
        "A stricter sensitivity version needs at least 2 references in Canon_P and at least 2 in Canon_L (column Min refs to each canon = 2 below), with the same denominator.\n"
        "* **Denominator.** Papers with a nonempty reference list, never all papers.\n")
    md.append("## Merged records\n")
    md.append(f"{len(comps)} groups of records were merged ({len(gmap)} ids): {n_paper_groups_merged} groups contain more than one corpus paper record and {n_work_only} contain at most one corpus paper record (cited works). "
              f"The corpus has {len(all0)} unique ids before merging and {len(all_ids)} papers after. {n_in_merged_bridge} of the {len(bdf)} bridge papers are merged records. "
              "Merged groups among the canon works (both top lists, before shared removal) are listed here. The representative id is the record cited most often. "
              f"The bare title rule is a heuristic: {n_ambiguous} of all groups used it with more than one longer title to choose from (it picks the most cited), and {n_canon_ambiguous} of the groups listed here did, so check those rows.\n")
    md.append(md_table(["Representative id", "Records", "Member ids (year)", "Title", "Rule"], canon_group_rows))
    md.append("\nBridge papers that are merged records of the same paper:\n")
    md.append(md_table(["Representative id", "Member ids", "Title", "Year"], bridge_merge_rows))
    md.append("\n## Reference coverage (merged records)\n")
    md.append(md_table(["Set", "Papers", "With reference list", "Share", "Total references", "Mean per paper with list", "Distinct works cited"], cov_rows))
    md.append("\nSaved list versus refetched list (unique papers, before merging): " + "; ".join(f"{k} {v}" for k, v in sorted(list_stats.items())) + ". "
              f"Under the union rule {n_changed_lists} papers have a different reference list than under the previous rule (refetched list if nonempty, else saved list). "
              f"Effect on the headline at N = {PRIMARY_N}, one reference per side: merged records {old_head[('merged', 1)]} of {old_head[('merged', 'den')]} before, {len(bdf)} of {len(with_refs)} now; "
              f"unmerged records {old_head[('unmerged', 1)]} of {old_head[('unmerged', 'den')]} before, {len(bridges0)} of {len(with0)} now. "
              f"Two references per side: merged {old_head[('merged', 2)]} before, {two50} now; unmerged {old_head[('unmerged', 2)]} before, {two50_0} now. "
              "Other rules for comparison (unmerged, N = 50, one reference per side): " + "; ".join(f"{k} {v[('unmerged', 1)]} of {v[('unmerged', 'den')]} (two references per side {v[('unmerged', 2)]})" for k, v in alt_heads.items())
              + f"; union (primary) {len(bridges0)} of {len(with0)} ({two50_0}). The choice of rule moves the unmerged count by up to {max(abs(v[('unmerged', 1)] - len(bridges0)) for v in alt_heads.values())} bridge papers.\n")
    md.append(f"\nPapers in both corpora: {len(both)}. Unique papers: {len(all_ids)}. "
              f"Papers with a reference list: {len(with_refs)}, which is the denominator below ({len(with0)} before merging).\n")
    md.append("## Counts at each N and variant\n")
    md.append(md_table(["Records", "Variant", "Min refs to each canon", "N", "Shared canon removed", "Canon_P size after removal", "Canon_L size after removal",
                        "Bridge papers", "Denominator", "Share"], sens_rows))
    md.append(f"\nPrimary result (merged records, most referenced works, N = {PRIMARY_N}): {len(bdf)} bridge papers out of {len(with_refs)} "
              f"papers with a reference list ({pct(len(bdf), len(with_refs))}). By corpus membership: "
              + ", ".join(f"{k} {v}" for k, v in nb_corp) + f". With unmerged records the same test gives {len(bridges0)} out of {len(with0)} ({pct(len(bridges0), len(with0))}); "
              f"the shared canon is {len(prim['shared'])} merged works against {len(prim0['shared'])} unmerged records.\n")
    md.append(f"At N = {PRIMARY_N} the cutoff count was {c50['cut'][0]} citing papers for Canon_P and {c50['cut'][1]} for Canon_L. "
              f"Works tied at the cutoff but left out: {c50['ties'][0]} for Canon_P and {c50['ties'][1]} for Canon_L.\n")
    kept = sens_df[(sens_df.records == "merged") & (sens_df.variant == "most referenced works, shared canon kept") & (sens_df.N == PRIMARY_N) & (sens_df.min_refs_each_canon == 1)].bridge_count.iloc[0]
    shared_names = "; ".join(f"{(info.get(w) or (None,))[0]} ({(info.get(w) or (None, None))[1]})" for w in sorted(prim["shared"], key=lambda w: (-(prim["counts_p"].get(w, 0) + prim["counts_l"].get(w, 0)), w)))
    md.append(f"The headline share depends on the shared canon choice: with the {len(prim['shared'])} shared works removed from both canons (primary) {len(bdf)} of {len(with_refs)} papers ({pct(len(bdf), len(with_refs))}) are bridges, and with the shared works kept in both canons {kept} of {len(with_refs)} ({pct(kept, len(with_refs))}); the shared works are: {shared_names}.\n")
    md.append("## Comparison with random reference baselines\n")
    md.append(f"Primary null (within corpus): each tested paper keeps its own number of references but draws them at random, as distinct works with no repeats, from the pooled references of its own corpus (polycrisis papers from polycrisis papers, LtG papers from LtG papers, papers in both corpora from the union), with works weighted by how many tested papers in that pool cite them. "
              f"{NULL_DRAWS} seeded draws, seed {SEED}. The canons are the real ones. This asks how many bridge papers appear if each paper cited its own field's literature at random, with no topic structure inside the field.\n")
    md.append(md_table(["N", "Min refs to each canon", "Observed bridge papers", "Within corpus null mean", "5th percentile", "95th percentile", "Maximum"],
                       [(n, m, obs, f"{mu:.1f}", f"{lo:.0f}", f"{hi:.0f}", mx) for n, m, obs, mu, lo, hi, mx in null_rows]))
    cat = ["below" if obs < lo else "above" if obs > hi else "inside" for n, m, obs, mu, lo, hi, mx in null_rows]
    nb_, ni_, na_ = cat.count("below"), cat.count("inside"), cat.count("above")
    if nb_ == len(cat):
        interp = "In every row the observed count is below the 5th percentile of the within corpus null, so the canons are cited together less often than random draws from each field's own literature would give. The count is low even after allowing for each field citing its own literature."
    elif na_ == len(cat):
        interp = "In every row the observed count is above the 95th percentile of the within corpus null, so papers cite both canons more often than random draws from their own field's literature."
    elif nb_ == 0 and na_ == 0:
        interp = "In every row the observed count is inside the 5th to 95th range of the within corpus null, so the low bridge count is what each field citing its own literature produces, and the data do not show avoidance of the other tradition."
    else:
        interp = (f"Of the {len(cat)} rows, {nb_} are below the within corpus null range, {ni_} are inside it and {na_} are above it. "
                  + "Rows below: " + (", ".join(f"N = {n} with {m} reference{'s' if m > 1 else ''} per side" for (n, m, *_), c in zip(null_rows, cat) if c == "below") or "none") + ". "
                  + "Rows above: " + (", ".join(f"N = {n} with {m} reference{'s' if m > 1 else ''} per side" for (n, m, *_), c in zip(null_rows, cat) if c == "above") or "none") + ". "
                  "Where the observed count is inside or above the range, the count is not lower than each field citing its own literature would give; where it is below, it is lower than that.")
    md.append("\n" + interp + " The model ignores topical clustering inside each field, which any real literature has, so being near or below it is expected to some degree.\n")
    md.append("Secondary null (pooled): the same draw, but from the pooled references of both corpora together. Because real papers live in one corpus, a count below this null mostly reflects the two corpora citing different literatures, so it is shown for comparison only.\n")
    md.append(md_table(["N", "Min refs to each canon", "Observed bridge papers", "Pooled null mean", "5th percentile", "95th percentile", "Maximum"],
                       [(n, m, obs, f"{mu:.1f}", f"{lo:.0f}", f"{hi:.0f}", mx) for n, m, obs, mu, lo, hi, mx in null_rows_pooled]))
    md.append("\nWithin corpus null for the variant that keeps the shared canon in both canons:\n")
    md.append(md_table(["N", "Min refs to each canon", "Observed bridge papers", "Within corpus null mean", "5th percentile", "95th percentile", "Maximum"],
                       [(n, m, obs, f"{mu:.1f}", f"{lo:.0f}", f"{hi:.0f}", mx) for n, m, obs, mu, lo, hi, mx in null_keep]))
    md.append("\nBoth nulls remove topic structure, and weighted draws favour heavily cited works, so papers with long reference lists hit both canons more often in the null.\n")
    md.append("## Canon cutoff ties\n")
    t50 = [r for r in tie_rows if r["N"] == PRIMARY_N][0]
    md.append(f"The canon is the top N works by number of citing papers, and many works tie at the cutoff count, so which tied works get in was decided by id order. "
              f"At N = {PRIMARY_N} the cutoff is {t50['cutoff_count_polycrisis']} citing papers for Canon_P ({t50['tied_polycrisis']} works tied for {t50['open_slots_polycrisis']} open slots) and {t50['cutoff_count_ltg']} for Canon_L ({t50['tied_ltg']} works tied for {t50['open_slots_ltg']} open slots). "
              f"The table shows bridge counts with the canon built three ways: id order (the primary), the strict canon (only works above the cutoff count, so the canon is smaller than N), and random tie breaking ({TIE_BRIDGE_DRAWS} seeded draws, mean and 5th to 95th percentile). Merged records, most referenced works.\n")
    md.append(md_table(["N", "Min refs to each canon", "Id order (primary)", "Strict canon", "Random ties: mean (5th to 95th)", "Tied works for open slots (Canon_P / Canon_L)"],
                       [(r["N"], r["min_refs_each_canon"], r["bridges_id_order"], r["bridges_strict_canon"],
                         f"{r['bridges_random_ties_mean']:.1f} ({r['bridges_random_ties_p5']:.0f} to {r['bridges_random_ties_p95']:.0f})",
                         f"{r['tied_polycrisis']} for {r['open_slots_polycrisis']} / {r['tied_ltg']} for {r['open_slots_ltg']}") for r in tie_rows]))
    md.append("\n")
    top_k = min(15, len(bdf))
    md.append(f"## Top {top_k} bridge papers by cited_by_count, with rows tied at rank {top_k} ({len(top15)} rows; N = {PRIMARY_N}, most referenced works, merged records)\n")
    md.append(f"Rank {top_k} has cited_by_count {cut15}, shared by {n_tied15} bridge papers, so all tied rows are shown ({len(top15)} rows). Ties are ordered by OpenAlex id.\n")
    md.append(md_table(["Rank", "Title", "Year", "Corpus", "Refs to Canon_P", "Refs to Canon_L", "Cited by"],
                       [(i, r.title, int(r.year) if pd.notna(r.year) else "", r.corpus, r.n_refs_canon_polycrisis,
                         r.n_refs_canon_ltg, r.cited_by_count) for i, r in enumerate(top15.itertuples(), 1)]))
    md.append(f"\n## Shared canon (N = {PRIMARY_N}, merged records): {len(prim['shared'])} works in both top lists\n")
    md.append(md_table(["Work id", "Title", "Year", "Polycrisis papers citing", "LtG papers citing", "Merged record ids"], shared_rows))
    md.append("\n## Trend over time\n")
    md.append(f"Years shown start at {FIG_START_YEAR}. Before that there are {len(pre)} papers with a reference list and {len(pre_b)} bridge papers, "
              f"so earlier years add almost nothing. {len(noyear)} papers with a reference list have no publication year and are left out of the yearly figures. "
              f"2026 is a partial year: the latest publication date in the data is {latest_date}, so the 2025 to 2026 period and the 2026 point are not comparable with full years.\n")
    md.append(md_table(["Period", "Papers with reference list", "Bridge papers", "Share"], prow))
    md.append("\nPer year values are in the figure `figures/bridge_papers_by_year.png` and printed by the script. "
              f"Hollow points in the figure mark years with fewer than {MIN_YEAR_DENOM} papers with a reference list, where the share is unstable.\n")
    md.append("Share by period: " + "; ".join(f"{r[0]} {r[3]} ({r[2]} of {r[1]})" for r in prow) + ". "
              f"The earliest bridge paper in the data is from {first_bridge_year}. The highest period share is {peak[0]} at {peak[3]}.\n")
    shares_p = [int(r[2]) / r[1] if r[1] else 0.0 for r in prow]
    peak_i = int(np.argmax(shares_p))
    if all(b >= a for a, b in zip(shares_p, shares_p[1:])):
        pattern_text = f"The period shares ({', '.join(r[3] for r in prow)}) never fall from one period to the next."
    else:
        pattern_text = (f"The period shares are {', '.join(f'{r[0]} {r[3]}' for r in prow)}. The share is highest in {prow[peak_i][0]} ({prow[peak_i][3]}) and is {prow[-1][3]} in {prow[-1][0]}, "
                        f"so the pattern is a low start ({prow[0][2]} bridge paper{'' if prow[0][2] == 1 else 's'} in {prow[0][0]}) followed by shares between {min(shares_p[1:]) * 100:.1f}% and {max(shares_p) * 100:.1f}% without a steady rise.")
    md.append("### Canon work years and the trend\n")
    md.append(f"The canon is made from what the corpus papers cite, and {pct(int((py >= 2024).sum()), len(py))} of polycrisis papers with a year are from 2024 or later, so the canon can hold recent works that older papers could not cite. "
              f"Canon_P works (after shared removal) were published from {min(yp)} to {max(yp)} (median {med(yp):.1f}, year known for {len(yp)} of {len(prim['eff_p'])} works); Canon_L works from {min(yl)} to {max(yl)} (median {med(yl):.1f}, year known for {len(yl)} of {len(prim['eff_l'])} works). "
              f"Pooled, the median canon work year is {med(ypool):.1f} and the newest is {Y_MAX}. A paper published before a canon work cannot cite it, so low shares in early periods are partly mechanical.\n")
    md.append(f"Restricted check: counting only papers published in {Y_MED} or later (the median canon year, rounded up), {r_med[2]} of {r_med[1]} papers with a reference list are bridges ({r_med[3]}). "
              f"Counting only papers published in {Y_MID} or later (the higher of the two canon medians), {r_mid[2]} of {r_mid[1]} ({r_mid[3]}). "
              f"Counting only papers published in {Y_MAX} or later (the newest canon work year), {r_max[2]} of {r_max[1]} ({r_max[3]}). "
              f"Shares by year from {Y_MED} for years with at least {MIN_YEAR_DENOM} papers with a reference list: "
              + ", ".join(f"{int(y)} {100 * x:.1f}%" for y, x in zip(rq.year, rq.share)) + f" (2026 is partial, through {latest_date}). "
              + ("These shares do not fall from one year to the next, which supports a rise within the restricted years. "
                 if r_rise else "These shares do not rise steadily from year to year, so the restricted check does not support a steady rise. ")
              + pattern_text + "\n")
    md.append("The counts per year are small, so a rise or fall of a point or two between periods is within what a few papers can change.\n")
    md.append(f"Bridge papers that are themselves canon works: {len(canon_is_bridge)} of {len(bdf)} (canon works after shared removal), plus {len(canon_shared_bridge)} that are in the shared canon. "
              "Canon works in the corpus can cite other canon works, so this counts toward the bridge total.\n")
    md.append("## Limitations\n")
    md.append(
        f"* **Reference coverage.** Only {len(with_refs)} of {len(all_ids)} papers after merging ({pct(len(with_refs), len(all_ids))}) have any reference list in OpenAlex. "
        "The bridge share is computed only over those. Papers without a list cannot be tested here, and nothing is claimed about them.\n"
        f"* **OpenAlex reference completeness.** The saved and refetched reference lists are different sets for {list_stats['different (neither contains the other)']} papers and one is a strict subset of the other for {list_stats['refetched shorter (subset of saved)'] + list_stats['refetched longer (superset of saved)']} more, so the lists are not fixed. Coverage differs by document type, which means the tested papers are not a random sample. "
        f"By type (papers with a reference list out of all papers; the {min(6, len(tcov))} most common of {len(tcov)} types): " + "; ".join(f"`{t}` {k} of {n}" for t, n, k in tcov[:6]) + ".\n"
        "* **Books and chapters.** Types of the " + str(len(canon_ids)) + " canon works in the two top lists before shared removal: "
        + "; ".join(f"`{t}` {c}" for t, c in canon_types) + ". "
        "A book cited by a paper counts like any other work. " + books_chap_text + "\n"
        "* **Corpus query scope.** The corpora come from keyword searches, so the polycrisis corpus holds papers that use that word and the LtG corpus holds papers that matched the refined query. "
        "Canon_P and Canon_L describe what these two retrieved sets cite, not every paper in either field. " + ltg_year_text + "\n"
        f"* **Circularity.** The canon is derived from the same corpus it is then tested on, so papers in the corpus are by construction likely to cite works that are frequent in the corpus. A bridge here means citing the corpus's own frequent works from each side, not citing an outside list of founding texts. {len(canon_is_bridge)} of the {len(bdf)} bridge papers are themselves canon works.\n"
        f"* **Low bridge bar.** The main definition needs only one reference on each side. Requiring at least 2 on each side gives {two50} bridge papers at N = {PRIMARY_N} instead of {one50} with merged records ({two50_0} instead of {len(bridges0)} unmerged). The main count should be read as the loosest version.\n"
        "* **Baseline.** The only baseline is the random reference draw above, which ignores topic. There is no comparison with a different pair of fields, so the bridge counts say how often both canons are cited, not whether that is more than usual for two related fields.\n"
        f"* **Unresolved titles.** {n_pool_untitled} of the {len(pool_ids)} works checked for duplicate records have no resolved title, so they cannot be merged.\n"
        f"* **Duplicate records.** Merging uses titles (and first author for papers) and was only run on works cited by at least {MIN_POOL_COUNT} corpus papers, so duplicates of less cited works are not merged, and different works with identical titles could be merged by mistake. The merged groups among the canon are listed above.\n"
        "* **Canon definition.** A canon built from counts depends on corpus size and composition. A paper in both corpora counts toward both counts. Sensitivity runs above show how much the answer moves with N and with the variant.\n"
        f"* **Refetch gaps.** 06a asked OpenAlex for every id; {n_not_refetched} of {len(all0)} ids returned no record, and for those the saved list was used. "
        f"Titles could not be resolved for {n_title_missing} of {len(canon_ids)} canon works.\n"
        f"* **Thin canons.** At N = {PRIMARY_N} the cutoff is {c50['cut'][0]} citing papers for Canon_P and {c50['cut'][1]} for Canon_L, and several works tie at the cutoff, "
        f"so which tied works are in the canon is decided by id order. At N = {PRIMARY_N} bridge counts are {t50_1['bridges_id_order']} (id order), {t50_1['bridges_strict_canon']} (strict canon) and {t50_1['bridges_random_ties_mean']:.1f} on average with random ties (5th to 95th {t50_1['bridges_random_ties_p5']:.0f} to {t50_1['bridges_random_ties_p95']:.0f}); see the cutoff ties section. Treat the canon as a rough core, not a settled list.\n"
        f"* **Partial 2026.** The latest publication date in the data is {latest_date}, so 2026 holds only part of a year.\n")
    text = "\n".join(md)
    (DOC_DIR / "RESULTS_citation_bridge.md").write_text(text)

    bad = [l for l in text.splitlines() if not l.startswith("|") and any(ch in re.sub(r"`[^`]*`", "", l) for ch in "-–—")]
    print(f"Prose lines containing a dash character: {len(bad)}")
    for l in bad:
        print("  DASH:", l[:140])


if __name__ == "__main__":
    main()
