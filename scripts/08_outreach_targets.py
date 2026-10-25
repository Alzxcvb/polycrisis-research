"""
Outreach targets: which authors are worth reading and citing first?
Project: Polycrisis & Limits to Growth Research
Author: Alex Coffman

What this does:
    Builds four ranked lists of authors from the authorships in the two raw corpus files.
    Identity is the OpenAlex author id, never the name. Papers are the merged records of
    06_citation_bridge.py (same loading, same reference list rule, same build_gmap), so paper
    counts agree with that script. An author counts once per merged paper.
    B1  most prolific polycrisis corpus authors (top 30)
    B2  authors with at least one paper in each corpus, ranked by the smaller of the two counts
    B3  central authors in the polycrisis coauthor network (degree and betweenness, top 30)
    B4  authors of 2024 to 2026 papers in the target journals named in docs/PUBLICATION_STRATEGY.md
    These are research leads. The script collects no contact details.
    Every number in the note is computed by the script and written to stdout or the CSVs.

Run AFTER 06_citation_bridge.py has written data/canon_metadata_cache.json (from the repo root):
    .venv/bin/python scripts/08_outreach_targets.py

Output:
    data/outreach_B1.csv, data/outreach_B2.csv, data/outreach_B3.csv, data/outreach_B4.csv, data/outreach_B3_sensitivity.csv
    docs/RESULTS_outreach_targets.md
Inputs (all in the gitignored data/): polycrisis_papers_raw.json, ltg_refined_papers_raw.json,
polycrisis_papers.csv, ltg_refined_papers.csv, references_refetch_2026-10-24.json, canon_metadata_cache.json,
and bridge_papers.csv if present. Requires networkx 3.2.1.
Network: none. The script reads local files only and stops with a clear message if
data/canon_metadata_cache.json (written by 06 on its first run) does not cover every id it needs, so it
never calls OpenAlex. data/ is gitignored, so a clean clone must rebuild the raw files, the 06 outputs
and the cache before this script can run. Runs are deterministic.
"""

import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DOC_DIR = ROOT / "docs"

TOP_N = 30
TABLE_N = 15
TARGET_JOURNALS = ["Global Sustainability", "Scientometrics", "Sustainability", "Journal of Industrial Ecology"]
YEAR_MIN, YEAR_MAX = 2024, 2026
WINDOW = f"{YEAR_MIN} to {YEAR_MAX}"
NX_TESTED = "3.2.1"      # networkx version the script was tested with
BTW_DECIMALS = 8        # B3 betweenness is rounded to this many decimals before ranking and printed with the same number
SENS_MAX_AUTHORS = 20   # B3 sensitivity: papers with more author ids than this are left out
INST_SHOWN = 3          # institutions shown per author in the note tables (the CSV has all)

try:
    import networkx as nx
except ImportError:
    raise SystemExit("networkx is required (tested with " + NX_TESTED + ") and is not in the committed requirements.txt. "
                     "Install it in the venv with: .venv/bin/python -m pip install networkx==" + NX_TESTED)

# ── Reuse 06 so counts agree ──────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location("bridge06", ROOT / "scripts" / "06_citation_bridge.py")
b06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b06)


def short(x):
    return str(x).rsplit("/", 1)[-1]


def merged_papers():
    """Merged units exactly as 06 builds them. Returns (unit meta, corpus membership of units, gmap, number of merged groups, set of CSV record ids)."""
    raw_meta, raw_members, raw_refs, _, _ = b06.load()
    ucount = Counter()
    pc = b06.ref_counts(raw_members["polycrisis"], raw_refs)
    lc = b06.ref_counts(raw_members["ltg"], raw_refs)
    for w in set(pc) | set(lc):
        ucount[w] = pc[w] + lc[w]
    pool = sorted(w for w in set(pc) | set(lc) if pc[w] >= b06.MIN_POOL_COUNT or lc[w] >= b06.MIN_POOL_COUNT)
    cache = json.load(open(b06.CACHE_PATH)) if b06.CACHE_PATH.exists() else {}
    missing = [i for i in pool if i not in cache]
    if missing:
        raise SystemExit(f"data/canon_metadata_cache.json is missing {len(missing)} of {len(pool)} ids. "
                         "Run scripts/06_citation_bridge.py once with network access to fill it. 08 never calls OpenAlex.")
    gmap, comps, _ = b06.build_gmap(raw_meta, b06.resolve_titles(pool), ucount)
    meta, members, _ = b06.make_context(raw_meta, raw_members, raw_refs, gmap)
    return meta, members, gmap, len(comps), set(raw_meta.index)


def load_authorships(gmap, csv_ids):
    """Per merged unit: {author id: {names, institutions}} plus sources and counts of lost authorships."""
    unit_auth = defaultdict(dict)       # unit -> author id -> {"names": Counter, "inst": Counter}
    unit_src = defaultdict(set)         # unit -> source display names
    stats = Counter()
    seen_rec = set()
    rec_info = []       # (source name, unit, record id, year, type, authorship count) per raw record
    for f in ["polycrisis_papers_raw.json", "ltg_refined_papers_raw.json"]:
        for w in json.load(open(DATA_DIR / f)):
            wid = short(w["id"])
            stats["entries"] += 1
            u = gmap.get(wid, wid)
            src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
            if src:
                unit_src[u].add(src)
            rec_info.append((src, u, wid, w.get("publication_year"), w.get("type"), len(w.get("authorships") or [])))
            if wid in seen_rec:
                continue                # id already read (in either file or earlier in this one): authorships counted once
            seen_rec.add(wid)
            for a in w.get("authorships") or []:
                stats["authorships"] += 1
                au = a.get("author") or {}
                aid = au.get("id")
                if not aid:
                    stats["without author id"] += 1
                    continue
                aid = short(aid)
                d = unit_auth[u].setdefault(aid, {"names": Counter(), "inst": Counter()})
                if au.get("display_name"):
                    d["names"][au["display_name"]] += 1
                for i in a.get("institutions") or []:
                    if i.get("display_name"):
                        d["inst"][i["display_name"]] += 1
    stats["distinct"] = len(seen_rec)
    assert seen_rec == set(csv_ids), "record ids in the raw JSON files differ from the ids in the corpus CSV files"
    return unit_auth, unit_src, stats, rec_info


def build_author_table(units, meta, unit_auth):
    """units: iterable of merged paper ids. Returns {aid: dict} counts, citations, latest year, names, institutions, papers."""
    tab = {}
    for u in sorted(units):
        for aid in sorted(unit_auth.get(u, {})):
            d = unit_auth[u][aid]
            t = tab.setdefault(aid, {"papers": [], "cites": 0, "year": 0, "names": Counter(), "inst": Counter()})
            t["papers"].append(u)
            t["cites"] += int(meta.loc[u, "cited_by_count"])
            y = meta.loc[u, "year"]
            if pd.notna(y):
                t["year"] = max(t["year"], int(y))
            t["names"].update(d["names"])
            t["inst"].update(d["inst"])
    return tab


def name_of(t):
    return sorted(t["names"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if t["names"] else ""


def insts(t):
    return [k for k, _ in sorted(t["inst"].items(), key=lambda kv: (-kv[1], kv[0]))]


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def esc(s):
    return str(s).replace("|", "/")


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(out)


def short_inst(t):
    xs = insts(t)
    s = "; ".join(xs[:INST_SHOWN])
    return s + (f" (+{len(xs) - INST_SHOWN} more)" if len(xs) > INST_SHOWN else "")


def pl(n, one, many):
    return f"{n:,} {one if n == 1 else many}"


def main():
    meta, members, gmap, n_groups, csv_ids = merged_papers()
    unit_auth, unit_src, st, rec_info = load_authorships(gmap, csv_ids)
    n_raw = st["distinct"]          # distinct record ids counted from the raw JSON; equal to the CSV ids (asserted)
    P, L = members["polycrisis"], members["ltg"]
    both_units = P & L
    print(f"Raw file entries: {st['entries']}; distinct record ids: {n_raw}; extra entries from repeated ids: {st['entries'] - n_raw}")
    print(f"Merged groups: {n_groups}; merged papers: {len(meta)}")
    print(f"Merged papers: polycrisis {len(P)}, LtG {len(L)}, in both {len(both_units)}")
    print(f"Authorships read over {n_raw} distinct record ids (repeated entries read once): {st['authorships']}; without author id: {st['without author id']}")
    print(f"Papers with no usable author id: {sum(1 for u in meta.index if not unit_auth.get(u))}")

    tabP = build_author_table(P, meta, unit_auth)
    tabL = build_author_table(L, meta, unit_auth)
    tabAll = build_author_table(P | L, meta, unit_auth)
    print(f"Distinct author ids: polycrisis {len(tabP)}, LtG {len(tabL)}, either {len(tabAll)}")

    # ── B1 ──
    b1 = sorted(tabP, key=lambda a: (-len(tabP[a]["papers"]), -tabP[a]["cites"], a))
    b1_top = b1[:TOP_N]
    cut = len(tabP[b1_top[-1]]["papers"])
    tie_out = sum(1 for a in b1[TOP_N:] if len(tabP[a]["papers"]) == cut)
    write_csv(DATA_DIR / "outreach_B1.csv",
              ["rank", "openalex_author_id", "name", "papers", "total_cited_by_count", "latest_year", "institutions"],
              [[i + 1, a, name_of(tabP[a]), len(tabP[a]["papers"]), tabP[a]["cites"], tabP[a]["year"], "; ".join(insts(tabP[a]))]
               for i, a in enumerate(b1_top)])
    print(f"B1: top {TOP_N} written; paper count at rank {TOP_N} is {cut}; authors tied at that count outside the list: {tie_out}")

    # ── B2 ──
    bp_path = DATA_DIR / "bridge_papers.csv"
    bridge_set = set()
    if bp_path.exists():
        bp = pd.read_csv(bp_path)
        bridge_set = set(bp["openalex_id"])
        bridge_title = dict(zip(bp["openalex_id"], bp["title"]))
        print(f"bridge_papers.csv found: {len(bridge_set)} bridge papers")
    else:
        bridge_title = {}
        print("bridge_papers.csv not found")
    both = sorted(set(tabP) & set(tabL))
    rows2 = []
    for a in both:
        np_, nl = len(tabP[a]["papers"]), len(tabL[a]["papers"])
        allp = tabAll[a]["papers"]
        xp = sum(1 for p in tabP[a]["papers"] if p not in both_units)   # polycrisis papers not also in LtG
        xl = sum(1 for p in tabL[a]["papers"] if p not in both_units)   # LtG papers not also in polycrisis
        rows2.append((a, np_, nl, min(np_, nl), len(allp), tabAll[a]["cites"], [p for p in allp if p in bridge_set], xp, xl))
    rows2.sort(key=lambda r: (-r[3], -r[4], -r[5], r[0]))
    write_csv(DATA_DIR / "outreach_B2.csv",
              ["rank", "openalex_author_id", "name", "polycrisis_papers", "ltg_papers", "min_count", "total_papers",
               "total_cited_by_count", "latest_year", "bridge_paper_ids", "bridge_paper_titles", "institutions",
               "polycrisis_papers_without_shared", "ltg_papers_without_shared"],
              [[i + 1, r[0], name_of(tabAll[r[0]]), r[1], r[2], r[3], r[4], r[5], tabAll[r[0]]["year"],
                "; ".join(r[6]), "; ".join(str(bridge_title.get(p, "")) for p in r[6]), "; ".join(insts(tabAll[r[0]])), r[7], r[8]]
               for i, r in enumerate(rows2)])
    b2_has_shared = sum(1 for r in rows2 if any(p in both_units for p in tabAll[r[0]]["papers"]))
    b2_only_shared = sum(1 for r in rows2 if r[7] == 0 or r[8] == 0)
    b2_all_shared = sum(1 for r in rows2 if r[7] == 0 and r[8] == 0)
    print(f"B2: authors with at least one paper that is in both corpora: {b2_has_shared}; authors who would not qualify without those papers: {b2_only_shared}; authors whose every paper is one of them: {b2_all_shared}")
    n_b2_with_bridge = sum(1 for r in rows2 if r[6])
    print(f"B2: authors in both corpora: {len(rows2)}; with at least one bridge paper: {n_b2_with_bridge}; min count 1: {sum(1 for r in rows2 if r[3] == 1)}")

    # ── B3 ──
    edges = set()
    max_auth = 0
    for u in sorted(P):
        ids = sorted(unit_auth.get(u, {}))
        max_auth = max(max_auth, len(ids))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                edges.add((ids[i], ids[j]))
    deg = Counter()
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    nodes = sorted(tabP)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(sorted(edges))
    btw = nx.betweenness_centrality(G, normalized=True)
    ncomp = nx.number_connected_components(G)
    giant = max(len(c) for c in nx.connected_components(G))
    print(f"B3: networkx {nx.__version__}; nodes {G.number_of_nodes()}, edges {G.number_of_edges()}, components {ncomp}, largest component {giant}")
    key = lambda a: (-round(btw[a], BTW_DECIMALS), -deg[a], a)
    b3_all = sorted(nodes, key=key)
    b3 = b3_all[:TOP_N]
    tie_key = lambda a: (round(btw[a], BTW_DECIMALS), deg[a])
    tie_size = Counter(tie_key(a) for a in nodes)
    other_tied = lambda a: tie_size[tie_key(a)] - 1      # other authors with the same degree and rounded betweenness
    tie15 = tie_key(b3_all[TABLE_N - 1])
    tie15_rows = [i for i, a in enumerate(b3_all) if tie_key(a) == tie15]
    tie_last = tie15_rows[-1]
    print(f"B3: authors tied with rank {TABLE_N} (same degree and rounded betweenness): {len(tie15_rows)}, occupying ranks {tie15_rows[0] + 1} to {tie_last + 1}")
    tie_cut = other_tied(b3_all[TOP_N - 1])
    print(f"B3: authors tied with rank {TOP_N}: {tie_cut}")
    big = [u for u in sorted(P) if len(unit_auth.get(u, {})) == max_auth]
    big_auth = set().union(*[set(unit_auth[u]) for u in big]) if big else set()
    n_big_top = sum(1 for a in b3 if a in big_auth)
    print(f"B3: papers with {max_auth} author ids: {big}; top {TOP_N} authors on them: {n_big_top}")
    # sensitivity: leave out papers with more than SENS_MAX_AUTHORS author ids
    sens_units = [u for u in sorted(P) if len(unit_auth.get(u, {})) <= SENS_MAX_AUTHORS]
    n_excl = len(P) - len(sens_units)
    sedges = set()
    for u in sens_units:
        ids = sorted(unit_auth.get(u, {}))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sedges.add((ids[i], ids[j]))
    sdeg = Counter()
    for a, b in sedges:
        sdeg[a] += 1
        sdeg[b] += 1
    SG = nx.Graph()
    SG.add_nodes_from(nodes)
    SG.add_edges_from(sorted(sedges))
    sb = nx.betweenness_centrality(SG, normalized=True)
    sens_rank = sorted(nodes, key=lambda a: (-round(sb[a], BTW_DECIMALS), -sdeg[a], a))
    sens_top = sens_rank[:TOP_N]
    sens_overlap = len(set(sens_top) & set(b3))
    skey = lambda a: (round(sb[a], BTW_DECIMALS), sdeg[a])
    s_last = skey(sens_rank[TOP_N - 1])
    s_tied_pos = [i for i, a in enumerate(sens_rank) if skey(a) == s_last]
    s_tie_other = len(s_tied_pos) - 1
    print(f"B3 sensitivity: other authors tied with rank {TOP_N}: {s_tie_other}, occupying ranks {s_tied_pos[0] + 1} to {s_tied_pos[-1] + 1}")
    print(f"B3 sensitivity: papers with more than {SENS_MAX_AUTHORS} author ids left out: {n_excl}; edges {len(sedges)}; overlap with primary top {TOP_N}: {sens_overlap}")
    print(f"B3: largest author list on one merged polycrisis paper: {max_auth}")
    write_csv(DATA_DIR / "outreach_B3.csv",
              ["rank", "openalex_author_id", "name", "degree", "betweenness", "other_authors_tied_with_this_row", "polycrisis_papers", "institutions"],
              [[i + 1, a, name_of(tabP[a]), deg[a], f"{btw[a]:.{BTW_DECIMALS}f}", other_tied(a), len(tabP[a]["papers"]), "; ".join(insts(tabP[a]))]
               for i, a in enumerate(b3)])
    write_csv(DATA_DIR / "outreach_B3_sensitivity.csv",
              ["rank", "openalex_author_id", "name", "degree", "betweenness", f"in_primary_top_{TOP_N}", "institutions"],
              [[i + 1, a, name_of(tabP[a]), sdeg[a], f"{sb[a]:.{BTW_DECIMALS}f}", "yes" if a in b3 else "no", "; ".join(insts(tabP[a]))]
               for i, a in enumerate(sens_top)])
    overlap_b1_b3 = len(set(b1_top) & set(b3))
    print(f"B3: authors in both the B1 and B3 top {TOP_N}: {overlap_b1_b3}")

    # ── B4 ──
    name_hits = Counter()
    near = Counter()
    pap4 = defaultdict(set)
    for u in sorted(P | L):
        y = meta.loc[u, "year"]
        for s in unit_src.get(u, ()):
            if pd.notna(y) and YEAR_MIN <= int(y) <= YEAR_MAX:
                if s in TARGET_JOURNALS:
                    name_hits[(s, int(y))] += 1
                    pap4[s].add(u)
                elif any(t.lower() in s.lower() for t in TARGET_JOURNALS):
                    near[s] += 1
    b4_units = set().union(*pap4.values()) if pap4 else set()
    tab4 = build_author_table(b4_units, meta, unit_auth)
    b4_noauth = sorted(u for u in b4_units if not unit_auth.get(u))
    noauth_n = {}
    for src_, u_, wid_, y_, ty_, n_ in rec_info:
        if u_ in b4_noauth:
            noauth_n[u_] = max(noauth_n.get(u_, 0), n_)
    print(f"B4: papers that add no author: {b4_noauth}; authorships on record: {noauth_n}")
    b4 = sorted(tab4, key=lambda a: (-len(tab4[a]["papers"]), -tab4[a]["cites"], a))
    jr = {a: sorted({s for s in TARGET_JOURNALS if any(p in pap4[s] for p in tab4[a]["papers"])}) for a in tab4}
    write_csv(DATA_DIR / "outreach_B4.csv",
              ["rank", "openalex_author_id", "name", f"papers_{YEAR_MIN}_{YEAR_MAX}_in_target_journals", "total_cited_by_count",
               "latest_year", "journals", "in_polycrisis", "in_ltg", "institutions"],
              [[i + 1, a, name_of(tab4[a]), len(tab4[a]["papers"]), tab4[a]["cites"], tab4[a]["year"], "; ".join(jr[a]),
                "yes" if a in tabP else "no", "yes" if a in tabL else "no", "; ".join(insts(tab4[a]))]
               for i, a in enumerate(b4)])
    print(f"B4: papers {len(b4_units)}; authors {len(tab4)}; per journal {{{', '.join(f'{s}: {len(pap4[s])}' for s in TARGET_JOURNALS)}}}")
    print(f"B4: near miss source names (case insensitive substring of a target name, not an exact match): {dict(near)}")

    def tie_span(order, keyf, pos):
        k = keyf(order[pos - 1])
        idx = [i for i, a in enumerate(order) if keyf(a) == k]
        return len(idx), idx[0] + 1, idx[-1] + 1
    tie_rows = []
    r2 = {r[0]: r for r in rows2}
    specs = [
        ("B1", "papers, then total citations, then author id", b1, lambda a: (len(tabP[a]["papers"]), tabP[a]["cites"])),
        ("B2", "smaller of the two corpus counts, then total papers, then total citations, then author id", [r[0] for r in rows2], lambda a: (r2[a][3], r2[a][4], r2[a][5])),
        ("B3", f"betweenness rounded to {BTW_DECIMALS} decimals, then degree, then author id", b3_all, tie_key),
        ("B4", "papers in the target journals, then total citations, then author id", b4, lambda a: (len(tab4[a]["papers"]), tab4[a]["cites"])),
    ]
    for nm, seq, order, kf in specs:
        cells = []
        for pos in (TABLE_N, TOP_N):
            if pos > len(order):
                cells.append("list ends before this rank")
                continue
            n_, lo, hi = tie_span(order, kf, pos)
            cells.append(f"{n_} tied, ranks {lo} to {hi}" if n_ > 1 else "no tie")
            print(f"Ties {nm} at rank {pos}: {pl(n_, 'author', 'authors')} with identical sort keys before the id, ranks {lo} to {hi}")
        tie_rows.append([nm, seq, cells[0], cells[1]])
    tietab = md_table(["List", "Order (each step applies only when the earlier ones tie)", f"Full tie at rank {TABLE_N}", f"Full tie at rank {TOP_N}"], tie_rows)
    def type_counts(units):
        c = Counter(str(meta.loc[u, "type"]) for u in units)
        non = sum(v for k, v in c.items() if k != "article")
        return non, len(units), c
    type_lists = [
        ("All merged papers in either corpus", P | L),
        ("Polycrisis papers of the B1 authors", set().union(*[set(tabP[a]["papers"]) for a in b1_top])),
        ("Papers of the B2 authors", set().union(*[set(tabAll[r[0]]["papers"]) for r in rows2])),
        ("Polycrisis papers of the B3 authors", set().union(*[set(tabP[a]["papers"]) for a in b3])),
        ("Papers in B4", b4_units),
    ]
    type_rows = []
    for lab, us in type_lists:
        non, tot, c = type_counts(us)
        brk = ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])) if k != "article")
        type_rows.append([lab, tot, non, brk if brk else "none"])
        print(f"Types: {lab}: {tot} papers, {non} not of type article ({brk if brk else 'none'})")
    ttab = md_table(["Set of papers", "Merged papers", "Not of type article", "Types other than article"], type_rows)
    yrs_by_unit = defaultdict(set)
    for src_, u_, wid_, y_, ty_, na_ in rec_info:
        if y_ is not None:
            yrs_by_unit[u_].add(int(y_))
    n_diff_years = sum(1 for u_ in (P | L) if len(yrs_by_unit.get(u_, ())) > 1)
    own_year_units = {u_ for src_, u_, wid_, y_, ty_, na_ in rec_info
                      if src_ in TARGET_JOURNALS and y_ is not None and YEAR_MIN <= int(y_) <= YEAR_MAX}
    same_result = own_year_units == b4_units
    print(f"Merged papers whose raw records have differing years (all records, with or without a source name): {n_diff_years}")
    print(f"B4: same paper set if each record's own year is used: {same_result}")
    yr_clause = ("; the set of matched papers is the same if each record's own year is used instead" if same_result
                 else "; using each record's own year would change the set of matched papers")
    recs = {}
    for src, u, wid, y, ty, na in rec_info:
        recs[(src, wid)] = (u, y, ty, na)
    why = {}
    for sname in TARGET_JOURNALS:
        rr = [v for (src, _), v in recs.items() if src == sname]
        out_years = sorted(int(v[1]) for v in rr if v[1] is not None and not YEAR_MIN <= int(v[1]) <= YEAR_MAX)
        noauth = [(int(v[1]), v[2]) for v in rr if v[1] is not None and YEAR_MIN <= int(v[1]) <= YEAR_MAX and v[3] == 0]
        why[sname] = (len(rr), out_years, noauth)
        print(f"B4 {sname}: raw records with this source name {len(rr)}; outside {YEAR_MIN} to {YEAR_MAX}, years {out_years}; in window with no authorships {noauth}")

    # ── Note ──
    jr_rows = []
    for s in TARGET_JOURNALS:
        yrs = {y: name_hits[(s, y)] for y in range(YEAR_MIN, YEAR_MAX + 1)}
        print(f"B4 {s}: merged papers by year {yrs}; authors {len({a for a in tab4 if s in jr[a]})}")
        jr_rows.append([s, "matched" if pap4[s] else "no match", len(pap4[s]), len(
            {a for a in tab4 if s in jr[a]})] + [yrs[y] for y in range(YEAR_MIN, YEAR_MAX + 1)])
    def yr_txt(ys):
        return ", ".join(str(y) for y in ys) if ys else "none"
    wtab = md_table(["Source display name", "Raw records with this name", "Outside the window (years)", "In window with no authorships (year, type)"],
                    [[sname, why[sname][0], yr_txt(why[sname][1]),
                      (", ".join(f"{y}, {t}" for y, t in why[sname][2]) if why[sname][2] else "none")] for sname in TARGET_JOURNALS])
    b3_rows = b3_all[:max(TABLE_N, tie_last + 1)]
    t1 = md_table(["Rank", "OpenAlex author id", "Name", "Papers", "Total citations", "Latest year", f"Institutions seen (up to {INST_SHOWN})"],
                  [[i + 1, a, name_of(tabP[a]), len(tabP[a]["papers"]), tabP[a]["cites"], tabP[a]["year"], short_inst(tabP[a])]
                   for i, a in enumerate(b1_top[:TABLE_N])])
    t2 = md_table(["Rank", "OpenAlex author id", "Name", "Polycrisis papers", "LtG papers", "Min", "Polycrisis without shared papers", "LtG without shared papers", "Total citations", "Bridge papers (06)", f"Institutions seen (up to {INST_SHOWN})"],
                  [[i + 1, r[0], name_of(tabAll[r[0]]), r[1], r[2], r[3], r[7], r[8], r[5], (", ".join(r[6]) if r[6] else "none"), short_inst(tabAll[r[0]])]
                   for i, r in enumerate(rows2[:TABLE_N])])
    bcol = "Betweenness"
    t3 = md_table(["Rank", "OpenAlex author id", "Name", "Degree", bcol, "Other authors tied with this row", "Polycrisis papers", f"Institutions seen (up to {INST_SHOWN})"],
                  [[i + 1, a, name_of(tabP[a]), deg[a], f"{btw[a]:.{BTW_DECIMALS}f}", other_tied(a), len(tabP[a]["papers"]), short_inst(tabP[a])]
                   for i, a in enumerate(b3_rows)])
    t3s = md_table(["Rank", "OpenAlex author id", "Name", "Degree", bcol, f"In primary top {TOP_N}"],
                   [[i + 1, a, name_of(tabP[a]), sdeg[a], f"{sb[a]:.{BTW_DECIMALS}f}", "yes" if a in b3 else "no"]
                    for i, a in enumerate(sens_top[:TABLE_N])])
    t4 = md_table(["Rank", "OpenAlex author id", "Name", "Target journal papers", "Total citations", "Journals", "In polycrisis", "In LtG", f"Institutions seen (up to {INST_SHOWN})"],
                  [[i + 1, a, name_of(tab4[a]), len(tab4[a]["papers"]), tab4[a]["cites"], ", ".join(jr[a]),
                    "yes" if a in tabP else "no", "yes" if a in tabL else "no", short_inst(tab4[a])]
                   for i, a in enumerate(b4[:TABLE_N])])
    tj = md_table(["Source display name", "Result", f"Merged papers {WINDOW}", "Authors"] + [str(y) for y in range(YEAR_MIN, YEAR_MAX + 1)], jr_rows)
    near_txt = ("; ".join(f"{s} ({c})" for s, c in sorted(near.items())) if near else "none")
    nodash_names = ", ".join(TARGET_JOURNALS)
    b3_method = (f"The network has one node per author id in the polycrisis corpus and one edge between two authors who share at least one merged polycrisis paper. "
                 f"It has {len(nodes):,} nodes and {len(edges):,} edges in {ncomp:,} connected components; the largest has {giant:,} nodes. "
                 f"Degree is the number of distinct coauthors. Betweenness is the networkx value for the unweighted graph (version {nx.__version__}), normalised, computed exactly. "
                 f"The list is ranked by betweenness, then degree, then author id in text order. "
                 f"The largest number of author ids on one paper is {max_auth:,}. Edges join only authors who have an author id, so authorships without an id add no edges. Every paper adds a full set of edges among its authors with an id, with no cap, so a long author list adds many edges. networkx {NX_TESTED} is required to run the script; the script stops with a message if networkx is missing.")
    lines = f"""# Outreach target lists from the corpus authorships

Generated by `scripts/08_outreach_targets.py`. Every number below is computed by the script and written to stdout or the CSVs. These are research leads. This note holds no contact details.

## Data and identity

- Papers are the merged records of `scripts/06_citation_bridge.py` (same loading, same `build_gmap`). {n_raw:,} distinct ids ({st['entries']:,} entries in the two raw files, {st['entries'] - n_raw:,} extra entries from repeated ids) became {len(meta):,} merged papers. Of these, {len(P):,} are in the polycrisis corpus, {len(L):,} in the LtG corpus and {len(both_units):,} in both.
- An author is an OpenAlex author id. Names are never used to match people. The name shown is the one most often seen for that id in the papers behind that list, and the first in text order if several are equally common. Institutions are ordered by how often they were seen, then in text order. Each list uses its own paper set, so one id can show a different name or institution order in different lists.
- A merged paper takes the authorships of all its raw records, joined by author id, so an author counts once per merged paper. Year and citation count are those of the merged paper's most cited record, as in 06.
- Authorships read: {st['authorships']:,}, counted over the {n_raw:,} distinct record ids (an id that appears in more than one entry is read once). Authorships with no author id: {st['without author id']:,}. These are left out of every list.
- Merged papers with no usable author id: {sum(1 for u in meta.index if not unit_auth.get(u)):,}.
- Distinct author ids: {len(tabP):,} in the polycrisis corpus, {len(tabL):,} in the LtG corpus, {len(tabAll):,} in either.
- Institutions are the institution names OpenAlex gives on the authorships. The CSV files list all of them; the tables below show up to {INST_SHOWN}, most frequent first.

## Basis of the figures

Total citations, latest year and institutions seen are taken from a different set of papers in each list. B1 and B3: the author's polycrisis papers. B2: the author's papers in either corpus. B4: the author's papers in the target journals in the {WINDOW} window only.

## B1. Most prolific polycrisis authors

Method: count merged polycrisis papers per author id. Rank by paper count, then total citations of those papers, then author id in text order. Top {TOP_N} kept.

Counts: {len(tabP):,} authors ranked. The paper count at rank {TOP_N} is {cut}; {pl(tie_out, 'author', 'authors')} outside the list also have that count. File: `data/outreach_B1.csv`.

{t1}

## B2. Authors who publish in both corpora

Method: an author id with at least one merged paper in the polycrisis corpus and at least one in the LtG corpus. A paper that is in both corpora counts in both. Rank by the smaller of the two counts, then total papers, then total citations, then author id. The bridge papers column lists this author's papers that appear in `data/bridge_papers.csv` from script 06{'' if bridge_set else ' (the file was not found, so the column is empty)'}.

Counts: {len(rows2):,} authors are in both corpora. {n_b2_with_bridge:,} of them have at least one bridge paper ({len(bridge_set):,} bridge papers in the file). {sum(1 for r in rows2 if r[3] == 1):,} have a smaller count of 1. File: `data/outreach_B2.csv`.

Effect of the shared papers: only {len(both_units):,} merged papers are in both corpora, yet {b2_has_shared:,} of the {len(rows2):,} authors have one of them, and {b2_only_shared:,} authors would not be in this list if those papers were left out of both counts. {b2_all_shared:,} of them have no other paper in either corpus. The two "without shared papers" columns show each author's counts with those papers removed.

{t2}

## B3. Central authors in the polycrisis coauthor network

Method: {b3_method} Betweenness is rounded to {BTW_DECIMALS} decimals before ranking; values equal after rounding count as tied and are then ordered by degree and author id.

Counts: {pl(overlap_b1_b3, 'author is', 'authors are')} in both the B1 and B3 lists. {pl(len(tie15_rows), 'author has', 'authors have')} the same degree and rounded betweenness as rank {TABLE_N}, at ranks {tie15_rows[0] + 1} to {tie_last + 1}; the table shows all of them, ordered only by author id. {pl(tie_cut, 'other author is', 'other authors are')} tied with rank {TOP_N}. Of the top {TOP_N}, {n_big_top:,} authors are coauthors on the {pl(len(big), 'paper', 'papers')} with the longest author list ({max_auth:,} author ids each): {', '.join(big)}. File: `data/outreach_B3.csv`.

{t3}

Sensitivity (the primary list above is unchanged): the same ranking with the {n_excl:,} polycrisis papers that have more than {SENS_MAX_AUTHORS} author ids left out. The network then has {len(sedges):,} edges. {sens_overlap:,} of its top {TOP_N} are also in the primary top {TOP_N}. {pl(s_tie_other, 'other author is', 'other authors are')} tied with rank {TOP_N} in this list, at ranks {s_tied_pos[0] + 1} to {s_tied_pos[-1] + 1}, so the cut at {TOP_N} {'falls inside a tie' if s_tie_other else 'is not inside a tie'}. File: `data/outreach_B3_sensitivity.csv`.

{t3s}

## B4. Authors in the target journals, {WINDOW}

Method: the target journals are the four journals chosen for this list (from the outreach plan): {nodash_names}. A paper matches when the source display name of its primary location equals one of these names exactly, for any raw record of the merged paper, in either corpus, with a publication year from {YEAR_MIN} to {YEAR_MAX}. The year is the merged paper's year (the year of its most cited record), not each raw record's own year. Of the {len(P | L):,} merged papers in either corpus, {n_diff_years:,} have records with differing years (counted over all their raw records, not only the B4 papers){yr_clause}. Rank authors by papers in this set, then total citations, then author id. The columns In polycrisis and In LtG say whether the author has any paper in that corpus, not necessarily a target journal paper.

Source names that contain a target name as a case insensitive substring but are not an exact match, and so were not used (papers in range): {near_txt}.

Counts: {len(b4_units):,} merged papers and {len(tab4):,} authors. {pl(len(b4_noauth), 'paper adds', 'papers add')} no author because no authorship on it has an author id ({', '.join(f'{u} ({noauth_n[u]} authorships on record)' for u in b4_noauth) if b4_noauth else 'none'}). File: `data/outreach_B4.csv`.

{tj}

Why some rows are small or empty. The raw files hold only the primary location of each record, so a paper listed under another location cannot be checked here. The small rows are explained by these records:

{wtab}

{t4}

## Ties

A full tie means every sort key before the author id is equal. Where a tie crosses the end of a {TABLE_N} row table or the cut at {TOP_N}, the rows beyond the cut have the same keys as the last row shown. The B3 table prints the whole tie at rank {TABLE_N}; the other tables stop at rank {TABLE_N}, and the CSV files hold the full lists (B1 and B3 hold the top 30).

{tietab}

## Work types

No list filters by work type. Errata, editorials and other non article records count as papers like any other. The type is OpenAlex's type for the merged paper's most cited record. Counts of papers that are not of type article, for the papers behind each list:

{ttab}

## Limitations

- OpenAlex author disambiguation makes errors. One person can be split across several ids, which lowers that person's counts. Two people can share one id, which raises them. Nothing here corrects this.
- {st['without author id']:,} of {st['authorships']:,} authorships (over the {n_raw:,} distinct record ids, repeated entries read once) have no author id and are not counted.
- The corpora are keyword searches, not the whole literature. {len(P):,} polycrisis papers and {len(L):,} LtG papers say nothing about authors who write on these topics under other terms. Counts describe the corpora only.
- B3 is built from polycrisis papers only. Coauthor links that exist only through LtG papers are not in the network. Papers with long author lists add many edges, and the primary list applies no cap; the sensitivity table shows the effect of a cap of {SENS_MAX_AUTHORS}.
- Ties are common in every list. The order inside a full tie comes from the author id, which carries no meaning. See the Ties section.
- Institution names come from OpenAlex affiliation matching and are not checked. The note tables show up to {INST_SHOWN} institutions per author and the CSV files list all of those seen, on the papers described under Basis of the figures. Treat them as unverified.
- B2 depends on the {len(both_units):,} papers that are in both corpora, as described above.
- The script needs these files in `data/`: polycrisis_papers_raw.json, ltg_refined_papers_raw.json, polycrisis_papers.csv, ltg_refined_papers.csv, references_refetch_2026-10-24.json and canon_metadata_cache.json (bridge_papers.csv is optional). `data/` is gitignored, so a clean clone must rebuild them (scripts 01 to 06) before this script can run.
- B4 uses the primary location only. A paper listed under another source name is not matched. Journal names were matched exactly as written, so renamed or abbreviated forms are missed.
- Citation totals are OpenAlex counts at the time of the data pull and are summed over an author's papers in the list.
- These are research leads, not a contact list. The script reads no email address, phone number or personal page, and none is stored in the outputs.
"""
    (DOC_DIR / "RESULTS_outreach_targets.md").write_text(lines, encoding="utf-8")
    print("Wrote data/outreach_B1..B4.csv and docs/RESULTS_outreach_targets.md")


if __name__ == "__main__":
    main()
