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
    Every number in the note is computed and printed here.

Run AFTER 06_citation_bridge.py has written data/canon_metadata_cache.json (from the repo root):
    .venv/bin/python scripts/08_outreach_targets.py

Output:
    data/outreach_B1.csv, data/outreach_B2.csv, data/outreach_B3.csv, data/outreach_B4.csv
    docs/RESULTS_outreach_targets.md
Network: none. Reads local files only. Runs are deterministic.
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
INST_SHOWN = 3          # institutions shown per author in the note tables (the CSV has all)

try:
    import networkx as nx
except ImportError:      # degree only
    nx = None

# ── Reuse 06 so counts agree ──────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location("bridge06", ROOT / "scripts" / "06_citation_bridge.py")
b06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b06)


def short(x):
    return str(x).rsplit("/", 1)[-1]


def merged_papers():
    """Merged units exactly as 06 builds them. Returns (unit meta, corpus membership of units, gmap)."""
    raw_meta, raw_members, raw_refs, _, _ = b06.load()
    ucount = Counter()
    pc = b06.ref_counts(raw_members["polycrisis"], raw_refs)
    lc = b06.ref_counts(raw_members["ltg"], raw_refs)
    for w in set(pc) | set(lc):
        ucount[w] = pc[w] + lc[w]
    pool = sorted(w for w in set(pc) | set(lc) if pc[w] >= b06.MIN_POOL_COUNT or lc[w] >= b06.MIN_POOL_COUNT)
    gmap, comps, _ = b06.build_gmap(raw_meta, b06.resolve_titles(pool), ucount)
    meta, members, _ = b06.make_context(raw_meta, raw_members, raw_refs, gmap)
    return meta, members, gmap, len(comps), len(raw_meta)


def load_authorships(gmap):
    """Per merged unit: {author id: {names, institutions}} plus sources and counts of lost authorships."""
    unit_auth = defaultdict(dict)       # unit -> author id -> {"names": Counter, "inst": Counter}
    unit_src = defaultdict(set)         # unit -> source display names
    unit_nauth = defaultdict(int)
    stats = Counter()
    seen_rec = set()
    for f in ["polycrisis_papers_raw.json", "ltg_refined_papers_raw.json"]:
        for w in json.load(open(DATA_DIR / f)):
            wid = short(w["id"])
            u = gmap.get(wid, wid)
            src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
            if src:
                unit_src[u].add(src)
            if wid in seen_rec:
                continue                # same record in both files: authorships counted once
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
    return unit_auth, unit_src, stats


def aid_key(a):
    return a


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
    meta, members, gmap, n_groups, n_raw = merged_papers()
    unit_auth, unit_src, st = load_authorships(gmap)
    P, L = members["polycrisis"], members["ltg"]
    both_units = P & L
    print(f"Raw records: {n_raw}; merged groups: {n_groups}; merged papers: {len(meta)}")
    print(f"Merged papers: polycrisis {len(P)}, LtG {len(L)}, in both {len(both_units)}")
    print(f"Authorships read: {st['authorships']}; without author id: {st['without author id']}")
    print(f"Papers with no authorship at all: {sum(1 for u in meta.index if not unit_auth.get(u))}")

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
    bridge_ids = {}
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
        rows2.append((a, np_, nl, min(np_, nl), len(allp), tabAll[a]["cites"], [p for p in allp if p in bridge_set]))
    rows2.sort(key=lambda r: (-r[3], -r[4], -r[5], r[0]))
    write_csv(DATA_DIR / "outreach_B2.csv",
              ["rank", "openalex_author_id", "name", "polycrisis_papers", "ltg_papers", "min_count", "total_papers",
               "total_cited_by_count", "latest_year", "bridge_paper_ids", "bridge_paper_titles", "institutions"],
              [[i + 1, r[0], name_of(tabAll[r[0]]), r[1], r[2], r[3], r[4], r[5], tabAll[r[0]]["year"],
                "; ".join(r[6]), "; ".join(str(bridge_title.get(p, "")) for p in r[6]), "; ".join(insts(tabAll[r[0]]))]
               for i, r in enumerate(rows2)])
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
    btw = {}
    if nx is not None:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(sorted(edges))
        btw = nx.betweenness_centrality(G, normalized=True)
        ncomp = nx.number_connected_components(G)
        giant = max(len(c) for c in nx.connected_components(G))
        print(f"B3: networkx {nx.__version__}; nodes {G.number_of_nodes()}, edges {G.number_of_edges()}, components {ncomp}, largest component {giant}")
        key = lambda a: (-round(btw[a], 8), -deg[a], a)
        b3 = sorted(nodes, key=key)[:TOP_N]
    else:
        ncomp = giant = None
        print(f"B3: networkx not installed; degree only. nodes {len(nodes)}, edges {len(edges)}")
        b3 = sorted(nodes, key=lambda a: (-deg[a], a))[:TOP_N]
    print(f"B3: largest author list on one merged polycrisis paper: {max_auth}")
    write_csv(DATA_DIR / "outreach_B3.csv",
              ["rank", "openalex_author_id", "name", "degree", "betweenness", "polycrisis_papers", "institutions"],
              [[i + 1, a, name_of(tabP[a]), deg[a], (f"{btw[a]:.8f}" if nx else ""), len(tabP[a]["papers"]), "; ".join(insts(tabP[a]))]
               for i, a in enumerate(b3)])
    top_deg = sorted(nodes, key=lambda a: (-deg[a], a))[:TOP_N]
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
    b4 = sorted(tab4, key=lambda a: (-len(tab4[a]["papers"]), -tab4[a]["cites"], a))
    jr = {a: sorted({s for s in TARGET_JOURNALS if any(p in pap4[s] for p in tab4[a]["papers"])}) for a in tab4}
    write_csv(DATA_DIR / "outreach_B4.csv",
              ["rank", "openalex_author_id", "name", "papers_2024_2026_in_target_journals", "total_cited_by_count",
               "latest_year", "journals", "in_polycrisis", "in_ltg", "institutions"],
              [[i + 1, a, name_of(tab4[a]), len(tab4[a]["papers"]), tab4[a]["cites"], tab4[a]["year"], "; ".join(jr[a]),
                "yes" if a in tabP else "no", "yes" if a in tabL else "no", "; ".join(insts(tab4[a]))]
               for i, a in enumerate(b4)])
    print(f"B4: papers {len(b4_units)}; authors {len(tab4)}; per journal {{{', '.join(f'{s}: {len(pap4[s])}' for s in TARGET_JOURNALS)}}}")
    print(f"B4: near miss source names (contain a target name, not an exact match): {dict(near)}")

    # ── Note ──
    jr_rows = []
    for s in TARGET_JOURNALS:
        yrs = {y: name_hits[(s, y)] for y in range(YEAR_MIN, YEAR_MAX + 1)}
        jr_rows.append([s, "matched" if pap4[s] else "no match", len(pap4[s]), len(
            {a for a in tab4 if s in jr[a]})] + [yrs[y] for y in range(YEAR_MIN, YEAR_MAX + 1)])
    # Source-name matching uses primary_location of every raw record of a merged paper.
    t1 = md_table(["Rank", "OpenAlex author id", "Name", "Papers", "Total citations", "Latest year", "Institutions seen (up to 3)"],
                  [[i + 1, a, name_of(tabP[a]), len(tabP[a]["papers"]), tabP[a]["cites"], tabP[a]["year"], short_inst(tabP[a])]
                   for i, a in enumerate(b1_top[:TABLE_N])])
    t2 = md_table(["Rank", "OpenAlex author id", "Name", "Polycrisis papers", "LtG papers", "Min", "Total citations", "Bridge papers (06)", "Institutions seen (up to 3)"],
                  [[i + 1, r[0], name_of(tabAll[r[0]]), r[1], r[2], r[3], r[5], (", ".join(r[6]) if r[6] else "none"), short_inst(tabAll[r[0]])]
                   for i, r in enumerate(rows2[:TABLE_N])])
    bcol = "Betweenness" if nx else "Betweenness (not computed)"
    t3 = md_table(["Rank", "OpenAlex author id", "Name", "Degree", bcol, "Polycrisis papers", "Institutions seen (up to 3)"],
                  [[i + 1, a, name_of(tabP[a]), deg[a], (f"{btw[a]:.6f}" if nx else "n/a"), len(tabP[a]["papers"]), short_inst(tabP[a])]
                   for i, a in enumerate(b3[:TABLE_N])])
    t4 = md_table(["Rank", "OpenAlex author id", "Name", "Target journal papers", "Total citations", "Journals", "In polycrisis", "In LtG", "Institutions seen (up to 3)"],
                  [[i + 1, a, name_of(tab4[a]), len(tab4[a]["papers"]), tab4[a]["cites"], ", ".join(jr[a]),
                    "yes" if a in tabP else "no", "yes" if a in tabL else "no", short_inst(tab4[a])]
                   for i, a in enumerate(b4[:TABLE_N])])
    tj = md_table(["Source display name", "Result", "Merged papers 2024 to 2026", "Authors"] + [str(y) for y in range(YEAR_MIN, YEAR_MAX + 1)], jr_rows)
    near_txt = ("; ".join(f"{s} ({c})" for s, c in sorted(near.items())) if near else "none")
    nodash_names = ", ".join(TARGET_JOURNALS)
    if nx:
        b3_method = (f"The network has one node per author id in the polycrisis corpus and one edge between two authors who share at least one merged polycrisis paper. "
                     f"It has {len(nodes):,} nodes and {len(edges):,} edges in {ncomp:,} connected components; the largest has {giant:,} nodes. "
                     f"Degree is the number of distinct coauthors. Betweenness is the networkx value for the unweighted graph (version {nx.__version__}), normalised, computed exactly. "
                     f"The list is ranked by betweenness, then degree, then author id in text order. "
                     f"The largest author list on one paper has {max_auth:,} authors. Every paper adds a full set of edges among its authors, with no cap, so a long author list adds many edges.")
    else:
        b3_method = (f"networkx is not installed in this environment, so only degree is computed. The network has {len(nodes):,} nodes and {len(edges):,} edges. "
                     f"The list is ranked by degree, then author id in text order.")
    lines = f"""# Outreach target lists from the corpus authorships

Generated by `scripts/08_outreach_targets.py`. Every number below is computed and printed by that script. These are research leads. This note holds no contact details.

## Data and identity

- Papers are the merged records of `scripts/06_citation_bridge.py` (same loading, same `build_gmap`). {n_raw:,} raw records became {len(meta):,} merged papers. Of these, {len(P):,} are in the polycrisis corpus, {len(L):,} in the LtG corpus and {len(both_units):,} in both.
- An author is an OpenAlex author id. Names are never used to match people. The name shown is the one most often seen for that id.
- A merged paper takes the authorships of all its raw records, joined by author id, so an author counts once per merged paper. Year and citation count are those of the merged paper's most cited record, as in 06.
- Authorships read: {st['authorships']:,}. Authorships with no author id: {st['without author id']:,}. These are left out of every list.
- Merged papers with no usable author id: {sum(1 for u in meta.index if not unit_auth.get(u)):,}.
- Distinct author ids: {len(tabP):,} in the polycrisis corpus, {len(tabL):,} in the LtG corpus, {len(tabAll):,} in either.
- Institutions are the institution names OpenAlex gives on the authorships. The CSV files list all of them; the tables below show up to {INST_SHOWN}, most frequent first.

## B1. Most prolific polycrisis authors

Method: count merged polycrisis papers per author id. Rank by paper count, then total citations of those papers, then author id in text order. Top {TOP_N} kept.

Counts: {len(tabP):,} authors ranked. The paper count at rank {TOP_N} is {cut}; {pl(tie_out, 'author', 'authors')} outside the list also have that count. File: `data/outreach_B1.csv`.

{t1}

## B2. Authors who publish in both corpora

Method: an author id with at least one merged paper in the polycrisis corpus and at least one in the LtG corpus. A paper that is in both corpora counts in both. Rank by the smaller of the two counts, then total papers, then total citations, then author id. The bridge papers column lists this author's papers that appear in `data/bridge_papers.csv` from script 06{'' if bridge_set else ' (the file was not found, so the column is empty)'}.

Counts: {len(rows2):,} authors are in both corpora. {n_b2_with_bridge:,} of them have at least one bridge paper ({len(bridge_set):,} bridge papers in the file). {sum(1 for r in rows2 if r[3] == 1):,} have a smaller count of 1. File: `data/outreach_B2.csv`.

{t2}

## B3. Central authors in the polycrisis coauthor network

Method: {b3_method}{'' if nx else ''}

Counts: {pl(overlap_b1_b3, 'author is', 'authors are')} in both the B1 and B3 lists. File: `data/outreach_B3.csv`.

{t3}

## B4. Authors in the target journals, 2024 to 2026

Method: the target journals are the ones docs/PUBLICATION_STRATEGY.md names as submission targets: {nodash_names}. A paper matches when the source display name of its primary location equals one of these names exactly, for any raw record of the merged paper, in either corpus, with a publication year from {YEAR_MIN} to {YEAR_MAX}. Rank authors by papers in this set, then total citations, then author id.

Source names that contain a target name but are not an exact match, and so were not used (papers in range): {near_txt}.

Counts: {len(b4_units):,} merged papers and {len(tab4):,} authors. File: `data/outreach_B4.csv`.

{tj}

{t4}

## Limitations

- OpenAlex author disambiguation makes errors. One person can be split across several ids, which lowers that person's counts. Two people can share one id, which raises them. Nothing here corrects this.
- {st['without author id']:,} of {st['authorships']:,} authorships have no author id and are not counted.
- The corpora are keyword searches, not the whole literature. {len(P):,} polycrisis papers and {len(L):,} LtG papers say nothing about authors who write on these topics under other terms. Counts describe the corpora only.
- B3 covers only the polycrisis corpus, so ties outside it do not show. Large author lists add many edges, and no cap was applied.
- B4 uses the primary location only. A paper listed under another source name is not matched. Journal names were matched exactly as written, so renamed or abbreviated forms are missed.
- Citation totals are OpenAlex counts at the time of the data pull and are summed over an author's papers in the list.
- These are research leads, not a contact list. The script reads no email address, phone number or personal page, and none is stored in the outputs.
"""
    (DOC_DIR / "RESULTS_outreach_targets.md").write_text(lines, encoding="utf-8")
    print("Wrote data/outreach_B1..B4.csv and docs/RESULTS_outreach_targets.md")


if __name__ == "__main__":
    main()
