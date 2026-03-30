# Polycrisis & Limits to Growth — Independent Research

Independent research project mapping the academic landscape of polycrisis and limits-to-growth scholarship, with plans for original empirical studies.

## Project Structure

```
polycrisis-research/
├── scripts/
│   ├── 01_fetch_openalex.py      # Pull all papers from OpenAlex API
│   ├── 02_bibliometric_analysis.py  # Trends, top authors, citations
│   └── 03_vosviewer_export.py    # Format for VOSviewer network maps
├── data/                         # Generated data files (gitignored)
├── figures/                      # Generated charts (gitignored)
├── notebooks/                    # Jupyter notebooks for exploration
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

1. **Collect data:** `python scripts/01_fetch_openalex.py`
2. **Analyze:** `python scripts/02_bibliometric_analysis.py`
3. **Visualize:** Download [VOSviewer](https://www.vosviewer.com/) and import `data/vosviewer_polycrisis.csv`

## Research Plan

See `../polycrisis-independent-research.md` for full research plan.

### Publication targets
- **Fast (2-3 months):** Bibliometric mapping → *Sustainability* (MDPI)
- **Medium (4-6 months):** Granger causality networks or EWS detection → *Global Sustainability*
- **Ambitious (6-12 months):** Live World3 dashboard + paper → *Journal of Industrial Ecology*

### Key contacts
- Gaya Herrington — gayaherrington.com/contact (research collaboration)
- Michael Lawrence — Cascade Institute, editor at Global Sustainability
- Submit to: Global Sustainability "Polycrisis in the Anthropocene" collection
