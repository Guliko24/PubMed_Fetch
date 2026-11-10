# Progress

Project: PubMed_Fetch (the plan document calls it "Litminer"; it is the same project).
Plan: 8-Week Daily Plan (Weeks 1–4 build, Weeks 5–8 deploy on Google Cloud).
Update this file at the end of each working day so any session can pick up where the last one stopped.

## Week 1 — Retrieval (done)

- Day 1: PubMed fetch, saved to `pubmed_abstracts.json`
- Day 2: BM25 lexical search
- Day 3: vector search with `all-MiniLM-L6-v2`, embeddings saved as `.npy`
- Day 4: hybrid search with Reciprocal Rank Fusion
- Day 6: clean-up merged (PR #1): pinned requirements, `fetch_pubmed()` in `search_utils.py`,
  `.gitignore`, `.env.example`, superseded notebooks moved to `archive/`

## Week 2 — Agents and structured extraction (in progress)

Goal: one abstract in, schema-valid and PMID-linked claims out, with unsupported quotes flagged.
Built with LangGraph. New habit: code lives in a `src/pubmed_fetch/` package, not in notebooks.

| Day | Deliverable | Status |
|---|---|---|
| 1 — Foundations | Architecture sketch; can define state, node, edge, tool, workflow, agent | |
| 2 — State and schemas | `Week2_Day2_Graph_Basics.ipynb`; claim schema in `src/pubmed_fetch/schemas.py` | |
| 3 — Structured extraction | One abstract yields schema-valid claims; missing items are not guessed | |
| 4 — Connect retrieval | `src/pubmed_fetch/extract.py` + `Week2_Day4_Extraction.ipynb` | |
| 5 — Reliability | Pass/fail table; pytest passes; README updated | |

## Notes

- Until `pyproject.toml` arrives in Week 5, Colab notebooks need `src/` on `sys.path`
  to import `pubmed_fetch`.
- On Windows, two files whose names differ only in capitalisation collide. Keep file names
  unique regardless of case.
