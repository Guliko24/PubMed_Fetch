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
| 1 — Foundations | Architecture sketch; can define state, node, edge, tool, workflow, agent | Done: sketch and definitions below |
| 2 — State and schemas | `Week2_Day2_Graph_Basics.ipynb`; claim schema in `src/pubmed_fetch/schemas.py` | Done 2026-11-18: passed in a fresh Colab runtime (invalid claims rejected; quote check True/False/False as expected) |
| 3 — Structured extraction | One abstract yields schema-valid claims; missing items are not guessed | |
| 4 — Connect retrieval | `src/pubmed_fetch/extract.py` + `Week2_Day4_Extraction.ipynb` | |
| 5 — Reliability | Pass/fail table; pytest passes; README updated | |

### Day 1 — Architecture sketch

Week 2 builds a **workflow**: the path is fixed in code, and the LLM is used inside one step.

```
query
  │
  ▼
hybrid search             BM25 + vector + RRF (search_utils.py, Week 1)
  │  top 3 results: {pmid, title, abstract}, one abstract at a time
  ▼
load_abstract             looks up the abstract for state["pmid"]          (Day 2, built)
  │
  ▼
extract                   LLM with structured output → list[Claim]          (Day 3, planned)
  │
  ▼
check_quotes              is each supporting_quote literally in the abstract? (Day 2, built)
  │
  ▼
result                    AbstractExtraction(pmid, claims) + quote_found per claim
```

The PMID comes from the pipeline, not the LLM. `Claim` holds only what the model reads in the
text (`cell_type`, `marker_genes`, `disease`, `claim_text`, `supporting_quote`).

### Day 1 — Definitions, with examples from this project

| Term | Definition | In PubMed_Fetch |
|---|---|---|
| State | The shared data passed through the graph; each node reads it and returns updates | `GraphState`: `pmid`, `claims`, `abstract`, `checked` |
| Node | A function that takes the state and returns only the keys it changes | `check_quotes` returns `{"checked": [...]}` |
| Edge | A connection that decides which node runs next; fixed or conditional | `load_abstract → check_quotes`, plus `START` and `END` |
| Tool | A function an LLM can choose to call, with arguments it writes | None in Week 2; hybrid search could become one later |
| Workflow | A graph whose path is fixed by code; an LLM may run inside a node | The Week 2 extraction graph |
| Agent | An LLM that decides at run time which step or tool comes next, often in a loop | Not built in Week 2 |

## Notes

- Until `pyproject.toml` arrives in Week 5, Colab notebooks need `src/` on `sys.path`
  to import `pubmed_fetch`.
- On Windows, two files whose names differ only in capitalisation collide. Keep file names
  unique regardless of case.
- Colab setup cells: `/content` survives "Restart session". The Day 2 setup cell switches an
  existing clone to `BRANCH` and prints the branch and commit; reuse it in new notebooks.
- Day 5 edge-case ideas from Day 2: abstracts with HTML tags (`<b>Rationale:</b>`) may fail the
  literal quote check; LLMs may return `""` instead of null; each marker gene should appear
  in its supporting quote.
