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
| 3 — Structured extraction | One abstract yields schema-valid claims; missing items are not guessed | Done 2026-11-25 after one revision. Second run (`ee31c61`, `gemini-3.6-flash`, 6 cases): schema-valid, 14/14 claims passed the automatic checks, 11/14 accepted on manual review, 3 `needs_review`; empty case gave 0 claims. One run, not a measured accuracy. Record: `results/week2_day3/20261125T151227Z_gemini-3.6-flash.json` |
| 4 — Connect retrieval | `src/pubmed_fetch/extract.py` + `Week2_Day4_Extraction.ipynb` | |
| 5 — Reliability | Pass/fail table; pytest passes; README updated | |

### Day 3 — Observations from one run (not measured accuracy)

| Abstract | Claims | Quotes found | Notes |
|---|---|---|---|
| PMID 34582785 (brain reward circuitry) | 4 | 4/4 | `marker_genes` empty in all claims; claim 4 puts "D1- and D2-expressing MSNs" in one `cell_type` |
| PMID 40585969 (hypoxia organoids, has `<b>` tags) | 5 | 5/5 | `disease: fetal hypoxia` in all 5 claims, but none of the quotes mention it |
| Made-up text naming no cell type | 0 | — | Expected 0 |

- `gemini-3.6-flash` ignores `temperature` (fixed sampling defaults), so runs may vary.
- Rule breaks to test on Day 5: disease taken from outside the quote; two populations merged
  in one `cell_type` despite "keep separate populations separate".
- Not exercised: no quote crossed a `<b>` tag, so the "copy markup" rule is still untested.
- `marker_genes` is empty because the rules only accept genes the evidence calls *markers*;
  decide on Day 5 whether "D1-expressing" should count. (Decided below.)
- Colab: `google-auth` version warning is expected (see requirements.txt); after changing
  `BRANCH` or requirements, use "Disconnect and delete runtime", not "Restart session".
- A second review found the main gap: **a quote being in the abstract does not show that it
  supports the claim** (e.g. "following hypoxia-reoxygenation" is not in its quote; pericyte
  and fibroblast claims are stronger than the grouped sentence they came from).

### Day 3 — Decisions (2026-11-25)

- **Evidence:** a claim's evidence is its `supporting_quote`. Its cell type, disease and marker
  names must appear in the quote; a disease named only elsewhere in the abstract is null.
- **Grouped cell types:** split them, one claim per named cell type, but each claim states only
  what the quote says about the group, never more strongly.
- **"Characterized" / "identified":** a claim only when the evidence also states what was found
  (marker genes or a new property).
- **Marker gene:** a gene or receptor whose expression the evidence explicitly uses to define,
  identify or distinguish a cell type or subtype (so "D1-expressing MSNs" counts).
  `MarkerGene(as_written="D1", gene_symbol="DRD1")`: `as_written` is copied from the quote;
  `gene_symbol` is the HGNC symbol the LLM maps it to (null if not certain), because receptor
  names will not match in Week 3. `gene_symbol` is the one place the model may use outside
  knowledge and no quote can verify it, so it must be checked against HGNC before Week 3
  relies on it. `symbol_status` (`unverified` / `verified`) records that check; it is hidden
  from the LLM's schema and only code sets it to `verified`. (Considered and set aside for
  now: code-filled symbols with an `ontology_status` field.)
- **Checks:** automatic checks are literal (quote in abstract; names in quote). Support for
  the claim is judged by hand: `accepted`, `needs_review` or `rejected` per claim.
- **Run records:** every run is saved as its own JSON file (model, time, commit, packages,
  prompt, inputs, claims, checks, review) and never overwritten.

### Day 3 — Second run (2026-11-25, commit `ee31c61`, `gemini-3.6-flash`)

Full record (prompt, inputs, claims, checks, review): `results/week2_day3/20261125T151227Z_gemini-3.6-flash.json`.

14 claims over 6 cases; all automatic checks passed. Fixed since the first run: no disease
taken from outside the quote; D1/D2 captured as markers (`DRD1`/`DRD2`, unverified); the
markup rule exercised (`html#1` kept `<b>`/`<i>` and its quote was found); negation kept;
empty case gave 0 claims. Manual review: 11 accepted, 3 `needs_review`:

- `real_hypoxia#1`, `#2`: `claim_text` adds "following hypoxia-reoxygenation", which is not in
  the quote. The automatic checks cannot see this; they do not check `claim_text`.
- `grouped#1`: the group label "Non-neural cells" was extracted as its own cell type
  (expected 2 claims, got 3).

Open questions: should a group label be a claim alongside its members? Should `real_msn`
split by disease as well as by marker (it gave 4 claims)?

### DEFERRED TO WEEK 3 (graph design): relation, condition and involved genes

The schema captures who (cell type, markers, disease) but not what happened or under which
condition. Decided 2026-11-25 to add these with the Week 3 graph design, not now:

- **relation / direction**: "reduced", "enriched for genetic risk", "did not alter"
- **condition / context**: genotype or perturbation, e.g. "*Scn1a* mutant cortex",
  "hypoxia-reoxygenation"
- **genes involved but not markers**: mechanism genes, e.g. Notch signalling in
  `real_hypoxia#4`/`#5` (a gene family, NOTCH1-4, so no single HGNC symbol)

Starting idea: plain-text `relation` and `condition` fields that must appear in the quote
(same `name_in_quote` check), turned into typed graph edges in Week 3.

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
