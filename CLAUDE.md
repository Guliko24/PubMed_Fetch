## Project
PubMed_Fetch: biomedical literature miner (PubMed → BM25 + vector + RRF → LLM extraction).
Learning plan: Weeks 1–4 build the pipeline; Weeks 5–8 deploy it on Google Cloud.
Current progress and the active week's plan: see docs/PROGRESS.md.

## How code runs
- Notebooks run in Google Colab and clone `main` from GitHub in their setup cell.
- To test a branch, add `"-b", "<branch>"` to the clone line, use a fresh runtime,
  and restart the session after changing `search_utils.py`.
- Test the new version on a branch before merging; never merge to fix a test problem.
- The NCBI email stays in the notebooks for now (owner's choice).

## Conventions
- Reusable code lives in modules, notebooks are demos only.
- Pin library versions in requirements.txt; no Colab magics in it.
- Never claim a metric that hasn't been measured.
