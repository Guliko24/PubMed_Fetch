# PubMed_Fetch
# this is an Ontology-Aware Literature Miner

A work-in-progress biomedical literature-mining project. The current Week 1 prototype retrieves a small set of PubMed abstracts and compares lexical search (BM25), semantic vector search, and hybrid retrieval using Reciprocal Rank Fusion (RRF).

The longer-term goal is to use the retrieved abstracts as evidence for structured extraction of cell types, marker genes, and diseases, followed by ontology validation. **Those extraction and validation stages are not part of the current Week 1 implementation.**

## What the current prototype does

1. Fetches PubMed records and saves their PMIDs, titles, and abstracts as JSON.
2. Builds a BM25 index for keyword-based retrieval.
3. Creates document embeddings with `sentence-transformers/all-MiniLM-L6-v2` and ranks documents by vector similarity.
4. Combines the BM25 and vector rankings using RRF.

The vector search operates on **the locally saved PubMed subset**, not on all of PubMed. The saved abstracts determine which papers can appear in the results.

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── .env.example                 # environment variables to set (no values)
├── search_utils.py              # fetch, BM25, vector search, RRF
├── pubmed_abstracts.json
├── day3_embeddings_<fingerprint>.npy
├── Week1_Day1_PubMed_Fetch.ipynb
├── Week1_Day2_Lexical_Search.ipynb
├── Week1_Day3_Vector_Search.ipynb
├── Week1_Day4_Hybrid_search.ipynb
├── How_to_clone_Git_to_colab.ipynb
└── archive/                     # superseded notebooks; do not run
```

`pubmed_abstracts.json` is the saved input corpus. The `.npy` file contains precomputed document embeddings. Its rows must correspond to the JSON documents **in exactly the same order** and must have been generated using the model and text preparation documented below.

The notebooks in `archive/` are kept for history only. They write an older version of `search_utils.py`, so running them would overwrite the current one.


## Run in Google Colab

Open the notebooks in numerical order. In a fresh Colab runtime, clone this repository and install its dependencies:

```python
!git clone https://github.com/Guliko24/PubMed_Fetch.git
%pip -q install -r /content/PubMed_Fetch/requirements.txt
```

Add the repository to Python's import path:

```python
import sys

repo_dir = "/content/PubMed_Fetch"
sys.path.insert(0, repo_dir)

import search_utils
print(search_utils.__file__)
```

The printed path should point to this repository's `search_utils.py`. If you update the utility on GitHub after cloning, pull the new commit and restart the runtime before running the notebook again.

Day 3 and Day 4 require `pubmed_abstracts.json` to be available at the path specified in those notebooks. Day 4 also requires the matching Day 3 embeddings file, unless you regenerate the document embeddings.

### Settings for fetching from PubMed

Only Day 1 contacts PubMed. NCBI requires a contact email, which the Day 1 notebook passes to `search_utils.fetch_pubmed(..., email=NCBI_EMAIL)`. If `email` is not passed, the function reads the `NCBI_EMAIL` environment variable instead. An optional `NCBI_API_KEY` raises the rate limit from 3 to 10 requests per second. `.env.example` lists both; never commit an API key.

## Methods

### BM25

BM25 ranks documents using query terms found in the indexed text. It provides a lexical baseline that can reward exact terminology. See the Day 2 notebook and `search_utils.py` for the actual tokenization and text fields used.

### Vector search

The Day 3 notebook embeds each document as its title and abstract joined into one text (`f"{title}. {abstract}"`) using `sentence-transformers/all-MiniLM-L6-v2`. It embeds the query with the same model and ranks documents by cosine similarity.

Do not use an embeddings file created from a different document order, a different model, or a different choice of text fields.

### Hybrid search

Day 4 runs BM25 and vector retrieval for the same query and combines their rankings with Reciprocal Rank Fusion:

```text
RRF(document) = Σ 1 / (rank_constant + rank_in_result_list)
```

The sum includes the ranked lists in which the document appears. RRF combines **ranks**, not raw BM25 and cosine-similarity scores. The prototype uses a rank constant of 60.

search_utils.reciprocal_rank_fusion() combines the ranked outputs of search_bm25() and search_vectors().

## Current status and limitations

- This is a small-corpus prototype, not a search service covering the full PubMed database.
- The quality of the results depends on the PubMed query used to create the saved corpus.
- The current general-purpose embedding model is a baseline; a biomedical/scientific embedding model is a possible later comparison.
- The presence of BM25, vector, and hybrid rankings does **not** by itself establish that hybrid retrieval performs better. That requires relevance judgments on a fixed test set.
- Cell-type/marker/disease extraction, Cell Ontology matching, and graph storage are planned future stages, not completed features.

## Next steps

1. Test the notebooks from a fresh Colab runtime.
2. Create a small set of queries with manually judged relevant PMIDs.
3. Compare BM25, vector, and hybrid results using a retrieval metric such as Precision@3.
4. Feed retrieved abstracts into a structured extraction workflow.

## Data and reproducibility

PubMed records are identified by PMID in the JSON file. The saved JSON makes the small test corpus reproducible without fetching records on every run. Record the embedding model and the exact text used to generate the `.npy` file before comparing results.

Do not commit API keys, passwords, or other credentials to this repository.
