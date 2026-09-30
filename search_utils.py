"""Reusable PubMed, BM25 and vector-search functions."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def load_pubmed_data(filename: str | Path) -> list[dict[str, Any]]:
    """Load the JSON file produced by Day 1."""
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Upload your Day 1 JSON file or set DATA_FILE "
            "in the notebook to its actual location."
        )

    with path.open("r", encoding="utf-8") as handle:
        documents = json.load(handle)

    if not isinstance(documents, list):
        raise ValueError("The JSON file must contain a list of documents.")

    cleaned = []

    for document in documents:
        if not isinstance(document, dict):
            continue

        abstract = str(document.get("abstract") or "").strip()
        if not abstract:
            continue

        cleaned.append(
            {
                "pmid": str(document.get("pmid") or ""),
                "title": str(document.get("title") or ""),
                "abstract": abstract,
            }
        )

    if not cleaned:
        raise ValueError("No documents with non-empty abstracts were found.")

    return cleaned


def _tokenize(text: str) -> list[str]:
    """Simple tokenization suitable for the Day 2 BM25 baseline."""
    return re.findall(r"[A-Za-z0-9]+(?:[-+][A-Za-z0-9]+)*", text.lower())


def build_bm25_index(documents: list[dict[str, Any]]):
    """Build a BM25 index from titles and abstracts."""
    from rank_bm25 import BM25Okapi

    if not documents:
        raise ValueError("Cannot build an index from an empty document list.")

    corpus = [
        _tokenize(f"{doc['title']} {doc['abstract']}")
        for doc in documents
    ]
    return BM25Okapi(corpus)


def search_bm25(
    bm25_index,
    documents: list[dict[str, Any]],
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return BM25-ranked documents."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    scores = bm25_index.get_scores(_tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {"score": float(scores[i]), "doc": documents[int(i)]}
        for i in top_indices
    ]


def get_embedding_model():
    """Load the model only when vector search actually needs it."""
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)

    return _model


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """Create one unit-normalized embedding per text."""
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def search_vectors(
    query: str,
    doc_embeddings: np.ndarray,
    documents: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rank documents by cosine similarity to the query."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    embeddings = np.asarray(doc_embeddings)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embeddings array; got shape {embeddings.shape}."
        )

    if len(documents) != embeddings.shape[0]:
        raise ValueError(
            "The number of embeddings must equal the number of documents. "
            f"Got {embeddings.shape[0]} embeddings and {len(documents)} documents."
        )

    model = get_embedding_model()
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    if embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError(
            "Embedding dimensions do not match. Your saved embeddings may "
            "have been generated using a different model."
        )

    similarities = embeddings @ query_embedding
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        {"score": float(similarities[i]), "doc": documents[int(i)]}
        for i in top_indices
    ]
def reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    top_k: int = 10,
    rank_constant: int = 60,
) -> list[dict]:
    """Fuse two ranked result lists using each paper's PMID as its ID."""
    if rank_constant < 1:
        raise ValueError("rank_constant must be at least 1.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    combined: dict[str, dict] = {}

    for method, results in (
        ("bm25", bm25_results),
        ("vector", vector_results),
    ):
        seen_in_this_list: set[str] = set()

        for rank, result in enumerate(results, start=1):
            doc = result["doc"]
            pmid = str(doc.get("pmid", "")).strip()

            if not pmid:
                raise ValueError("Every document needs a PMID for RRF.")

            if pmid in seen_in_this_list:
                raise ValueError(
                    f"PMID {pmid} occurs twice in the {method} results."
                )

            seen_in_this_list.add(pmid)

            if pmid not in combined:
                combined[pmid] = {
                    "pmid": pmid,
                    "doc": doc,
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "vector_rank": None,
                }

            combined[pmid]["rrf_score"] += 1.0 / (
                rank_constant + rank
            )
            combined[pmid][f"{method}_rank"] = rank

    return sorted(
        combined.values(),
        key=lambda item: (-item["rrf_score"], item["pmid"]),
    )[:top_k]
