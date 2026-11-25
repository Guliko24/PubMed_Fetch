"""Week 2 Day 2: a two-node LangGraph workflow with no LLM.

    START -> load_abstract -> check_quotes -> END

load_abstract looks up the abstract for the PMID in the state.
check_quotes runs the automatic evidence checks on every claim:
the quote must be in the abstract, and the claim's cell type, disease
and marker names must be in its quote. Day 4 reuses the same checks
after real LLM extraction.

The checks are literal. They show whether the evidence contains the
words, not whether it supports the claim; that still needs a human.
"""

import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from pubmed_fetch.schemas import Claim


class GraphState(TypedDict, total=False):
    # The PMID lives in the state, not in each claim: every claim in one run
    # comes from this one abstract.
    pmid: str
    claims: list[Claim]
    abstract: str
    checked: list[dict[str, Any]]


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def quote_in_abstract(quote: str, abstract: str) -> bool:
    """Literal check: the quote must appear in the abstract word for word.

    Only runs of whitespace are normalised, because PubMed abstracts and
    LLM output often differ in line breaks. Case and punctuation must match.
    An empty quote never counts as found.
    """
    quote = _normalise_whitespace(quote)
    if not quote:
        return False
    return quote in _normalise_whitespace(abstract)


def name_in_quote(name: str, quote: str) -> bool:
    """Is a cell type, disease or marker name written in the quote?

    Case is ignored, because a name may start a sentence in the abstract
    but not in the extracted field. Everything else must match.
    """
    name = _normalise_whitespace(name).lower()
    if not name:
        return False
    return name in _normalise_whitespace(quote).lower()


def evidence_checks(claim: Claim, abstract: str) -> dict[str, Any]:
    """Run every automatic check on one claim.

    A field that is empty (None or no markers) has nothing to check and is
    reported as None, not as a failure. automatic_checks_passed is True only
    when every check that applies passed.
    """
    quote = claim.supporting_quote
    checks = {
        "quote_found": quote_in_abstract(quote, abstract),
        "cell_type_in_quote": (
            None if claim.cell_type is None else name_in_quote(claim.cell_type, quote)
        ),
        "disease_in_quote": (
            None if claim.disease is None else name_in_quote(claim.disease, quote)
        ),
        "markers_in_quote": [
            name_in_quote(marker.as_written, quote) for marker in claim.marker_genes
        ],
    }
    results = [
        checks["quote_found"],
        checks["cell_type_in_quote"],
        checks["disease_in_quote"],
        *checks["markers_in_quote"],
    ]
    checks["automatic_checks_passed"] = all(r for r in results if r is not None)
    return checks


def build_graph(documents: list[dict[str, Any]]):
    """Compile the graph for a fixed set of documents (from load_pubmed_data)."""
    abstracts = {doc["pmid"]: doc["abstract"] for doc in documents}

    def load_abstract(state: GraphState) -> dict[str, Any]:
        pmid = state["pmid"]
        if pmid not in abstracts:
            raise KeyError(f"PMID {pmid} is not in the loaded documents.")
        return {"abstract": abstracts[pmid]}

    def check_quotes(state: GraphState) -> dict[str, Any]:
        checked = [
            {"claim": claim, **evidence_checks(claim, state["abstract"])}
            for claim in state.get("claims", [])
        ]
        return {"checked": checked}

    builder = StateGraph(GraphState)
    builder.add_node("load_abstract", load_abstract)
    builder.add_node("check_quotes", check_quotes)
    builder.add_edge(START, "load_abstract")
    builder.add_edge("load_abstract", "check_quotes")
    builder.add_edge("check_quotes", END)
    return builder.compile()
