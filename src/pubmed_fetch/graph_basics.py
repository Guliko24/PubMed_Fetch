"""Week 2 Day 2: a two-node LangGraph workflow with no LLM.

    START -> load_abstract -> check_quotes -> END

load_abstract looks up the abstract for the PMID in the state.
check_quotes marks each claim's supporting quote as found or not found
in that abstract. Day 4 reuses the same check after real LLM extraction.
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


def build_graph(documents: list[dict[str, Any]]):
    """Compile the graph for a fixed set of documents (from load_pubmed_data)."""
    abstracts = {doc["pmid"]: doc["abstract"] for doc in documents}

    def load_abstract(state: GraphState) -> dict[str, Any]:
        pmid = state["pmid"]
        if pmid not in abstracts:
            raise KeyError(f"PMID {pmid} is not in the loaded documents.")
        return {"abstract": abstracts[pmid]}

    def check_quotes(state: GraphState) -> dict[str, Any]:
        checked = []
        for claim in state.get("claims", []):
            checked.append(
                {
                    "claim": claim,
                    "quote_found": quote_in_abstract(
                        claim.supporting_quote, state["abstract"]
                    ),
                }
            )
        return {"checked": checked}

    builder = StateGraph(GraphState)
    builder.add_node("load_abstract", load_abstract)
    builder.add_node("check_quotes", check_quotes)
    builder.add_edge(START, "load_abstract")
    builder.add_edge("load_abstract", "check_quotes")
    builder.add_edge("check_quotes", END)
    return builder.compile()
