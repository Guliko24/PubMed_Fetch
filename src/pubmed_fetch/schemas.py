"""Pydantic schemas for claims extracted from PubMed abstracts.

The LLM fills in Claim objects only. The PMID is added by the pipeline,
which already knows which abstract it sent, so the model cannot get it wrong.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema


class MarkerGene(BaseModel):
    """One marker named in the evidence, plus the gene symbol to match it on.

    as_written must appear in the claim's supporting_quote (checked by
    graph_basics.evidence_checks). gene_symbol is the HGNC symbol used for
    matching in Week 3, e.g. "D1" (a receptor name) -> "DRD1".
    symbol_status records whether code has checked gene_symbol against HGNC.
    It is hidden from the LLM's schema, so the model cannot mark its own
    mapping as verified.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    as_written: str = Field(
        min_length=1,
        description=(
            "Gene, receptor, or marker label exactly as it appears in the "
            "supporting quote, including capitalisation and hyphens."
        ),
    )
    gene_symbol: str | None = Field(
        default=None,
        description=(
            "Official HGNC gene symbol for the as_written name "
            "(for example, D1 -> DRD1); use null if not certain."
        ),
    )
    symbol_status: SkipJsonSchema[Literal["unverified", "verified"]] = "unverified"


class Claim(BaseModel):
    """One finding stated in one abstract, with the text that supports it.

    Biomedical fields are optional: None (or an empty list) means the
    abstract does not state it, never a guess.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    cell_type: str | None = Field(
        default=None,
        description="One cell type explicitly named in this claim's evidence; otherwise null.",
    )
    marker_genes: list[MarkerGene] = Field(
        default_factory=list,
        description=(
            "Genes or receptors whose expression the evidence explicitly uses to define, "
            "identify or distinguish this cell type or subtype. Empty if absent."
        ),
    )
    disease: str | None = Field(
        default=None,
        description="Disease explicitly linked to this claim in the evidence; otherwise null.",
    )
    claim_text: str = Field(
        min_length=1,
        description=(
            "One short statement supported by the quote, preserving qualifications "
            "and direction of the reported relationship."
        ),
    )
    supporting_quote: str = Field(
        min_length=1,
        description="Exact sentence or phrase copied from the abstract that states the claim.",
    )


class ExtractedClaims(BaseModel):
    """What the LLM returns for one abstract: claims only, no PMID."""

    model_config = ConfigDict(extra="forbid")

    claims: list[Claim] = Field(
        default_factory=list,
        description="Every claim the abstract states; an empty list if it states none.",
    )


class AbstractExtraction(BaseModel):
    """All claims extracted from one abstract (empty if it states none)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    pmid: str = Field(min_length=1)
    claims: list[Claim] = Field(default_factory=list)
