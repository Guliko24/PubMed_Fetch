"""Pydantic schemas for claims extracted from PubMed abstracts.

The LLM fills in Claim objects only. The PMID is added by the pipeline,
which already knows which abstract it sent, so the model cannot get it wrong.
"""

from pydantic import BaseModel, ConfigDict, Field


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
    marker_genes: list[str] = Field(
        default_factory=list,
        description=(
            "Marker gene symbols explicitly linked to this cell type in the evidence; "
            "preserve source spelling. Empty if absent."
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


class AbstractExtraction(BaseModel):
    """All claims extracted from one abstract (empty if it states none)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    pmid: str = Field(min_length=1)
    claims: list[Claim] = Field(default_factory=list)
