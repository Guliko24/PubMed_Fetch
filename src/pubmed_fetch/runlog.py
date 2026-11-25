"""Save everything needed to reproduce and review one extraction run.

One JSON file per run, named by UTC timestamp and model, never overwritten:
a revised prompt produces a new file next to the old ones.
"""

import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PACKAGES = ("langchain-google-genai", "google-genai", "langchain-core", "langgraph", "pydantic")
REVIEW_STATUSES = ("accepted", "needs_review", "rejected")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_commit(repo_dir: str | Path) -> str | None:
    """Short hash of the code that produced the run, or None outside a clone."""
    try:
        return subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_run_record(
    *,
    model: str,
    prompt: str,
    results: list[dict[str, Any]],
    review: dict[str, tuple[str, str]],
    repo_dir: str | Path,
) -> dict[str, Any]:
    """Build the record for one run.

    results: one dict per input with keys label, pmid, abstract, expect,
    extraction (AbstractExtraction) and checks (one evidence_checks dict per claim).
    review: manual statuses keyed "label#claim number", e.g. "real_msn#2";
    claims without an entry are recorded as "unreviewed".
    """
    for key, (status, _note) in review.items():
        if status not in REVIEW_STATUSES:
            raise ValueError(f"{key}: status must be one of {REVIEW_STATUSES}, got {status!r}")

    cases = []
    for result in results:
        claims = []
        for number, (claim, checks) in enumerate(
            zip(result["extraction"].claims, result["checks"]), start=1
        ):
            status, note = review.get(f"{result['label']}#{number}", ("unreviewed", ""))
            claims.append(
                {
                    "number": number,
                    **claim.model_dump(),
                    "checks": checks,
                    "review": {"status": status, "note": note},
                }
            )
        cases.append(
            {
                "label": result["label"],
                "pmid": result["pmid"],
                "expect": result["expect"],
                "abstract": result["abstract"],
                "claims": claims,
            }
        )

    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "git_commit": git_commit(repo_dir),
        "packages": {name: _package_version(name) for name in PACKAGES},
        "prompt": prompt,
        "cases": cases,
    }


def save_run_record(record: dict[str, Any], folder: str | Path) -> Path:
    """Write the record as a new JSON file; refuse to overwrite an existing one."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = record["timestamp_utc"].replace("-", "").replace(":", "")
    path = folder / f"{stamp}_{record['model']}.json"
    if path.exists():
        raise FileExistsError(f"{path} already exists; run records are never overwritten.")
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
