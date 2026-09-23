"""Analysis viewers this workspace contributes to the vivarium-workbench.

The workbench discovers a workspace's viewers by importing
``<package>.workbench_viewers`` and calling ``get_viewers(ws_root)`` (see
vivarium_workbench.lib.analysis_viewers). Each returned dict describes a tool
shown under the **Analyses** tab.

We contribute the **BioModels corpus explorer**: an interactive, model-by-model
browser over the full multi-simulator reproduction dataset (every engine's
trajectory overlaid per variable). It is a self-contained static app hosted on
Cloudflare R2, so its target carries an external ``href`` that opens directly in
both the live workbench and the read-only published dashboard — no local launch
backend needed (mirrors v2ecoli's hosted 3D viewer).
"""
from __future__ import annotations

from pathlib import Path

# Self-contained explorer app (reads {base}/index.json + {base}/series/<id>.json
# from the same R2 prefix). Committed source lives at
# viva_biomodels/viewers/corpus_explorer.html and is uploaded here.
CORPUS_EXPLORER_URL = (
    "https://pub-eb913fbbdc584bd7add047c823570b13.r2.dev"
    "/biomodels-corpus/corpus_explorer.html"
)


def _corpus_index(ws_root) -> Path:
    return Path(ws_root) / "datasets" / "corpus_all_engines" / "index.json"


def _has_corpus(ws_root) -> bool:
    """Show the explorer only when the committed corpus dataset is present."""
    return _corpus_index(ws_root).is_file()


def _corpus_targets(ws_root) -> list:
    """One target: the full-corpus reproduction, deep-linked to the hosted app.

    Reads the committed index.json for a live model/engine count in the label,
    falling back to a static label if it can't be read.
    """
    label = "Full BioModels corpus"
    detail = "every model across every simulator"
    try:
        import json
        idx = json.loads(_corpus_index(ws_root).read_text(encoding="utf-8"))
        n = idx.get("n_models")
        engines = idx.get("engines") or []
        if n:
            label = f"Full BioModels corpus — {n} models"
        if engines:
            detail = f"{len(engines)} engines: {', '.join(engines)}"
    except Exception:
        pass
    return [{
        "study": "full-corpus-reproduction",
        "label": label,
        "detail": detail,
        "href": CORPUS_EXPLORER_URL,
    }]


def get_viewers(ws_root) -> list:
    """Contribute the BioModels corpus explorer to the Analyses tab."""
    return [
        {
            "id": "corpus-explorer",
            "title": "BioModels corpus explorer",
            "description": (
                "Browse the full multi-simulator BioModels reproduction "
                "model-by-model: every engine's trajectory (COPASI, Tellurium, "
                "simbio, AMICI, PySCeS) overlaid per variable, so agreement and "
                "divergence are directly visible."
            ),
            "kind": "launcher",
            "applies": _has_corpus,
            "targets": _corpus_targets,
        },
    ]
