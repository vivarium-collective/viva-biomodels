"""Run provenance for the batch-compare diagnostics tab.

Captures *where* and *when* a batch ran and *which build* of each simulator
produced the numbers: host, platform, Python, an ISO-8601 UTC timestamp, plus
per-simulator library + wrapper versions and git commit hashes (so a result can
be traced back to an exact checkout).

Everything is best-effort: a missing module, a non-git install, or a git binary
that isn't on PATH degrades to ``None`` rather than raising — provenance must
never break a simulation run.
"""
from __future__ import annotations

import importlib
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# sim name -> (wrapper module, wrapper dist, candidate lib modules, lib dist)
_SIM_PACKAGES: Dict[str, Dict[str, Any]] = {
    "copasi":    {"wrapper_mod": "pbg_copasi",    "wrapper_dist": "pbg-copasi",
                  "lib_mods": ["basico", "COPASI"], "lib_dist": "python-copasi"},
    "tellurium": {"wrapper_mod": "pbg_tellurium", "wrapper_dist": "pbg-tellurium",
                  "lib_mods": ["tellurium"],       "lib_dist": "tellurium"},
    "simbio":    {"wrapper_mod": "viva_simbio",    "wrapper_dist": "viva-simbio",
                  "lib_mods": ["simbio"],          "lib_dist": "simbio"},
    "amici":     {"wrapper_mod": "viva_amici",     "wrapper_dist": "viva-amici",
                  "lib_mods": ["amici"],           "lib_dist": "amici"},
}


def host_info() -> Dict[str, str]:
    """Global run context — host, platform, Python, and an ISO-8601 UTC stamp."""
    return {
        "host":        socket.gethostname(),
        "platform":    platform.platform(),
        "python":      platform.python_version(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _dist_version(dist: Optional[str]) -> Optional[str]:
    if not dist:
        return None
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        return None


def _module_version(mod_names) -> Optional[str]:
    for name in mod_names:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        v = getattr(mod, "__version__", None)
        if v:
            return str(v)
    return None


def _module_dir(mod_name: str) -> Optional[str]:
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return None
    f = getattr(mod, "__file__", None)
    return os.path.dirname(f) if f else None


def git_info(mod_name: str) -> Optional[Dict[str, Any]]:
    """Short commit + dirty flag for the git repo containing ``mod_name``.

    Returns ``None`` for a non-git (e.g. wheel) install or if git is unavailable.
    """
    cwd = _module_dir(mod_name)
    if not cwd:
        return None
    # A wheel/regular install lives under site-packages; `git rev-parse` there
    # walks up to whatever repo happens to enclose the venv (e.g. the workspace)
    # and reports a misleading commit. Only editable installs (source outside
    # site-packages) carry a meaningful simulator commit.
    if "site-packages" in cwd or os.sep + ".venv" + os.sep in cwd:
        return None
    try:
        commit = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if commit.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "-C", cwd, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return {
            "commit": commit.stdout.strip(),
            "dirty":  bool(dirty.stdout.strip()),
        }
    except Exception:
        return None


def simulator_provenance(sim: str) -> Dict[str, Any]:
    """Library + wrapper version and git commit for one simulator."""
    pkg = _SIM_PACKAGES.get(sim, {})
    wrapper_mod = pkg.get("wrapper_mod")
    return {
        "simulator":       sim,
        "lib_version":     _module_version(pkg.get("lib_mods", []))
                           or _dist_version(pkg.get("lib_dist")),
        "lib_git":         git_info(pkg["lib_mods"][0]) if pkg.get("lib_mods") else None,
        "wrapper":         pkg.get("wrapper_dist"),
        "wrapper_version": _dist_version(pkg.get("wrapper_dist")),
        "wrapper_git":     git_info(wrapper_mod) if wrapper_mod else None,
    }
