"""SED-ML report-variable extraction + engine-agnostic leaf completion.

The BioSimulators SED-ML reference reports every dataGenerator variable the
SED-ML declares — for the auto-generated BioModels SED-ML that is *all* the
model's species, global parameters, compartments, and reaction fluxes. The live
simulator wrappers, by contrast, emit only the state (floating species). That
coverage gap — not numerical divergence — is the dominant reason a live engine
fails to "repeat-match" its own reference result (most reference observables
have no live counterpart to compare against).

This module closes the gap for the **constant** part of that set — global
parameters and compartments whose value does not change over the course of the
simulation (no assignment/rate rule, no initial assignment). Every engine reads
the same SBML, so their value is unambiguous and identical across engines: we
fill each requested constant as a flat series on the run's time grid. This is
exactly the class of "values that don't change over the simulation" the
BioSimulators PySCeS wrapper had to special-case
(Biosimulators_PySCeS commit 91b3afc0).

Species come from the engine (already emitted). Time-varying reaction fluxes and
assignment/rate-rule quantities are **not** filled here — they need the engine's
own output (or a kinetic-law evaluation) and are handled separately.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import libsbml

from viva_biomodels.result_leaf import TIME_KEY

# `[@id='X']` or `[@id="X"]` (also tolerates @name=) inside a SED-ML XPath target.
_XPATH_ID = re.compile(r"@(?:id|name)=['\"]([^'\"]+)['\"]")


def sbml_id_from_target(target: Optional[str]) -> Optional[str]:
    """Pull the SBML element id out of a SED-ML variable ``target`` XPath.

    Returns the last `[@id='…']` match (the XPath's leaf element), or ``None``
    for a symbol-only variable (e.g. ``urn:sedml:symbol:time``).
    """
    if not target:
        return None
    hits = _XPATH_ID.findall(target)
    return hits[-1] if hits else None


def report_variable_ids(sed_doc: Any) -> List[str]:
    """SBML ids referenced by every dataGenerator variable in the SED-ML.

    De-duplicated, order-preserving. Symbol-only variables (time) contribute no
    id and are skipped — the reserved time row is handled by the reader.
    """
    seen: set = set()
    ids: List[str] = []
    for i in range(int(sed_doc.getNumDataGenerators())):
        dg = sed_doc.getDataGenerator(i)
        for j in range(dg.getNumVariables()):
            var = dg.getVariable(j)
            sid = sbml_id_from_target(var.getTarget())
            if sid and sid not in seen:
                seen.add(sid)
                ids.append(sid)
    return ids


def _has_rule_or_assignment(model: Any, sid: str) -> bool:
    """True when ``sid`` is driven by an assignment/rate rule or initialAssignment.

    Such a quantity is not a plain constant — its reported value can differ from
    the declared attribute value (initialAssignment) or vary over time
    (assignment/rate rule) — so we must NOT fill it as a flat constant here.
    """
    rule = model.getRule(sid) if hasattr(model, "getRule") else None
    if rule is not None:
        return True
    ia = model.getInitialAssignment(sid) if hasattr(model, "getInitialAssignment") else None
    return ia is not None


def constant_values(model: Any, wanted_ids: List[str]) -> Dict[str, float]:
    """`{sbml_id: value}` for the requested ids that are genuine constants.

    Covers global parameters and compartments that (a) are among ``wanted_ids``,
    (b) carry a set value/size, and (c) are not driven by a rule or initial
    assignment. Species and reactions are intentionally excluded (species are
    emitted by the engine; reaction fluxes vary over time).
    """
    wanted = set(wanted_ids)
    out: Dict[str, float] = {}
    for p in model.getListOfParameters():
        sid = p.getId()
        if sid in wanted and p.isSetValue() and not _has_rule_or_assignment(model, sid):
            out[sid] = float(p.getValue())
    for c in model.getListOfCompartments():
        sid = c.getId()
        if sid in wanted and c.isSetSize() and not _has_rule_or_assignment(model, sid):
            out[sid] = float(c.getSize())
    return out


def constants_for_sbml(sbml_path: str, sed_doc: Any) -> Dict[str, float]:
    """Convenience: constant report values for a model given its SBML + SED-ML."""
    sd = libsbml.readSBML(str(sbml_path))
    model = sd.getModel()
    if model is None:
        return {}
    return constant_values(model, report_variable_ids(sed_doc))


# --- variable role classification -------------------------------------------
# A reported variable is one of four semantic roles. Species and reaction fluxes
# are the *dynamical* output of a simulation; parameters and compartments are
# the *constant* observables that the SED-ML report also names and that
# ``constant_values`` fills (report_variables PR #35, "full observable
# coverage"). Labelling each trace with its role lets a downstream analysis keep
# the full coverage while setting the constants aside, so flat parameter /
# compartment traces no longer dilute cross-engine agreement statistics.
ROLE_SPECIES = "species"
ROLE_FLUX = "flux"
ROLE_PARAMETER = "parameter"
ROLE_COMPARTMENT = "compartment"
#: The two constant-by-construction roles — flat over a UTC run by definition.
CONSTANT_ROLES = frozenset({ROLE_PARAMETER, ROLE_COMPARTMENT})


def classify_report_variables(model: Any, var_ids: List[str]) -> Dict[str, str]:
    """Map each reported variable to its SBML role.

    ``{var_id: "species"|"flux"|"parameter"|"compartment"}``. Classification is
    by SBML id (and, for robustness against engines that emit a display *name*
    — e.g. COPASI — also by name). Anything that is not a declared species,
    global parameter or compartment is a reaction output and is labelled
    ``flux`` (reaction ids are the engines' time-varying flux/rate columns).

    SBML ids are unique across the document, so species/parameter/compartment
    never collide; the name fallback is applied only when an id does not match.
    """
    def _ids_and_names(getter: Any) -> tuple:
        ids, names = set(), set()
        for e in getter():
            ids.add(e.getId())
            if e.isSetName() and e.getName():
                names.add(e.getName())
        return ids, names

    sp_ids, sp_names = _ids_and_names(model.getListOfSpecies)
    pa_ids, pa_names = _ids_and_names(model.getListOfParameters)
    co_ids, co_names = _ids_and_names(model.getListOfCompartments)

    out: Dict[str, str] = {}
    for v in var_ids:
        if v in co_ids or v in co_names:
            out[v] = ROLE_COMPARTMENT
        elif v in pa_ids or v in pa_names:
            out[v] = ROLE_PARAMETER
        elif v in sp_ids or v in sp_names:
            out[v] = ROLE_SPECIES
        else:
            out[v] = ROLE_FLUX
    return out


def variable_roles_for_sbml(sbml_path: str, var_ids: List[str]) -> Dict[str, str]:
    """Convenience: role of each reported variable given a model's SBML path.

    Returns ``{}`` when the SBML has no model. Ids that cannot be resolved still
    receive a role (``flux``) so every variable is labelled.
    """
    sd = libsbml.readSBML(str(sbml_path))
    model = sd.getModel()
    if model is None:
        return {}
    return classify_report_variables(model, list(var_ids))


def complete_utc_leaf(leaf: Dict[str, List[float]], constants: Dict[str, float]) -> Dict[str, List[float]]:
    """Add missing constant observables to a UTC results leaf, in place.

    For each `{id: value}` not already present in ``leaf``, append a flat series
    of ``value`` matched to the leaf's time-grid length. A leaf with no time key
    (or empty) is returned unchanged. Existing keys (species the engine already
    emitted) are never overwritten.
    """
    if not constants or not leaf:
        return leaf
    n = len(leaf.get(TIME_KEY) or [])
    if n == 0:
        return leaf
    for sid, value in constants.items():
        if sid not in leaf:
            leaf[sid] = [float(value)] * n
    return leaf
