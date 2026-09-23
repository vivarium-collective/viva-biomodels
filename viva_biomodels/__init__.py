"""viva_biomodels — workspace Python package.

Importing the package fires the ``@composite_generator`` decorators inside
``composites/`` so ``discover_generators()`` finds them without callers
having to import each generator module explicitly.

Also provides the shared bigraph-schema type dictionaries absorbed from
pbg-biomodels-bundle (register_types, TYPES_DICT) and re-exports the
public Steps so callers can do ``from viva_biomodels import SimulatorComparisonStep``.
"""
# Importing composites fires the @composite_generator decorators. Guarded so the
# package still imports in a deps-light environment (e.g. the read-only publish
# job installs `-e . --no-deps`, without the simulator backends): composites then
# simply aren't registered here (the workbench renders them from spec), but
# lightweight submodules like `workbench_viewers` — which the Analyses-tab
# discovery imports — remain importable.
try:
    from viva_biomodels import composites  # noqa: F401
except Exception:  # pragma: no cover - depends on optional simulator backends
    composites = None  # type: ignore

# ---------------------------------------------------------------------------
# Shared bigraph-schema type registrations (absorbed from pbg-biomodels-bundle)
# ---------------------------------------------------------------------------

sed_types = {
    'result': {
        'time': 'list[float]',
        'species_concentrations': 'map[list[float]]',
    },
    'results': 'map[result]'
}

standard_types = {
    'numeric_result': {
        'time': 'list[float]',
        'columns': 'list[string]',
        'values': 'list[list[float]]',
    },
    'numeric_results': 'map[numeric_result]',
    'columns_of_interest': 'list[string]'
}

TYPES_DICT = {
    **standard_types,
    **sed_types
}


from viva_biomodels.types import register_simulation_types  # noqa: E402


def register_types(core):
    """Register pbg-biomodels bigraph-schema types into a ProcessBigraph core."""
    core.register_types(TYPES_DICT)
    register_simulation_types(core)
    return core


# Re-export the public Step so callers can do
# `from viva_biomodels import SimulatorComparisonStep`.
# Imported here (not at the top) so that `register_types` keeps working even
# when downstream dependencies of the steps subpackage are missing. Guarded for
# the same deps-light reason as the composites import above.
try:
    from viva_biomodels.steps import SimulatorComparisonStep  # noqa: E402
except Exception:  # pragma: no cover - depends on optional simulator backends
    SimulatorComparisonStep = None  # type: ignore

__all__ = ["TYPES_DICT", "register_types", "SimulatorComparisonStep"]
