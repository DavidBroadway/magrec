"""
Propagators and Pipeline for magnetic field reconstruction.
"""

# Core Pipeline classes (lightweight imports)
from magrec.prop.Pipeline import (
    Pipeline,
    Step,
    ValueStep,
    DatasetStep,
    Show,
    Dipoles,
    DipoleLocator,
    Propagator,
    Projection,
    Put,
    Function,
    Optimizable,
    Optimize,
)

# Lazy imports for heavy dependencies
def _lazy_import_propagators():
    """Import propagators only when needed."""
    from magrec.prop.Propagator import (
        MagneticDipolePropagator,
        CurrentDipolePropagator,
        AxisProjectionPropagator,
    )
    return MagneticDipolePropagator, CurrentDipolePropagator, AxisProjectionPropagator

# Make them available but don't import yet
__all__ = [
    'Pipeline',
    'Step',
    'ValueStep',
    'DatasetStep',
    'Show',
    'Dipoles',
    'DipoleLocator',
    'Propagator',
    'Projection',
    'Put',
    'Function',
    'Optimizable',
    'Optimize',
]

