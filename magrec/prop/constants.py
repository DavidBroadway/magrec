# Defines global physical constants to use throughout the code everywhere!
#
# Units of the constant match those of magpylib choice.
#
# To quote from Magpylib:
# https://magpylib.readthedocs.io/en/latest/_pages/page_07_physics_computation.html?highlight=units#units-and-scaling-property
#
# | Magpylib uses the following physical units:
# |     [mT]: for the B-field and the magnetization (µ0*M).
# |     [kA/m]: for the H-field.
# |     [mm]: for position and length inputs.
# |     [deg]: for angle inputs by default.
# |     [A]: for current inputs.


twopi = 6.2831853071795862

MU0: float = 1.25663706212  # [mT * mm / A]
"""Magnetic permeability of free space, 4π * 1e-7 in [H/m = T * m / A]."""

EPSILON0 = 8.854187817e-12

    
def unit_to_exponent(unit: str):
    """
    Convert a unit prefix to an exponent.
    """
    if len(unit) == 1:
        return 1
    
    prefix = unit[0]
    
    match prefix:
        case 'n':
            return 1e-9
        case 'u':
            return 1e-6
        case 'μ':
            return 1e-6
        case 'm':
            return 1e-3
        case 'k':
            return 1e3
        case 'M':
            return 1e6
        case 'G':
            return 1e9