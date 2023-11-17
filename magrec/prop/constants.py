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

DEFAULT_UNITS = {
    "length": "mm",
    "layer_thickness": "mm",
    "height": "mm",
    "current": "A",
    "magnetic_field": "mT",
}

def get_exponent_from_unit(unit: str) -> int:
    """Returns the exponent of the unit prefix.

    Args:
        unit (str): Unit with a prefix, e.g. "mm" for millimeter, "k" for kilometer, etc.

    Returns:
        int: Exponent of the unit prefix.
    """
    # Assume so far that all units are of length 1. If not, this will need to be changed.
    if len(unit) == 1:
        return 0
    
    unit_prefix = unit[0]
    
    if unit_prefix == "f":                          # femto
        return -15
    elif unit_prefix == "p":                        # pico
        return -12
    elif unit_prefix == "n":                        # nano
        return -9
    elif unit_prefix == "u" or unit_prefix == "μ":  # micro
        return -6
    elif unit_prefix == "m":                        # milli
        return -3
    elif unit_prefix == "k":                        # kilo
        return 3
    elif unit_prefix == "M":                        # mega
        return 6
    elif unit_prefix == "G":                        # giga
        return 9
    elif unit_prefix == "T":                        # tera
        return 12
    else:
        raise ValueError(f"Unknown unit prefix: {unit_prefix}")