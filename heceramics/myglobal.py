import os, sys
from monty.serialization import loadfn

import numpy as np
import json
from itertools import combinations
from pymatgen.core.composition import Composition

try:
    config_vars = loadfn(os.path.join(os.path.expanduser('~'), 'heceramics.yaml'))
except:
    sys.exit('No heceramics.yaml file was found. Please configure the '
             ' heceramics.yaml and put it in your home directory.')

VERY_SMALL_VALUE = 1.0e-40
significant_figure4composition = 4
significant_figure = 6

Constants = {"kb": 8.6173324E-5, "bohr2angstrom": 0.529177249, "density2gcm": 1.6605402,
             "eVA2GPa": 160.2177, "GPa2eVA": 0.00624150648, "KJ2eVA": 1.0/96.4915666370759,
             "J2eVA": 1.0/96491.5666370759,  "TC2CPA_volume": 1660539.0671738465, "float_format": "%.8f"}

Element_list = ['Ce', 'Y', 'Pu', 'Th', 'Hf', 'Zr', 'Sc', 'Ta', 'Pa', 'Ti', 'Mn', 'Nb', 'Al', 'V',
                'Zn', 'Cr', 'Fe', 'Co', 'Re', 'Cu', 'Tc', 'Ni', 'Mo', 'Pd', 'Os', 'Ir', 'Ru', 'Rh', 'W', 'C']
Element_negativity = ['Ce', 'Y', 'Pu', 'Th', 'Hf', 'Zr', 'Sc', 'Ta', 'Pa', 'Ti', 'Mn', 'Nb', 'Al', 'V',
                      'Zn', 'Cr', 'Fe', 'Co', 'Re', 'Cu', 'Tc', 'Ni', 'Mo', 'Pd', 'Os', 'Ir', 'Ru', 'Rh', 'W', 'C']

nele_default = len(Element_negativity)

KeyColumns = ["Composition", "Pretty_Formula"]





