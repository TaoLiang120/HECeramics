import os
import numpy as np
import pandas as pd
from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

Element_file = "../myelements/element.csv"
Element_negativity = ['Y', 'Hf', 'Zr', 'Sc', 'Ta', 'Ti', 'Nb', 'V', 'Mo', 'W', 'C']
DF_elements = pd.read_csv(Element_file)

Binary_file = "../myelements/rocksalt_binary.csv"
cols = ["Composition", "lattice_parameter", "enthalpy", "formation_energy"]
cs = ['YC', 'HfC', 'ZrC', 'ScC', 'TaC', 'TiC', 'NbC', 'VC', 'MoC', 'WC']
zeros = np.zeros([len(cs), len(cols)])
df = pd.DataFrame(zeros, columns=cols)
df["Composition"] = cs
latts = [5.068994241, 4.64817261, 4.710810015, 4.682212543, 4.478164867,
         4.334478416, 4.506849102, 4.160500543, 4.366050792, 4.386753926]
efs = [0.035224924, -0.857638218, -0.947242779, -0.135349084, -0.583209891,
       -0.81274695, -0.464024883, -0.415270879, 0.16214403, 0.309852213]

df["lattice_parameter"] = latts
df["formation_energy"] = efs
df.to_csv(Binary_file, index=False)
print(df)



