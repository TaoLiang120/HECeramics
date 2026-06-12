import os
import numpy as np
import pandas as pd
from pymatgen.core.periodic_table import Element

Element_file = "../myelements/element.csv"
Element_negativity = ['Y', 'Hf', 'Zr', 'Sc', 'Ta', 'Ti', 'Nb', 'V', 'Mo', 'W', 'C']
VECs = [3, 4, 4, 3, 5, 4, 5, 5, 6, 6, 4]
cols = ["Composition", "amass", "E_negativity", "VEC", "atomic_radius", "metallic_radius",  "enthalpy"]
zeros = np.zeros([len(Element_negativity), len(cols)])
df = pd.DataFrame(zeros, columns=cols)
df["Composition"] = Element_negativity
df["VEC"] = VECs
ars = []
mrs = []
amasses = []
enegs = []
for i in range(len(Element_negativity)):
    el = Element(Element_negativity[i])
    d = el.data
    ar = d.get("Atomic radius")
    mr = d.get("Metallic radius")
    if mr == "no data":
        mr = ar
    amass = d.get("Atomic mass")
    eneg = el.X
    ars.append(ar)
    mrs.append(mr)
    enegs.append(eneg)
    amasses.append(amass)
df["amass"] = amasses
df["E_negativity"] = enegs
df["atomic_radius"] = ars
df["metallic_radius"] = mrs
df.to_csv("element.csv", index=False)
print(df)


