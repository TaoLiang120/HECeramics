import pandas as pd
from pymatgen.core.periodic_table import Element
rename_cols = {"CompositionA": "Composition", "vec": "VEC"}
fname = "element_data.csv"
df = pd.read_csv(fname)
df = df.sort_values(by="electronegativity")
df = df.rename(columns=rename_cols)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
df.to_csv("element_data_sorted.csv", index=False)
print(df["Composition"].tolist())
compstrs = df["Composition"].tolist()

Element_list = ['Ce', 'Y', 'Pu', 'Th', 'Hf', 'Zr', 'Sc', 'Ta', 'Pa', 'Ti', 'Mn', 'Nb', 'Al', 'Tl', 'V', 'Zn', 'Cr', 'Cd', 'Fe', 'Co', 'Re', 'Cu', 'Tc', 'Ni', 'Mo', 'Pd', 'Os', 'Ir', 'Ru', 'Rh', 'W', 'C']

columns = df.columns.tolist()
for isym in range(len(Element_list)):
    sym = Element_list[isym]
    if sym not in compstrs:
        el = Element(sym)
        print(f"adding {sym} to element_data.csv")
        print(el.data)
        thisdict = {}

        for col in columns:
            thisdict[col] = 0.0
            if col == "Composition":
                thisdict[col] = el.symbol
            elif col == "atomic_mass":
                thisdict[col] = el.atomic_mass
            elif col == "electronegativity":
                thisdict[col] = el.X
            elif col == "atomic_radius":
                thisdict[col] = el.atomic_radius
            elif col == "metallic_radius":
                thisdict[col] = el.metallic_radius
            elif col == "melting_point":
                thisdict[col] = el.melting_point
            elif col == "thermal_expansion":
                thisdict[col] = el.data['Coefficient of linear thermal expansion']
            elif col == "thermal_conductivity":
                thisdict[col] = el.data['Thermal conductivity']
            elif col == "vickers_hardness":
                thisdict[col] = el.vickers_hardness
            elif col == "bulk_modulus":
                thisdict[col] = el.bulk_modulus
            elif col == "youngs_modulus":
                thisdict[col] = el.youngs_modulus
            elif col == "poissons_ratio":
                thisdict[col] = el.poissons_ratio
            elif col == "VEC":
                if sym == "Tl":
                    thisdict[col] = 3
                elif sym == "Cd":
                    thisdict[col] = 2
                else:
                    thisdict[col] = 0
        print(thisdict)
        df[len(df)] = thisdict

df = df.sort_values(by="electronegativity")
df.to_csv("element.csv", index=False)
print(df["Composition"].tolist())
