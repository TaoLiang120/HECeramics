import os
import numpy as np
import pandas as pd

from heceramics.myglobal import config_vars, Constants
from heceramics.myglobal import VERY_SMALL_VALUE, significant_figure
from heceramics.myglobal import Element_negativity, nele_default, KeyColumns
from heceramics.data.data import myData, DATA_PATH


Element_negativity = ['Ce', 'Y', 'Pu', 'Th', 'Hf', 'Zr', 'Sc', 'Ta', 'Pa', 'Ti', 'Mn', 'Nb', 'Al', 'V',
                      'Zn', 'Cr', 'Fe', 'Co', 'Re', 'Cu', 'Tc', 'Ni', 'Mo', 'Pd', 'Os', 'Ir', 'Ru', 'Rh', 'W', 'C']

rename_cols = {"CompositionA": "Composition_A", "vec": "VEC"}


fname = "aflow.csv"
df_aflow = pd.read_csv(fname)
df_aflow = df_aflow.rename(columns=rename_cols)

fname = "icsd.csv"
df_icsd = pd.read_csv(fname)
df_icsd = df_icsd.rename(columns=rename_cols)


def remove_unwanted_elements(df):
    df = df.sort_values(by="Composition_A")
    acompstrs = df["Composition_A"].to_numpy()
    goodinds = []
    for i in range(len(acompstrs)):
        compstr = acompstrs[i]
        if compstr in Element_negativity:
            goodinds.append(i)
    return df.iloc[goodinds]


def get_unique_entry(df):
    compstrs = df["Composition_A"].tolist()
    cols = df.columns.tolist()

    unique_compstrs, counts = np.unique(compstrs, return_counts=True)
    sumcounts = np.cumsum(counts)
    sumcounts = np.append([0], sumcounts)
    df_list = []
    for i in range(len(unique_compstrs)):
        thisdf = df.iloc[sumcounts[i]:sumcounts[i+1]]
        df_list.append(thisdf)
        print(f"Total {i+1} out of {len(unique_compstrs)}")

    for i in range(len(df_list)):
        thisdf = df_list[i]
        thisdict = {}
        for col in cols:
            if col == "Composition":
                thisdict[col] = thisdf.iloc[0][col]
            elif col == "Composition_A":
                thisdict[col] = thisdf.iloc[0][col]
            else:
                thisv = 0
                nvalid = 0
                for idf in range(len(thisdf)):
                    v = thisdf.iloc[idf][col]
                    if str(v) == "nan":
                        pass
                    else:
                        thisv += v
                        nvalid += 1
                if nvalid > 0:
                    thisdict[col] = thisv/nvalid
                else:
                    thisdict[col] = "nan"
        new_unique_df = pd.DataFrame(columns=cols)
        new_unique_df.loc[len(new_unique_df)] = thisdict
        if i == 0:
            df_unique = new_unique_df.copy(deep=True)
        else:
            df_unique = pd.concat([df_unique, new_unique_df])

        if i == 0:
            print(thisdf)
            print(thisdict)
            print(new_unique_df)
            print(f"==== 00 ====")
        print(f"Total {i+1} out of {len(df_list)}")
    return df_unique

def get_merged_df(df1, df2):
    cols = df1.columns.tolist()
    compstrs1 = df1["Composition_A"].tolist()
    compstrs2 = df2["Composition_A"].tolist()
    df_merged = df1.copy(deep=True)
    for i in range(len(compstrs2)):
        compstr2 = compstrs2[i]
        if compstr2 not in compstrs1:
            thisdict = {}
            for col in cols:
                thisdict[col] = df2.iloc[i][col]
            df_merged.loc[len(df_merged)] = thisdict
        else:
            ind1 = compstrs1.index(compstr2)
            thisdict = {}
            for col in cols:
                thisv1 = df1.iloc[ind1][col]
                thisv2 = df2.iloc[i][col]
                if col == "Composition_A" or col == "Composition":
                    thisdict[col] = thisv1
                else:
                    if str(thisv1) == "nan" and str(thisv2) != "nan":
                        thisdict[col] = thisv2
                    elif str(thisv1) != "nan" and str(thisv2) != "nan":
                        thisdict[col] = thisv2
                        thisdiff = thisv2 - thisv1
                        if thisdiff != 0:
                            print(f"Warning: {compstr2} {col} {thisv1} {thisv2}")
                    else:
                        thisdict[col] = thisv1
            df_merged.loc[ind1] = thisdict
    return df_merged





'''
fnames = ["aflow.csv", "icsd.csv"]
outfnames = ["aflow_clean.csv", "icsd_clean.csv"]
for i in range(len(fnames)):
    fname = fnames[i]
    outfname = outfnames[i]
    df = pd.read_csv(fname)
    df = df.rename(columns=rename_cols)
    df = remove_unwanted_elements(df)
    df.to_csv(outfname, index=False)
    print(f"Finished {fname}")
'''

'''
fnames = ["aflow_clean.csv", "icsd_clean.csv"]
outfnames = ["aflow_unique.csv", "icsd_unique.csv"]
for i in range(0, len(fnames)):
#for i in range(0, 1):
    fname = fnames[i]
    outfname = outfnames[i]
    df = pd.read_csv(fname)
    df_unique = get_unique_entry(df)
    df_unique.to_csv(outfname, index=False)
    print(f"Finished {outfname}")
'''

'''
fname1 = "aflow_unique.csv"
fname2 = "icsd_unique.csv"
outfname = "merged_rock_salt.csv"
df1 = pd.read_csv(fname1)
df2 = pd.read_csv(fname2)
df_merged = get_merged_df(df1, df2)
df_merged.to_csv(outfname, index=False)
'''

outfname = "merged_rock_salt.csv"
df = pd.read_csv(outfname)
Acompstrs = df["Composition_A"].tolist()
compstrs = []
for i in range(len(Acompstrs)):
    acompstr = Acompstrs[i]
    compstrs.append(acompstr + 'C')
df['Composition'] = compstrs
df.to_csv(outfname, index=False)
