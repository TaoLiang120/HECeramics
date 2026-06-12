import os

import numpy as np
import pandas as pd

from heceramics.data.dataframe_util import generate_list_props, list_props2df
from heceramics.myelements.myelements import DF_elements, DF_binaries
from heceramics.myglobal import HECs, SQSs, RCUT, Vol_scale, Element_negativity
from heceramics.myglobal import config_vars, Constants

CONTCAR_PATH = config_vars["CONTCAR_PATH"]
DATA_PATH = config_vars["DATA_PATH"]
float_format = Constants["float_format"]

HECs = ["HEC3", "HEC4", "HEC6", "HEC9", "HEC10", "HEC16",
        "HEC1000", "HEC2001", "HEC2002", "HEC2003", "HEC3001", "HEC4001"]
Weights = [1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 5]


for ihec in range(len(HECs)):
    hec = HECs[ihec]
    thisfname = os.path.join(DATA_PATH, hec + ".csv")
    thisdf = pd.read_csv(thisfname)
    weight = Weights[ihec]
    for iw in range(weight):
        if ihec == 0 and iw == 0:
            df = thisdf.copy()
        else:
            df = pd.concat([df, thisdf])

fname = "Weighted_ALL_HECs.csv"
fname = os.path.join(DATA_PATH, fname)
df.to_csv(fname, index=False, float_format=float_format)
print("==== finished all ====")
