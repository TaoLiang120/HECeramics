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

list_props_all = []
for hec in HECs:
    list_props = []
    fname4calc = hec + "_Vacancy_Formation_original.xlsx"
    df_calc = pd.read_excel(os.path.join(CONTCAR_PATH, fname4calc))
    cid_start = 0
    for iSQS in range(len(SQSs)):
        sqs = SQSs[iSQS]
        fin = os.path.join(CONTCAR_PATH, hec, sqs, "CONTCAR")
        list_props, cid_start = generate_list_props(list_props, fin,
                                                    hec=hec, sqs=sqs, sort=False,
                                                    rcut=RCUT, cid_start=cid_start, df_calc=df_calc,
                                                    scale=Vol_scale, elements=Element_negativity)
        print(f"finished HEC:{hec} sqs:{sqs}")
        print("------")
    list_props_all += list_props
    fname = hec + ".csv"
    list_props2df(list_props, fout=os.path.join(DATA_PATH, fname))
    print(f"finished HEC:{hec}")
    print("========")

list_props_all = np.array(list_props_all)
fname = "ALL_HECs.csv"
list_props2df(list_props_all, fout=os.path.join(DATA_PATH, fname))
df = pd.read_csv(os.path.join(DATA_PATH, fname))
print(df.columns.tolist())
print("==== finished all ====")
