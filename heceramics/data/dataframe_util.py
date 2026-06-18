import os
import copy
import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition
from heceramics.myglobal import VERY_SMALL_VALUE, Element_list, Element_negativity
from heceramics.myglobal import config_vars, Constants
from heceramics.myelements.myelements import get_DF_Index
from heceramics.data.compstr_util import CompstrUtil


float_format = Constants["float_format"]

class DataFrameUtils:
    def __init__(self, df):
        self.df = df

    def get_pugh_ratio(self):
        if 'Bulk_modulus_ROM' in df_columns:
            Bs = df['Bulk_modulus_ROM'].to_numpy()
        else:
            Bs = df['Bulk_modulus'].to_numpy()
        if 'Shear_modulus_ROM' in df_columns:
            Ss = df['Shear_modulus_ROM'].to_numpy()
        else:
            Ss = df['Shear_modulus'].to_numpy()
        df['Pugh_ratio'] = Bs/Ss
        return df


