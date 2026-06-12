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


