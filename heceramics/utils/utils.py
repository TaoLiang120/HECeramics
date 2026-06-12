import os
import copy
import numpy as np
import pandas as pd
import itertools
import json

class MyUtils:
    @staticmethod
    def value_normalization(thisvals, vmin, vmax):
        v = (thisvals - vmin) / (vmax - vmin)
        v = np.select([v < 0.0, v < 1.0, v >= 1.0], [0.0, v, 1.0])
        return v

    @staticmethod
    def cap_value(v, cap=1, style=0):
        if style == 1:
            if v > cap:
                v = cap
        else:
            if v > cap:
                v = cap
            elif v < -cap:
                v = -cap
        return v

    @staticmethod
    def load_complist_from_json(fname):
        with open(fname, "r") as f:
            s = f.read()
        complist = json.loads(s)
        return complist

    @staticmethod
    def save_complist_to_json(fname, complist):
        with open(fname, "w") as f:
            json.dump(complist, f)