import os

import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

from heceramics.myglobal import Element_negativity
from heceramics.myglobal import config_vars, Constants
from heceramics.utils.utils import MyUtils
from heceramics.myelements.myelements import get_DF_Index
from heceramics.myelements.myelements import DF_elements, DF_binaries
from heceramics.data.compstr_util import CompstrUtil
from heceramics.data.dataframe_util import DataFrameUtils


CONTCAR_PATH = config_vars["CONTCAR_PATH"]
DATA_PATH = config_vars["DATA_PATH"]
float_format = Constants["float_format"]


class myData:
    def __init__(self, fname):
        self.fname = fname

        self.df = pd.read_csv(self.fname)
        self.ntot = len(self.df)
        self.elements = Element_negativity

        self.gooddf = self.df.copy(deep=True)
        self.baddf = pd.DataFrame(columns=self.df.columns.tolist())
        self.ngood = len(self.gooddf)
        self.compstrs = self.gooddf["Composition"].to_numpy()

    def save_to(self, fname, df=None):
        if df is None: df = self.df.copy()
        df.to_csv(fname, index=False, float_format=float_format)

    def normalization(self, df, keys=None, savefile=False, outfile=None, Add_norm=False):
        columns = df.columns.tolist()
        if keys is None:
            thiskeys = columns[0:len(columns)]
        else:
            if isinstance(keys, str): keys = [keys]
            thiskeys = []
            for key in keys:
                if key in df.columns:
                    thiskeys.append(key)

        for ikey in range(len(thiskeys)):
            thiskey = thiskeys[ikey]
            thisnorm = True
            thisvals = df[thiskey].to_numpy()
            vmin = np.min(thisvals)
            vmax = np.max(thisvals)
            if vmax == vmin: thisnorm = False
            if thisnorm:
                normvals = MyUtils.value_normalization(thisvals, vmin, vmax)
            else:
                normvals = np.ones(len(df))
            if Add_norm: df[thiskey + "_norm"] = normvals

        if savefile:
            if outfile is None:
                outfile = self.fname
                if "_normed" not in outfile:
                    outfile = outfile.replace(".csv", "")
                    outfile = outfile + "_normed.csv"
            self.save_to(outfile, df=df)
        return df

    ############################################################
    def select_by_CS(self, df, compspace, condition, ncompon=None, style=0, set2gooddf=False):
        inds = np.arange(len(df), dtype=int)
        compstrs = df['Composition'].to_numpy()
        bads = []
        for icomp in range(len(df)):
            compstr = compstrs[icomp]
            comp = Composition(compstr)
            thissyms = []
            for isym in range(len(comp.elements)):
                thissyms.append(comp.elements[isym].symbol)
            if condition[0:3].upper() == "ALL":
                isValid = True
            else:
                if isinstance(ncompon, int):
                    if len(comp.elements) == ncompon:
                        isValid = True
                    else:
                        isValid = False
                else:
                    isValid = True
                if condition[0:3].upper() == "EXA":
                    if len(comp.elements) != len(compspace):
                        isValid = False

                if isValid:
                    if style == 0:
                        for sym in compspace:
                            if condition[0:3].upper() == "EXC":
                                if sym in thissyms:
                                    isValid = False
                            else:
                                if sym not in thissyms:
                                    isValid = False
                        if not isValid:
                            bads.append(icomp)
                    else:
                        for sym in thissyms:
                            if condition[0:3].upper() == "EXC":
                                if sym in compspace:
                                    isValid = False
                            else:
                                if sym not in compspace:
                                    isValid = False
                        if not isValid:
                            bads.append(icomp)
            if icomp % 10000 == 0: print(f"finished {icomp} structures - select_by_CS!")

        bads = np.array(bads, dtype=int)
        goods = np.delete(inds, bads)
        if set2gooddf:
            self.baddf = df.iloc[bads]
            self.gooddf = df.iloc[goods]
            self.ngood = len(self.gooddf)
            self.compstrs = self.gooddf["Composition"].to_numpy()
        return df.iloc[goods], df.iloc[bads]

    def select_by_ncompons(self, df, ncompons, set2gooddf=False):
        if isinstance(ncompons, int):
            ncompons = [ncompons]
        else:
            ncompons = list(ncompons)
        inds = np.arange(len(df), dtype=int)
        compstrs = df['Composition'].to_numpy()
        bads = []
        for icomp in range(len(df)):
            compstr = compstrs[icomp]
            comp = Composition(compstr)
            thisncompon = len(comp.elements)
            if thisncompon in ncompons:
                pass
            else:
                bads.append(icomp)
        bads = np.array(bads, dtype=int)
        goods = np.delete(inds, bads)
        if set2gooddf:
            self.baddf = df.iloc[bads]
            self.gooddf = df.iloc[goods]
            self.ngood = len(self.gooddf)
            self.compstrs = self.gooddf["Composition"].to_numpy()
        return df.iloc[goods], df.iloc[bads]

    def select_by_DF(self, df, key, condition, set2gooddf=False):
        if key in df.columns:
            values = df[key].to_numpy()
            inds = np.arange(len(df), dtype=int)
            goods = np.compress(eval(condition), inds)
            bads = np.delete(inds, goods)
            if set2gooddf:
                self.baddf = df.iloc[bads]
                self.gooddf = df.iloc[goods]
                self.ngood = len(self.gooddf)
                self.compstrs = self.gooddf["Composition"].to_numpy()
            return df.iloc[goods], df.iloc[bads]
        else:
            raise ValueError("KeyError: key is not found")

    def validate_dataframe(self, df=None, set2data=False):
        if df is None: df = self.df.copy(deep=True)
        if "vacancy_formation" in df.columns:
            key = "vacancy_formation"
            condition = "values<2.0"
            gooddf, baddf1 = self.select_by_DF(df, key, condition, set2gooddf=set2data)

            key = "vacancy_formation"
            condition = "values>-2.0"
            gooddf, baddf2 = self.select_by_DF(gooddf, key, condition, set2gooddf=set2data)

            if set2data:
                self.baddf = pd.concat([baddf1, baddf2])
                self.gooddf = gooddf.copy()
                self.ngood = len(self.gooddf)
                self.compstrs = self.gooddf["Composition"].to_numpy()
            return gooddf, pd.concat([baddf1, baddf2])
        else:
            return self.gooddf, self.baddf


