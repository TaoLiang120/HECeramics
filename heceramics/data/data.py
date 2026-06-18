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


DATA_PATH = config_vars["DATA_PATH"]
float_format = Constants["float_format"]

Elemental_ROM_KEYs = []
Compound_ROM_KEYs = []
Elemental_DELTA_KEYs = []
Compound_DELTA_KEYs = []
Elemental_DISTORT_KEYs = []
Compound_DISTORT_KEYs = []

class myData:
    def __init__(self, fname, style=1):
        self.fname = fname
        self.df = pd.read_csv(self.fname)
        self.ntot = len(self.df)
        self.elements = Element_negativity
        self.style = style
        self.gooddf = self.df.copy(deep=True)
        self.baddf = pd.DataFrame(columns=self.df.columns.tolist())
        self.update_data()

    def update_data(self):
        self.ngood = len(self.gooddf)
        self.nbad = len(self.baddf)
        self.compstrs = self.gooddf["Composition"].to_numpy()
        if self.style == 0:
            self.compstrs_A = self.gooddf["Composition"].to_numpy()
            self.compstrs_B = self.gooddf["Composition"].to_numpy()
        elif self.style == 1:
            self.compstrs_A = self.gooddf["Composition_A"].to_numpy()
            self.compstrs_B = self.gooddf["Composition_A"].to_numpy()
        elif self.style == 2:
            self.compstrs_A = self.gooddf["Composition_A"].to_numpy()
            self.compstrs_B = self.gooddf["Composition_B"].to_numpy()

    def save_to(self, fname, df=None):
        if df is None: df = self.gooddf.copy()
        df.to_csv(fname, index=False, float_format=float_format)

    def get_all_features(self, df=None):
        if df is None: df = self.gooddf.copy()
        self.gooddf = self.get_elmental_ROMs(df=df)
        self.gooddf = self.get_compound_ROMs(df=df)
        self.gooddf = self.get_elmental_DELTAs(df=df)
        self.gooddf = self.get_compound_DELTAs(df=df)
        self.gooddf = self.get_elmental_DISTORTs(df=df)
        self.gooddf = self.get_compound_DISTORTs(df=df)
        self.gooddf = self.get_constructed_features(df=df)
        return df

    def get_elmental_ROMs(self, keys=Elemental_ROM_KEYs, df=None, DF_REFERENCE=DF_elements, Index_style=0):
        if df is None: df = self.gooddf.copy()
        for ikey in range(len(keys)):
            key = keys[ikey]
            if self.style == 0:
                compstrs = df["Composition"].to_numpy()
            elif self.style == 1:
                compstrs = df["Composition_A"].to_numpy()

            vs = []
            for icomp in range(len(compstrs)):
                compstr = compstrs[icomp]
                thisCU = CompstrUtil(compstr)
                thisv = thisCU.compstr2ROM(key, DF_REFERENCE, Index_style=Index_style)
                vs.append(thisv)
            outkey = key + "_ROM"
            df[outkey] = vs
        return df


    def get_compound_ROMs(self, keys=Compound_ROM_KEYs, df=None, DF_REFERENCE=DF_binaries, Index_style=1):
        if df is None: df = self.gooddf.copy()
        for ikey in range(len(keys)):
            key = keys[ikey]
            if self.style == 0:
                compstrs = df["Composition"].to_numpy()
            elif self.style == 1:
                compstrs = df["Composition_A"].to_numpy()

            vs = []
            for icomp in range(len(compstrs)):
                compstr = compstrs[icomp]
                thisCU = CompstrUtil(compstr)
                thisv = thisCU.compstr2ROM(key, DF_REFERENCE, Index_style=Index_style)
                vs.append(thisv)
            outkey = key + "_ROM"
            df[outkey] = vs
        return df

    def get_elmental_DELTAs(self, keys=Elemental_DELTA_KEYs, df=None, DF_REFERENCE=DF_elements, Index_style=0):
        if df is None: df = self.gooddf.copy()
        for ikey in range(len(keys)):
            key = keys[ikey]
            if self.style == 0:
                compstrs = df["Composition"].to_numpy()
            elif self.style == 1:
                compstrs = df["Composition_A"].to_numpy()
            meankey = key + "_ROM"
            means = df[meankey].to_numpy()
            vs = []
            for icomp in range(len(compstrs)):
                compstr = compstrs[icomp]
                mean = means[icomp]
                thisCU = CompstrUtil(compstr)
                thisv = thisCU.compstr2delta(key, mean, DF_REFERENCE, compstr=compstr, Index_style=Index_style)
                vs.append(thisv)
            outkey = key + "_DELTA"
            df[outkey] = vs
        return df

    def get_compound_DELTAs(self, keys=Compound_DELTA_KEYs, df=None, DF_REFERENCE=DF_binaries, Index_style=1):
        if df is None: df = self.gooddf.copy()
        for ikey in range(len(keys)):
            key = keys[ikey]
            if self.style == 0:
                compstrs = df["Composition"].to_numpy()
            elif self.style == 1:
                compstrs = df["Composition_A"].to_numpy()
            meankey = key + "_ROM"
            means = df[meankey].to_numpy()
            vs = []
            for icomp in range(len(compstrs)):
                compstr = compstrs[icomp]
                mean = means[icomp]
                thisCU = CompstrUtil(compstr)
                thisv = thisCU.compstr2delta(key, mean, DF_REFERENCE, compstr=compstr, Index_style=Index_style)
                vs.append(thisv)
            outkey = key + "_DELTA"
            df[outkey] = vs
        return df

    def get_elmental_DISTORTs(self, keys=Elemental_DISTORT_KEYs, df=None, DF_REFERENCE=DF_elements, Index_style=0):
        if df is None: df = self.gooddf.copy()
        for ikey in range(len(keys)):
            key = keys[ikey]
            if self.style == 0:
                compstrs = df["Composition"].to_numpy()
            elif self.style == 1:
                compstrs = df["Composition_A"].to_numpy()
            vs = []
            vmaxes = []
            for icomp in range(len(compstrs)):
                compstr = compstrs[icomp]
                mean = means[icomp]
                thisCU = CompstrUtil(compstr)
                thisv, thismax = thisCU.compstr2distort(key, DF_REFERENCE, compstr=compstr, Index_style=Index_style)
                vs.append(thisv)
                vmaxes.append(thismax)
            outkey = key + "_DISTORT"
            df[outkey] = vs
            df[outkey + "_MAX"] = vmaxes
        return df

    def get_compound_DISTORTs(self, keys=Compound_DISTORT_KEYs, df=None, DF_REFERENCE=DF_binaries, Index_style=1):
        if df is None: df = self.gooddf.copy()
        for ikey in range(len(keys)):
            key = keys[ikey]
            if self.style == 0:
                compstrs = df["Composition"].to_numpy()
            elif self.style == 1:
                compstrs = df["Composition_A"].to_numpy()
            vs = []
            vmaxes = []
            for icomp in range(len(compstrs)):
                compstr = compstrs[icomp]
                thisCU = CompstrUtil(compstr)
                thisv, thismax = thisCU.compstr2delta(key, DF_REFERENCE, compstr=compstr, Index_style=Index_style)
                vs.append(thisv)
                vmaxes.append(thismax)
            outkey = key + "_DISTORT"
            df[outkey] = vs
            df[outkey + "_MAX"] = vmaxes
        return df

    def get_constructed_features(self, df=None):
        if df is None: df = self.gooddf.copy()
        DFU = DataFrameUtils(df)
        df = DFU.get_constructed_features()
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


