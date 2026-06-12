import os
import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

from heceramics.myglobal import Element_negativity, RCUT
from heceramics.myglobal import config_vars, Constants
from heceramics.data.data_generator import HECMaker, DataMaker
from heceramics.data.data import myData
from heceramics.models.datafill import DataFill

CONTCAR_PATH = config_vars["CONTCAR_PATH"]
DATA_PATH = config_vars["DATA_PATH"]
float_format = Constants["float_format"]
cols_headers = ["HEC", "Composition", "formation_energy", "vacancy_formation"]
cols_global = ["global_H", "global_a", "global_X", "global_VEC", "global_radius", "global_bond"]
cols_important = ["local_Composition",
                   "nn_mean_H", "nn_min_rdiff_a", "nn_mean_X", "nn_mismatch_X", "nn_max_rdiff_a",
                   "vacancy_formation_M1"]

showcols1 = cols_headers + cols_global
showcols2 = cols_headers + cols_important
shortcols = cols_headers + cols_global + cols_important
def display_df(df, sort_by=["vacancy_formation_M1"], DisplayCols=None):
    pd.set_option('display.max_rows', 100)
    df = df.sort_values(sort_by)
    if DisplayCols is None:
        print(df[showcols1])
        print(df[showcols2])
        print("************")
    else:
        print(df[DisplayCols])
        print("************")

class Predictor:
    def __init__(self, compstr, outheader, supercell=[5, 5, 5]):
        self.compstr = compstr
        self.outfile = outheader + ".csv"
        self.supercell = supercell
        thisHEC = HECMaker(self.compstr, supercell=self.supercell)
        fname = outheader + ".CONTCAR"
        thisHEC.to_file(fname=fname, direct=True, significant_figures=16)
        thisDataMaker = DataMaker(fname, scale=thisHEC.scale)
        thisDataMaker.make_data(outfile=self.outfile, rcut=RCUT, df_calc=None)
        self.data = myData(self.outfile)

    def get_predictions(self, savefile=True):
        key = "vacancy_formation"
        featuress =[["nn_mean_H","nn_max_abs_rdiff_a","nn_mismatch_X","nn_mean_X","nn_mean_diff_VEC"]]
        mname_headers = ["Vacancy_formation"]
        outkeys = ["vacancy_formation_M1"]
        modelname = "GBR"
        for ifrom in range(0, len(featuress)):
            features = featuress[ifrom]
            mname_header = mname_headers[ifrom] + str(ifrom + 1)
            outkey = outkeys[ifrom]
            thisdatafill = DataFill(self.data, key, features,
                                    modelname=modelname,
                                    mname_header=mname_header,
                                    outkey=outkey)
            thisdatafill.fill_data(savefile=savefile, outfile=self.outfile)

        key = "original_vacancy_formation"
        featuress =[["nn_mean_H","nn_max_abs_rdiff_a","nn_mismatch_X","nn_mean_X","nn_mean_diff_VEC"]]
        mname_headers = ["Original_vacancy_formation"]
        outkeys = ["original_vacancy_formation_M1"]
        modelname = "GBR"
        for ifrom in range(0, len(featuress)):
            features = featuress[ifrom]
            mname_header = mname_headers[ifrom] + str(ifrom + 1)
            outkey = outkeys[ifrom]
            thisdatafill = DataFill(self.data, key, features,
                                    modelname=modelname,
                                    mname_header=mname_header,
                                    outkey=outkey)
            thisdatafill.fill_data(savefile=savefile, outfile=self.outfile)

    def display_predictions(self):
        pd.set_option('display.max_rows', 100)
        thisdata = myData(self.outfile)
        print(thisdata.df[showcols2])
        thisdata.save_to(self.outfile, df=thisdata.df)


class Screener:
    def __init__(self, fname, from_database=False):
        self.fname = fname
        if from_database:
            self.df = pd.read_csv(os.path.join(DATA_PATH,fname))
        else:
            self.df = pd.read_csv(fname)

    def screen_elements(self, elements, style="EXCLUDE"):
        ys = self.df["Composition"].to_numpy()
        inds = np.arange(len(ys), dtype=int)
        bads = []
        for i in range(len(ys)):
            compstr = ys[i]
            comp = Composition(compstr)
            eles = []
            for el in comp.elements:
                eles.append(el.symbol)

            isValid = True
            if style[0:3].upper() == "EXA":
                if len(eles) != len(elements):
                    isValid = False
            if isValid:
                for sym in elements:
                    if sym in eles:
                        if style[0:3].upper() == "EXC":
                            isValid = False
                            break
                    else:
                        if style[0:3].upper() == "INC" or style[0:3].upper() == "EXA":
                            isValid = False
                            break
            if not isValid:
                bads.append(i)
        bads = np.array(bads).astype(int)
        inds = np.delete(inds, bads)
        self.df = self.df.iloc[inds]

    def screen_compstrs(self, compstrs):
        rcs = []
        for compstr in compstrs:
            comp = Composition(compstr)
            rcs.append(comp.reduced_formula)
        rcs = np.array(rcs)
        self.df = self.df.set_index("ReducedFormula")
        self.df = self.df.loc[rcs]

    def screen_ncompon(self, ncompons):
        ys = self.df["Composition"].to_numpy()
        inds = np.arange(len(ys), dtype=int)
        bads = []
        for i in range(len(ys)):
            compstr = ys[i]
            isValid = True
            comp = Composition(compstr)
            thisn = len(comp.elements)
            if thisn in ncompons:
                pass
            else:
                isValid = False
            if not isValid:
                bads.append(i)
        bads = np.array(bads).astype(int)
        inds = np.delete(inds, bads)
        self.df = self.df.iloc[inds]

    def ratio_screener(self, key, rmin, rmax):
        ys = self.df[key].to_numpy()
        inds = np.arange(len(ys))
        maxv = np.max(ys)
        if isinstance(rmin, float) or isinstance(rmin, int):
            inds = np.compress(ys > rmin * maxv, inds)
            ys = ys[inds]
        if isinstance(rmax, float) or isinstance(rmax, int):
            inds = np.compress(ys < rmax * maxv, inds)
        self.df = self.df.iloc[inds]

    def value_screener(self, key, vmin, vmax):
        ys = self.df[key].to_numpy()
        inds = np.arange(len(ys))
        if isinstance(vmin, float) or isinstance(vmin, int):
            inds = np.compress(ys >= vmin, inds)
            ys = ys[inds]
        if isinstance(vmax, float) or isinstance(vmax, int):
            inds = np.compress(ys <= vmax, inds)
        self.df = self.df.iloc[inds]

    def percentage_screener(self, key, vmin, vmax):
        ys = self.df[key].to_numpy()
        maxv = np.max(ys)
        minv = np.min(ys)
        span = maxv - minv
        inds = np.arange(len(ys))
        if isinstance(vmin, float) or isinstance(vmin, int):
            inds = np.compress(ys >= vmin * span + minv, inds)
            ys = ys[inds]
        if isinstance(vmax, float) or isinstance(vmax, int):
            inds = np.compress(ys <= vmax * span + minv, inds)
        self.df = self.df.iloc[inds]

    def display_screener(self, sort_by=["vacancy_formation_M1"], DisplayCols=None):
        display_df(self.df, sort_by=sort_by, DisplayCols=DisplayCols)

    def save_screener(self, outfile=None, ShortCols=None):
        if outfile is None:
            outfile = self.fname
            outfile = outfile.split("/")
            outfile = outfile[-1]
            outfile = outfile.replace(".csv", "")
            outfile1 = outfile + "_Screen.csv"
            outfile2 = outfile + "_Screen_short.csv"
        else:
            outfile1 = outfile
            outfile2 = outfile.replace(".csv", "")
            outfile2 = outfile2 + "_short.csv"
        self.df.to_csv(outfile1, index=False, float_format=float_format)
        if ShortCols is None:
            self.df[shortcols].to_csv(outfile2, index=False, float_format=float_format)
        else:
            self.df[ShortCols].to_csv(outfile2, index=False, float_format=float_format)

