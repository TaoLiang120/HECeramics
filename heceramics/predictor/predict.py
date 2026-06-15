import os
import numpy as np
import pandas as pd
import yaml
from pymatgen.core.composition import Composition

from heceramics.myglobal import Element_negativity
from heceramics.myglobal import config_vars, Constants
from heceramics.data.data import myData
from heceramics.models.datafill import DataFill

CONTCAR_PATH = config_vars["CONTCAR_PATH"]
DATA_PATH = config_vars["DATA_PATH"]
float_format = Constants["float_format"]


showcols1 = []
showcols2 = []
shortcols = []
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
    def __init__(self, indict):
        self.indict = indict

    @staticmethod
    def parse_input(fname):
        with open(fname, 'r') as f:
            setts = yaml.safe_load(f)

        indict = {}
        indict["InFile"] = "compstrs.json"
        indict["OutFile"] = "prediction_out.csv"
        indict["Input_fname"] = None

        if "InFile" in setts:
            indict["InFile"] = setts["InFile"]

        if "OutFile" in setts:
            indict["OutFile"] = setts["OutFile"]

        if "Input_fname" in setts and isinstance(setts["Input_fname"], str):
            indict["Input_fname"] = setts["Input_fname"]
        return indict

    @classmethod
    def from_finput(cls, finput):
        indict = cls.parse_input(finput)
        thisobj = cls(indict)
        return thisobj

    def compute_features(self, input_fname=None):
        thisfname = self.indict["OutFile"]
        if isinstance(input_fname, str):
            thisfname = input_fname




    def get_predictions(self, savefile=True, input_fname=None):

        key = "lattice_constant"
        featuress =[["features"]] ##put features here
        mname_headers = ["lattice_constant"]
        outkeys = ["lattice_constant_M1"]
        modelname = "GBR"
        for ifrom in range(0, len(featuress)):
            features = featuress[ifrom]
            mname_header = mname_headers[ifrom] + str(ifrom + 1)
            outkey = outkeys[ifrom]
            thisdatafill = DataFill(self.data, key, features,
                                    modelname=modelname,
                                    mname_header=mname_header,
                                    outkey=outkey)
            thisdatafill.fill_data(savefile=savefile, outfile=self.indict["outfile"])


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
        self.df = self.df.set_index("pretty_formula")
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

    def display_screener(self, sort_by=["Composition"], DisplayCols=None):
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

