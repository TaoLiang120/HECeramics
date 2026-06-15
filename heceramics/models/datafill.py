import os
import numpy as np
import pandas as pd

from heceramics.data.compstr_util import CompstrUtil
from heceramics.data.data import myData
from heceramics.myglobal import config_vars, Constants, Element_negativity, KeyColumns
from heceramics.models.models import MLRegressor, get_default_outfile

float_format = Constants["float_format"]


class Compstrs2data:
    def __init__(self, compstrs, outfile="mydata.csv"):
        self.compstrs = compstrs
        self.outfile = outfile

    def compstrs2dataframe(self, savefile=True):
        df = pd.DataFrame(columns=KeyColumns)
        df["Composition"] = self.compstrs
        df["pretty_formula"] = df["Composition"].apply(lambda x: CompstrUtil(x).pretty_formula)
        df.to_csv(self.outfile, index=False, float_format=float_format)
        thisdata = myData(self.outfile)
        thisdata.df = thisdata.get_all_features()
        if savefile:
            thisdata.save_to(self.outfile, df=thisdata.df)
        return thisdata


class DataFill:
    def __init__(self, data, key, features, modelname="GBR", mname_header=None, outkey=None):
        self.data = data
        self.key = key
        self.features = features
        self.modelname = modelname
        self.mname_header = mname_header
        if outkey is None:
            if mname_header is None:
                outkey = self.key
            else:
                outkey = self.mname_header
        self.outkey = outkey

        outfile = get_default_outfile(self.data.fname)
        outfile = outfile + "_" + self.modelname + ".csv"
        self.default_outfile = outfile

        self.model = MLRegressor(self.data, [self.key], self.features,
                                 modelname=self.modelname,
                                 mname_header=self.mname_header,
                                 SHAP_Plot=False)

        self.X_test = self.model.generate_X()


    def fill_data(self, savefile=False, outfile=None):
        mname = self.key + self.model.keyapps[0]
        isValid = self.model.load_model(self.key)

        if not isValid:
            raise ValueError(f"Fail to load {mname} model!")

        if self.X_test.shape[0] > 0:
            outs = self.model.get_predictions(self.key, self.X_test)
            self.data.df[self.outkey] = outs

        if savefile:
            if outfile is None:
                outfile = self.default_outfile
            self.data.df.to_csv(outfile, index=False, float_format=float_format)