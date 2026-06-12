import os
from heceramics.myglobal import config_vars, Constants, Element_negativity
from heceramics.models.models import MLRegressor, get_default_outfile

float_format = Constants["float_format"]

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