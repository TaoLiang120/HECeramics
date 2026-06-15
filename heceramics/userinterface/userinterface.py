import yaml

from heceramics.myglobal import config_vars, Constants
from heceramics.predictor.predict import Predictor

class UserInterface:
    def __init__(self, finput="input.yaml"):
        self.finput = finput
        self.parse_input()

    def parse_input(self):
        with open(self.input, 'r') as f:
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
        self.indict = indict
        return indict

    def get_predictions(self):
        thispre = Predictor(self.indict)
        thispre.get_predictions(savefile=True, input_fname=self.indict["Input_fname"])


    def run(self):
        self.get_predictions()


