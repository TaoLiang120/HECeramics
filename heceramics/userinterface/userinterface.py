import os
import copy
import numpy as np
import pandas as pd
import yaml

from heceramics.myglobal import config_vars, Constants, HECs
from heceramics.predictor.predict import Predictor
from heceramics.plot.plot import plot_xss_yss_lines
from heceramics.utils.utils import PrettyFormula, validate_compstr, validate_HECs
from heceramics.utils.utils import load_complist_from_json, save_complist_to_json

Validate_keys = ["original_vacancy_formation_M1", "vacancy_formation_M1"]
Labels4prediction =  ["Pred_Eform_Gr", "Pred_Eform"]

class UserInterface:
    def __init__(self):
        self.input = "input.yaml"
        self.parse_input()
    def parse_input(self):
        with open(self.input, 'r') as f:
            setts = yaml.safe_load(f)

        self.purpose = "prediction+plot"
        if "purpose" in setts:
            purposestr = setts["purpose"].lower()
            if "plot" in purposestr and "pred" in purposestr:
                self.purpose = "prediction+plot"
            elif "plot" in purposestr:
                self.purpose = "plot"
            elif "pred" in purposestr:
                self.purpose = "prediction"
            else:
                raise ValueError("Invalid purpose !")


        if "HECs" not in setts:
            raise ValueError("You must provide HECs in input.yaml!")
        thisHECs = setts["HECs"]
        source = "manual"
        if "source" in thisHECs:
            thissource = thisHECs["source"]
            if "file" in thissource:
                source = "file"

        self.source = source
        if source == "file":
            if not "filename" in thisHECs:
                raise ValueError("You must provide a filename (json format) for HECs !")
            else:
                self.compstrs = load_complist_from_json(thisHECs["filename"])
        else:
            thisstr = thisHECs["HECs"]
            thisstr = "".join(thisstr.split(" "))
            self.compstrs = thisstr.split(",")

        self.supercells = [[4, 4, 4]]
        if "supercell" in setts:
            supercellstr = setts["supercell"]
            supercells = []
            for i in range(len(supercellstr)):
                thissc = supercellstr[i].split(",")
                try:
                    xyz = [int(thissc[0]), int(thissc[1]), int(thissc[2])]
                    supercells.append(xyz)
                except:
                    pass
            if len(supercells) > 0:
                self.supercells = supercells

        self.keys4plot = ["vacancy_formation_M1", "original_vacancy_formation_M1"]
        if "keys4plot" in setts:
            thisstr = setts["keys4plot"]
            if "both" in thisstr.lower():
                pass
            else:
                thisstr = "".join(thisstr.split(" "))
                thisstrs = thisstr.split(",")
                keys4plot = []
                for thisstr in thisstrs:
                    if "graphite" in thisstr.lower():
                        keys4plot.append(Validate_keys[0])
                    elif "carbide" in thisstr.lower():
                        keys4plot.append(Validate_keys[1])

        self.style4plot = "sep"
        if "style4plot" in setts:
            thisstr = setts["style4plot"]
            if thisstr[0:3].upper() == "SEP":
                self.style4plot = "separate"
            elif thisstr[0:3].upper() == "GRO":
                self.style4plot = "group"
            elif thisstr[0:3].upper() == "ONE":
                self.style4plot = "one"
            else:
                raise ValueError("Invalid style4plot ! Options are SEP, GRO, ONE")

        self.show_legend = True
        if "show_legend" in setts:
            self.show_legend = setts["show_legend"]

        self.fig_format = "svg"
        if "fig_format" in setts:
            self.fig_format = setts["fig_format"]

        self.savefig = True
        if "savefig" in setts:
            self.savefig = setts["savefig"]

        compstrs = []
        for compstr in self.compstrs:
            thisPF = PrettyFormula(compstr)
            compstr = thisPF.pretty_formula
            isvalid = validate_compstr(compstr)
            if isvalid:
                compstrs.append(compstr)
        if len(compstrs) == 0:
            raise ValueError("no validate HEC composition!")
        else:
            self.compstrs = copy.deepcopy(compstrs)

    def get_predictions(self):
        for icomp in range(len(self.compstrs)):
            compstr = self.compstrs[icomp]
            outfold = compstr
            if not os.path.isdir(outfold):
                os.makedirs(outfold)

            for isc in range(len(self.supercells)):
                thissc = self.supercells[isc]
                outhead = str(thissc[0]) + "x" + str(thissc[1]) + "x" + str(thissc[2])
                outheader = outfold + "/" + outhead
                thispre = Predictor(compstr, outheader, supercell=thissc)
                thispre.get_predictions(savefile=True)

    def get_plot(self):
        xss4one = []
        yss4one = []
        labels4one = []
        outfig4one = "HECs" + "." + self.fig_format
        for icomp in range(len(self.compstrs)):
            compstr = self.compstrs[icomp]
            outfold = compstr
            if not os.path.isdir(outfold):
                raise ValueError("The folder for " + compstr + " is not exist!")

            xss4group = []
            yss4group = []
            labels4group = []
            outfig4group = outfold + "/" + compstr + "." + self.fig_format
            for isc in range(len(self.supercells)):
                thissc = self.supercells[isc]
                outhead = str(thissc[0]) + "x" + str(thissc[1]) + "x" + str(thissc[2])
                outheader = outfold + "/" + outhead
                fname = outheader + ".csv"
                if not os.path.isfile(fname):
                    raise ValueError("The " + fname + " is not exist!")
                df = pd.read_csv(fname)
                xss4sep = []
                yss4sep = []
                labels4sep = []
                outfig4sep = outheader + "." + self.fig_format
                for ikey in range(len(self.keys4plot)):
                    ykey = self.keys4plot[ikey]
                    ilabel = Validate_keys.index(ykey)
                    thislabel = Labels4prediction[ilabel]

                    df = df.sort_values([ykey])
                    xs = np.arange(len(df), dtype=int) + 1
                    ys = df[ykey].to_numpy()

                    if "sep" in self.style4plot:
                        xss4sep.append(xs)
                        yss4sep.append(ys)
                        labels4sep.append(thislabel)
                    elif "group" in self.style4plot:
                        xss4group.append(xs)
                        yss4group.append(ys)
                        labels4group.append(outhead + "_" + thislabel)
                    elif "one" in self.style4plot:
                        xss4one.append(xs)
                        yss4one.append(ys)
                        labels4one.append(compstr + "_" + outhead + "_" + thislabel)

                if "sep" in self.style4plot:
                    plot_xss_yss_lines(xss4sep, yss4sep, style="scatter", labels=labels4sep,
                                       show_legend=self.show_legend, Label_compstr=False, Comp_Labels=None,
                                       ncol=1, savefig=self.savefig, outfile=outfig4sep, format=self.fig_format)
            if "group" in self.style4plot:
                plot_xss_yss_lines(xss4group, yss4group, style="scatter", labels=labels4group,
                                   show_legend=self.show_legend, Label_compstr=False, Comp_Labels=None,
                                   ncol=1, savefig=self.savefig, outfile=outfig4sep, format=self.fig_format)
        if "one" in self.style4plot:
            plot_xss_yss_lines(xss4one, yss4one, style="scatter", labels=labels4one,
                               show_legend=self.show_legend, Label_compstr=False, Comp_Labels=None,
                               ncol=1, savefig=self.savefig, outfile=outfig4sep, format=self.fig_format)

    def run(self):
        if "pred" in self.purpose:
            self.get_predictions()

        if "plot" in self.purpose:
            self.get_plot()
