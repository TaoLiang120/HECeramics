import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from heceramics.myglobal import config_vars, Constants
from heceramics.data.data import myData
from heceramics.models.datafill import DataFill
from heceramics.predictor.predict import Predictor
from heceramics.plot.plot import plot_linear_xys, plot_xss_yss_lines, bar_plot, get_3d_bar, quartiles_plot

CONTCAR_PATH = config_vars["CONTCAR_PATH"]
DATA_PATH = config_vars["DATA_PATH"]
float_format = Constants["float_format"]

supercell = [5, 5, 5]
ykeys = ["vacancy_formation_M1", "original_vacancy_formation_M1"]
Plot_Prediction = True
savefig = True
compstrs = ["MoWVScYC5", "ScYMoWTiC5"]

compstrs = ["TiTaVWZrC5"]
for i in range(len(compstrs)):
    compstr = compstrs[i]
    outfile = compstr + ".csv"
    thispre = Predictor(compstr, outfile, supercell=supercell)
    thispre.get_predictions(savefile=True)
    #thispre.display_predictions()

    if Plot_Prediction:
        for ikey in range(len(ykeys)):
            ykey = ykeys[ikey]
            xss = []
            yss = []
            labels = []
            df = pd.read_csv(outfile)
            df = df.sort_values([ykey])
            xss.append(np.arange(len(df), dtype=int) + 1)
            yss.append(df[ykey].to_numpy())
            labels.append(compstr)
            outfig = compstr + "_" + ykey + ".png"
            plot_xss_yss_lines(xss, yss, style="scatter", labels=labels,
                               Label_compstr=False, Comp_Labels=None,
                               savefig=savefig, outfile=outfig)



One_Plot = True
savefig = True
outfig = "custom_"+ykeys[0] + ".png"
ykeys = ["vacancy_formation_M1"]
ncol = 2

if One_Plot:
    xss = []
    yss = []
    labels = []
else:
    xss = None
    yss = None
    labels = None

for i in range(len(compstrs)):
    compstr = compstrs[i]
    fname = compstr + ".csv"

    df = pd.read_csv(fname)
    inds = np.arange(len(df), dtype=int)
    inds = np.compress(df[ykeys[0]] < 2.0, inds)
    df = df.loc[inds]
    #df = df.sort_values([ykeys[0]])

    if One_Plot:
        pass
    else:
        xss = []
        yss = []
        labels = []

    for iykey in range(len(ykeys)):
        if len(ykeys) == 2:
            if iykey == 0:
                labels.append(compstr + "_calc")
            else:
                labels.append(compstr + "_pred")
        else:
            labels.append(compstr)
        df = df.sort_values([ykeys[iykey]])
        xss.append(np.arange(len(df), dtype=int)+1)
        ys = df[ykeys[iykey]].to_numpy()
        ntot = len(ys)
        nstart = int(ntot / 4)
        nend = int(3 * ntot / 4)
        ys_chop = ys[nstart:nend]
        ymean = np.mean(ys_chop)
        yshift = np.append([ys_chop[0]], ys_chop[0:len(ys_chop)-1])
        ydiffs = ys_chop-yshift
        ydiff_mean = np.mean(ydiffs)
        ydiff_std = np.std(ydiffs)
        yss.append(ys)
        print(f"compstr:{compstr} ykey:{ykeys[iykey]}")
        print(f"ydiff_mean:{ydiff_mean} ydiff_std:{ydiff_std}")
        print("---")


    if One_Plot:
        pass
    else:
        outfig = compstr + "_" + ykeys[len(ykeys)-1] + ".png"
        plot_xss_yss_lines(xss, yss, style="scatter", labels=labels,
                           Label_compstr=False, Comp_Labels=None,
                           ncol=ncol, savefig=savefig, outfile=outfig)

if One_Plot:
    plot_xss_yss_lines(xss, yss, style="scatter", labels=labels,
                       Label_compstr=False, Comp_Labels=None,
                       ncol=ncol, savefig=savefig, outfile=outfig)
else:
    pass




