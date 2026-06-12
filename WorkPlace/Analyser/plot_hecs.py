import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from heceramics.myglobal import config_vars, Constants, HECs
from heceramics.data.data import myData
from heceramics.models.datafill import DataFill
from heceramics.plot.plot import plot_linear_xys, plot_xss_yss_lines, bar_plot, get_3d_bar, quartiles_plot

CONTCAR_PATH = config_vars["CONTCAR_PATH"]
DATA_PATH = config_vars["DATA_PATH"]
float_format = Constants["float_format"]

fname4ALL = "ALL_HECs.csv"
fname4weighted_ALL = "Weighted_ALL_HECs.csv"

thisHECs = ["MoWVScYC5", "ScYMoWTiC5", "HfZrTiNbTaC5"]
thisHECs = HECs
from_DATA_PATH = True
Fill_Data = True

## parameter for DataFill ##
keys = ["vacancy_formation", "original_vacancy_formation"]
featuress = [
            ["nn_mean_H","nn_max_abs_rdiff_a","nn_mismatch_X","nn_mean_X","nn_mean_diff_VEC"],
            ["nn_mean_H","nn_max_abs_rdiff_a","nn_mismatch_X","nn_mean_X","nn_mean_diff_VEC"],
            ]
mname_headers = [
                 "Vacancy_formation1", "Original_vacancy_formation1"
                ]
outkeys = [
           "vacancy_formation_M1", "original_vacancy_formation_M1",
           ]
if Fill_Data:
    for i in range(len(thisHECs)):
        hec = thisHECs[i]
        if from_DATA_PATH:
            fname = os.path.join(DATA_PATH, hec + ".csv")
        else:
            fname = hec + ".csv"

        for ifill in range(len(keys)):
            data = myData(fname)
            thisdatafill = DataFill(data, keys[ifill], featuress[ifill],
                                modelname="GBR", mname_header=mname_headers[ifill], outkey=outkeys[ifill])
            thisdatafill.fill_data(savefile=True, outfile=fname)

        data = myData(fname)
        data.df = data.df.sort_values(["vacancy_formation"])
        data.save_to(fname, df=data.df)

## parameter for DataFill ##

ykeys = ["original_vacancy_formation", "original_vacancy_formation_M1"]
ykeys = ["vacancy_formation", "vacancy_formation_M1"]

ykeys = ["original_vacancy_formation_M1"]
ykeys = ["vacancy_formation_M1"]
ykeys = ["vacancy_formation", "vacancy_formation_M1"]
One_Plot = False
savefig = True
outfig = "custom_"+ykeys[0] + ".png"

if One_Plot:
    xss = []
    yss = []
    labels = []
else:
    xss = None
    yss = None
    labels = None

for i in range(len(thisHECs)):
    hec = thisHECs[i]
    if from_DATA_PATH:
        fname = os.path.join(DATA_PATH, hec + ".csv")
    else:
        fname = hec + ".csv"

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

    compstr = df.at[0, "Composition"]
    for iykey in range(len(ykeys)):
        if len(ykeys) == 2:
            if iykey == 0:
                labels.append(hec + "_calc")
            else:
                labels.append(hec + "_pred")
        else:
            labels.append(hec)
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
        print(f"hec: {hec} compstr:{compstr} ykey:{ykeys[iykey]}")
        print(f"ydiff_mean:{ydiff_mean} ydiff_std:{ydiff_std}")
        print("---")
    print("===")

    if One_Plot:
        pass
    else:
        outfig = hec + "_" + compstr + "_" + ykeys[len(ykeys)-1] + ".png"
        plot_xss_yss_lines(xss, yss, style="scatter", labels=labels,
                           Label_compstr=False, Comp_Labels=None,
                           savefig=savefig, outfile=outfig)

if One_Plot:
    plot_xss_yss_lines(xss, yss, style="scatter", labels=labels,
                       Label_compstr=False, Comp_Labels=None,
                       savefig=savefig, outfile=outfig)
else:
    pass




