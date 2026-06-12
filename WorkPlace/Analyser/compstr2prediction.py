import os
import pandas as pd
import numpy as np
from heceramics.predictor.predict import Predictor, Screener
from heceramics.plot.plot import plot_linear_xys, plot_xss_yss_lines, bar_plot, get_3d_bar, quartiles_plot

supercell = [4, 4, 4]
ykeys = ["vacancy_formation_M1", "original_vacancy_formation_M1"]
Plot_Prediction = True
savefig = False

compstrs = ["MoWVScYC5", "ScYMoWTiC5", "HfZrTiNbTaC5"]
for i in range(len(compstrs)):
    compstr = compstrs[i]
    outfile = compstr + ".csv"
    thispre = Predictor(compstr, outfile, supercell=supercell)
    thispre.get_predictions(savefile=True)
    thispre.display_predictions()

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

