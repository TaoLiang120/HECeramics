import os

import numpy as np
import pandas as pd

from heceramics.data.data import myData, DATA_PATH
from heceramics.models.models import MLRegressor


float_format = '%.8f'

performance_keys = ["iloop", "key", "mse_test", "rmse_test", "mae_test", "rmae_test", "r2_test",
                    "mse_all", "rmse_all", "mae_all", "rmae_all", "r2_all"]
def train_model(fname, mname_header, thres, keys, features, regressor,
                    HECS=None, colorkey=None, loadmodel=False, savemodel=True, nloop=100,
                    perform_df=None, elements=None,
                    thres_diff=5.0, set2all=False, min_samples_leaf=None, warm_start=True, TEST=False,
                    SHAP_Plot=True, SHAP_output=False, PDP_output=False):

    thisdata = myData(os.path.join(DATA_PATH, fname))
    thisdata.validate_dataframe(df=None, set2data=True)
    if HECS is None:
        pass
    else:
        thisdata.gooddf, baddf = thisdata.select_by_HECS(thisdata.gooddf, HECS, set2gooddf=True)

    if perform_df is None: perform_df = pd.DataFrame(columns=performance_keys)
    perform_df = perform_df.set_index(performance_keys[0])

    normalization = False
    extended = False
    Visualize_Results = False
    Find_PDP = True
    Find_interPDP = False
    interactive_features = None

    Save_Pred2Data = False
    savefig = True

    activation = "relu"
    learning_rate_style = "adaptive"  #"constant"
    learning_rate = 0.01
    max_iter = 1000

    plot_predict = savemodel
    plot_predict_all = True
    compute_R2 = True
    if TEST:
        Find_PDP = False
        SHAP_Plot = False
        Save_Pred2Data = False
    R2 = 0.0
    R2_diff = 100.0
    iloop = 0
    r2_avg = 0.0
    while R2 < thres and iloop < nloop:
        thisMLR = MLRegressor(thisdata, keys, mname_header=mname_header, modelname=regressor,
                              elements=elements, features=features, Find_PDP=Find_PDP, Find_interPDP=Find_interPDP,
                              interactive_features=interactive_features, SHAP_Plot=SHAP_Plot,
                              activation=activation, learning_rate_style=learning_rate_style,
                              learning_rate=learning_rate, min_samples_leaf=min_samples_leaf, warm_start=warm_start)
        thisMLR.regression_models(Save_Pred2Data=Save_Pred2Data, loadmodel=loadmodel, savemodel=savemodel,
                                  savefig=savefig, plot_predict=plot_predict, plot_predict_all=plot_predict_all,
                                  colorkey=colorkey, SHAP_data="ALL", SHAP_output=SHAP_output, PDP_output=PDP_output)

        R2 = thisMLR.performance_df.loc[keys[0], "r2_test"]
        R2_all = thisMLR.performance_df.loc[keys[0], "r2_all"]
        R2_diff = abs(R2 - R2_all) * 100.0
        print("00000 GBR FITTING 00000")
        print(f"mname:{mname_header} iloop:{iloop} R2: {R2} R2_All: {R2_all} R2_diff:{R2_diff}")

        thisdf = thisMLR.performance_df.copy()
        thisdf[performance_keys[0]] = [iloop]
        thisdf = thisdf.reset_index().set_index(performance_keys[0])
        perform_df.loc[iloop] = thisdf.loc[iloop]

        if set2all: R2 = R2_all
        if R2_diff > thres_diff: R2 = thres - 0.01
        r2_avg += R2_all
        print(f"*********** R2:  {R2} *****************")
        iloop += 1
    r2_avg /= iloop
    print(f"===== r2_avg: {r2_avg}  =====")
    perform_df["R2_AVG_ALL"] = r2_avg
    return perform_df
