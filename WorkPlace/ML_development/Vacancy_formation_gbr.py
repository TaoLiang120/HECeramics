from heceramics.work.GBR_functions import train_model
float_format = '%.8f'
featuress = [["nn_mean_H","nn_min_rdiff_a","nn_mean_X","nn_mismatch_X","nn_max_rdiff_a"]]


featuress = [
             ["nn_mean_H","nn_max_abs_rdiff_a","nn_mismatch_X","nn_mean_X","nn_mean_diff_VEC"]
             ]
isLOAD = False
SHAP_Plot = isLOAD
infnames = ["Weighted_ALL_HECs.csv"]
mname_headers = ["Vacancy_formation"]
colorkey = None
keys = ["vacancy_formation"]
thress = [0.90, 0.916]  ##0.916 for vacancy_formation, 0.865 for original
regressor = "GBR"
HECS = None
warm_start = True
if isLOAD:
    loadmodel = True
    savemodel = False
    nloop = 1
else:
    loadmodel = False
    savemodel = True
    nloop = 1000
min_samples_leaf = 6
thres_diffs = [3.0, 4.5]
savemodel = True

itypestart = 0
for ifrom in range(0,len(infnames)):
    for itype in range(itypestart,len(featuress)):
        fname = infnames[ifrom]
        features = featuress[itype]
        mname_header = mname_headers[ifrom] + str(itype+1)
        thres = thress[itype]
        thres_diff = thres_diffs[itype]

        train_model(fname, mname_header, thres, keys, features, regressor,
                  HECS=HECS, colorkey=colorkey, loadmodel=loadmodel, savemodel=savemodel,
                  nloop=nloop, thres_diff=thres_diff,
                  set2all=True, min_samples_leaf=min_samples_leaf,
                  warm_start=warm_start, SHAP_Plot=SHAP_Plot, SHAP_output=False)
        print(f"###############")
        print(f"#### done ######")
        print(f"###############")




featuress = [
            ["nn_mean_H","nn_max_abs_rdiff_a","nn_mismatch_X","nn_mean_X","nn_mean_diff_VEC"]
            ]

isLOAD = False
SHAP_Plot = isLOAD
infnames = ["Weighted_ALL_HECs.csv"]
mname_headers = ["Original_vacancy_formation"]
colorkey = None
keys = ["original_vacancy_formation"]
thress = [0.85, 0.86]  ##0.916 for vacancy_formation, 0.865 for original
regressor = "GBR"
HECS = None
warm_start = True
if isLOAD:
    loadmodel = True
    savemodel = False
    nloop = 1
else:
    loadmodel = False
    savemodel = True
    nloop = 1000
min_samples_leaf = 6
thres_diffs = [3.0, 5.0]

savemodel = True
itypestart  = 0
for ifrom in range(0,len(infnames)):
    for itype in range(itypestart,len(featuress)):
        fname = infnames[ifrom]
        features = featuress[itype]
        mname_header = mname_headers[ifrom] + str(itype+1)
        thres = thress[itype]
        thres_diff = thres_diffs[itype]

        train_model(fname, mname_header, thres, keys, features, regressor,
                  HECS=HECS, colorkey=colorkey, loadmodel=loadmodel, savemodel=savemodel,
                  nloop=nloop, thres_diff=thres_diff,
                  set2all=True, min_samples_leaf=min_samples_leaf,
                  warm_start=warm_start, SHAP_Plot=SHAP_Plot, SHAP_output=False)

        print(f"###############")
        print(f"#### done ######")
        print(f"###############")

