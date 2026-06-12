import copy
import os
import pickle
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
#from scipy import interpolate
#from sklearn import svm
from sklearn import ensemble
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
#from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

#from heceramics.data.data import myData
from heceramics.myglobal import config_vars, Constants, Element_negativity
from heceramics.plot.plot import plot_linear_xys

MODEL_PATH = config_vars["MODEL_PATH"]
float_format = Constants["float_format"]


def get_default_outfile(fname):
    fname = fname.split("/")
    fname = fname[-1]
    fname = fname.replace(".csv", "")
    fname = fname.replace("_GBR", "")
    fname = fname.replace("_HGBR", "")
    fname = fname.replace("_MLPR", "")
    #fname = fname.replace("_NDIP", "")
    return fname
def get_r2(ytest, ypred):
    yr = ytest - ypred
    ydiff = ytest - np.mean(ytest)
    r2 = 1.0 - np.sum(yr * yr) / np.sum(ydiff * ydiff)
    return r2

class MLRegressor:
    def __init__(self, data, keys, features, modelname="GBR", mname_header=None, elements=Element_negativity,
                 normalization=False, valid_size=0.1, random_state=None,
                 learning_rate=0.01, loss="squared_error", tol=0.001,
                 Visualize_Results=False, Importance_mode=[0, 1], Find_PDP=False,
                 Vertical=False, features4plot=None, Find_interPDP=False, interactive_features=None, subsample=None,
                 SHAP_Plot=False, SHAP_style="beeswarm", max_display=15,
                 n_estimators=500, max_depth=None, min_samples_split=None, min_samples_leaf=None,
                 max_iter=500, max_leaf_nodes=None,
                 hidden_layer_sizes=None, learning_rate_style="adaptive", activation="relu", solver="adam",
                 early_stopping=True, max_fun=15000, warm_start=True):

        self.SAVE_PATH = "MLR_Models"

        self.data = data
        #self.data.validate_dataframe(df=None, set2data=True)
        if "HGBR" in modelname:
            self.modelname = "HGBR"
        elif "GBR" in modelname:
            self.modelname = "GBR"
        else:
            self.modelname = "MLPR"

        if mname_header is None:
            outfile = get_default_outfile(self.data.fname)
            outfile += "_" + self.modelname
        else:
            outfile = mname_header + "_" + self.modelname

        self.outfig_header = outfile + "_"
        self.mname_header = outfile + "_"

        self.keys = keys
        if elements is None:
            elements = copy.deepcopy(Element_negativity)
        self.elements = elements

        self.normalization = normalization
        self.valid_size = valid_size
        self.random_state = random_state

        self.feature_type = "custom"
        labels = []
        for feature in features:
            labels.append(feature)

        self.features = features
        self.labels = labels
        self.nfeature = len(self.features)

        if features4plot is None: features4plot = np.arange(self.nfeature, dtype=int)
        self.features4plot = features4plot

        self.nsplit = 1

        if self.normalization:
            self.data.gooddf = self.data.normalization(self.data.gooddf, keys=self.keys,
                                                       savefile=False, outfile=None, Add_norm=False)
            self.data.gooddf = self.data.normalization(self.data.gooddf, keys=self.features,
                                                       savefile=False, outfile=None, Add_norm=False)

        for key in self.keys:
            if key in self.features:
                raise ValueError(f"The {key} in both features and target keys!")

        self.keyapps = ["_reg", "_mse", "_rmse", "_mae", "_rmae"]
        self.init_models()

        self.Visualize_Results = Visualize_Results
        if isinstance(Importance_mode, bool):
            if Importance_mode:
                Importance_mode = [0, 1]
            else:
                Importance_mode = None
        elif isinstance(Importance_mode, str):
            if Importance_mode.upper() == "BOTH":
                Importance_mode = [0, 1]
            elif "PERMU" in Importance_mode.upper():
                Importance_mode = [1]
            else:
                Importance_mode = [0]
        elif isinstance(Importance_mode, list):
            m = []
            l = [0, 1]
            for i in Importance_mode:
                if i in l: m.append(i)
            if len(m) == 0:
                Importance_mode = None
            else:
                Importance_mode = m[0:len(m)]
        else:
            Importance_mode = None

        self.Importance_mode = Importance_mode
        self.Find_PDP = Find_PDP
        self.Find_interPDP = Find_interPDP
        self.interactive_features = interactive_features
        if self.Find_interPDP:
            if interactive_features is None:
                self.inter_features = []
                for i in range(self.nfeature):
                    for j in range(i + 1, self.nfeature):
                        self.inter_features.append((i, j))
                self.ninter_plot = len(self.inter_features)
            else:
                interactive_features = np.array(interactive_features)
                if len(interactive_features.shape) == 1:
                    self.inter_features = []
                    ninter_plot = int(len(interactive_features) / 2)
                    for i in range(ninter_plot):
                        ii = self.features.index(interactive_features[i * 2])
                        jj = self.features.index(interactive_features[2 * i + 1])
                        self.inter_features.append((ii, jj))
                else:
                    for i in range(interactive_features.shape[0]):
                        ii = self.features.index(interactive_features[i][0])
                        jj = self.features.index(interactive_features[i][1])
                        self.inter_features.append((ii, jj))
                self.ninter_plot = len(self.inter_features)
        else:
            self.inter_features = None
            self.ninter_plot = 0

        self.Vertical = Vertical
        ndata = len(self.data.gooddf)
        if subsample is None:
            subsample = min(int(ndata / 30), 100)
        self.subsample = subsample
        if self.modelname == "HGBR" or self.modelname == "MLPR":
            self.Visualize_Results = False
            self.Importance_mode = None

        self.SHAP_Plot = SHAP_Plot
        self.SHAP_style = SHAP_style
        self.max_display = max_display

        self.nele = len(self.elements)
        if max_depth is None: max_depth = self.nfeature
        if min_samples_leaf is None:
            min_samples_leaf = 3

        if self.modelname == "HGBR":
            if max_leaf_nodes is None:
                max_leaf_nodes = 8 * 3
            self.params = {
                "loss": loss,
                "learning_rate": learning_rate,
                "max_iter": max_iter,
                "max_leaf_nodes": max_leaf_nodes,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "validation_fraction": valid_size,
                "tol": tol,
                "random_state": random_state,
                "warm_start": warm_start,
            }
        elif self.modelname == "GBR":
            if min_samples_split is None:
                min_samples_split = 3
            self.params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "learning_rate": learning_rate,
                "loss": loss,
                "tol": tol,
                "warm_start": warm_start,
            }
        else:
            if hidden_layer_sizes is None:
                nlayer = max(int(ndata / 64), 1)
                nlayer = min(nlayer, 100)
                nlayer += 2
                nsize = 2.0 * int(np.sqrt(ndata / (nlayer - 2)))
                nint = round(np.log(nsize) / np.log(2), 0)
                nsize = int(np.power(2, nint))
                if nsize < 4: nsize = 4
                if nsize > 128: nsize = 128
                hidden_layer_sizes = (nlayer, nsize)
            self.hidden_layer_sizes = hidden_layer_sizes

            if early_stopping: max_iter = 100000
            self.params = {
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "activation": activation,
                "solver": solver,
                "learning_rate": learning_rate_style,
                "learning_rate_init": learning_rate,
                "max_iter": max_iter,
                "random_state": random_state,
                "early_stopping": early_stopping,
                "validation_fraction": valid_size,
                "tol": tol,
                "max_fun": max_fun,
            }

        self.performance_keys = ["key", "mse_test", "rmse_test", "mae_test", "rmae_test", "r2_test",
                                 "mse_all", "rmse_all", "mae_all", "rmae_all", "r2_all"]
        self.performance_file = self.mname_header + "performance.csv"
        self.performance_df = pd.DataFrame(columns=self.performance_keys)
        self.performance_df["key"] = self.keys
        for key in self.performance_keys[1:len(self.performance_keys)]:
            self.performance_df[key] = np.zeros(len(self.keys))
        self.performance_df = self.performance_df.set_index("key")

    def init_models(self):
        self.models = {}
        for key in self.keys:
            for app in self.keyapps[0:1]:
                thiskey = key + app
                self.models[thiskey] = None

    def load_model(self, key):
        isValid = True
        mname = key + self.keyapps[0]
        fname = self.mname_header + mname
        if os.path.isfile(os.path.join(MODEL_PATH, self.SAVE_PATH, fname)):
            with open(os.path.join(MODEL_PATH, self.SAVE_PATH, fname), 'rb') as f:
                try:
                    self.models[mname] = pickle.load(f)
                except:
                    print(f"Fail to load {os.path.join(MODEL_PATH, self.SAVE_PATH, fname)}")
                    isValid = False
        else:
            print(f"File {os.path.join(MODEL_PATH, self.SAVE_PATH, fname)} is not existed!")
            isValid = False
        return isValid

    def save_model(self, key):
        mname = key + self.keyapps[0]
        fname = self.mname_header + mname
        if not os.path.isdir(os.path.join(MODEL_PATH, self.SAVE_PATH)):
            os.mkdir(os.path.join(MODEL_PATH, self.SAVE_PATH))
        with open(os.path.join(MODEL_PATH, self.SAVE_PATH, fname), 'wb') as f:
            pickle.dump(self.models[mname], f)

    def save_to_summary(self, performdict):
        fname = self.performance_file
        if os.path.isfile(os.path.join(os.getcwd(), self.SAVE_PATH, fname)):
            df = pd.read_csv(os.path.join(os.getcwd(), self.SAVE_PATH, fname))
            self.performance_df = df.copy()
            self.performance_df = self.performance_df.set_index("key")
        if "key" not in performdict: raise ValueError("performance dict must have 'key' keyword.")
        for key in performdict:
            if key in self.performance_keys[1:len(self.performance_keys)]:
                self.performance_df.loc[performdict["key"], key] = performdict[key]
        self.performance_df.to_csv(os.path.join(os.getcwd(), self.SAVE_PATH, fname), index=True,
                                   float_format=float_format)

    def generate_X(self):
        X = []
        for feature in self.features:
            thisx = self.data.gooddf[feature].to_numpy()
            X.append(thisx)
        X = np.array(X)
        X = X.T
        return X

    def generate_data(self, key, test_size=None, random_state=None):
        if random_state is None: random_state = self.random_state
        if test_size is None: test_size = self.valid_size

        X = self.generate_X()
        y = self.data.gooddf[key].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test, X, y

    def get_predictions(self, key, x):
        mname = key + self.keyapps[0]
        model = self.models[mname]
        preds = model.predict(x)
        return preds

    def plot_predictions(self, ytest, ypred, colorkey=None):
        if colorkey is None:
            colors = None
        else:
            colors = self.data.gooddf[colorkey].to_numpy()
            if len(colors) != len(ytest): colors = None

        fig = plot_linear_xys(ytest, ypred, cmap="jet", colors=colors, style="scatter", plot_xx=True, ShowColorbar=True)
        return fig

    def get_mse_mae(self, X_test, y_test, key, plot_predict=True, savefig=False, colorkey=None,
                    App="_test", loadmodel=False):
        if loadmodel:
            isValid = self.load_model(key)
            if not isValid:
                print("Cannot load " + self.mname_header + key + " model!")
                return
        performdict = {}
        performdict["key"] = key
        y_pred = self.get_predictions(key, X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse = np.sqrt(mse)
        rmse = 100.0 * mse / np.mean(y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmae = 100.0 * mae / np.mean(y_pred)
        r2 = get_r2(y_test, y_pred)

        performdict["mse" + App] = mse
        performdict["rmse" + App] = rmse
        performdict["mae" + App] = mae
        performdict["rmae" + App] = rmae
        performdict["r2" + App] = r2
        print("*******************************************")
        print(f"Evaluation of {self.modelname} Gradient Boost regressor _ {key}:")
        print(f"MSE on {App} set: {mse} Relative MSE: {rmse}")
        print(f"MAE on {App} set: {mae} Relative MAE: {rmae} R2:{r2}")
        self.save_to_summary(performdict)
        if plot_predict:
            self.plot_predictions(y_test, y_pred, colorkey=colorkey)
            if savefig:
                outfile = self.outfig_header + key + App + "_predict.png"
                outfile = os.path.join(self.SAVE_PATH, outfile)
                plt.savefig(outfile, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        print("*******************************************")

    def save_predictions2data(self, key, predictions2data=True, plot_predict_all=False, savefig=False, colorkey=None):
        preds = self.get_predictions(key, self.generate_X())
        outkey = "Predicted_" + key
        if predictions2data:
            df = self.data.gooddf
            df[outkey] = preds
            fname = self.data.fname
            if df.index.name == "CompID":
                df.to_csv(fname, index=True, float_format=float_format)
            else:
                df.to_csv(fname, index=False, float_format=float_format)

        if plot_predict_all:
            y_test = self.data.gooddf[key].to_numpy()
            self.get_mse_mae(self.generate_X(), y_test, key, plot_predict=True, savefig=savefig, colorkey=colorkey,
                             App="_all")

    def visualize_results(self, X_test, y_test, key):
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        mname = key + self.keyapps[0]
        model = self.models[mname]
        test_score = np.zeros((self.params["n_estimators"],), dtype=np.float64)
        for i, y_pred in enumerate(model.staged_predict(X_test)):
            test_score[i] = mean_squared_error(y_test, y_pred)

        fig = plt.figure(figsize=(6, 6))
        plt.subplot(1, 1, 1)
        plt.title("Deviance")
        plt.plot(
            np.arange(self.params["n_estimators"]) + 1,
            model.train_score_,
            "b-",
            label="Training Set Deviance",
        )
        plt.plot(
            np.arange(self.params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
        )
        plt.legend(loc="upper right")
        plt.xlabel("Boosting Iterations")
        plt.ylabel("Deviance")
        fig.tight_layout()
        plt.show()

    def shap_plot(self, X_test, key, style="beeswarm", max_display=15, figapp="", savefig=True, loadmodel=False,
                  output=False):
        if loadmodel:
            isValid = self.load_model(key)
            if not isValid:
                print("Cannot load " + self.mname_header + key + " model!")
                return
        mname = key + self.keyapps[0]
        model = self.models[mname]

        Xdf = pd.DataFrame(X_test, columns=self.labels)
        explainer = shap.Explainer(model, Xdf)
        shap_values = explainer(Xdf, check_additivity=False)

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams.update({'figure.autolayout': True})
        color_bar = True
        figsize = (6, 4)
        if style.lower() == "bar":
            ax = shap.plots.bar(shap_values, max_display=max_display, show=False)
        elif style.lower() == "heatmap":
            ax = shap.plots.heatmap(shap_values, max_display=max_display, show=False)
        elif style.lower() == "waterfall":
            ax = shap.plots.waterfall(shap_values[0], max_display=str(max_display), show=False)
        else:
            ax = shap.plots.beeswarm(shap_values, max_display=max_display, plot_size=figsize,
                                     color_bar_label='', color_bar=color_bar, show=False)

        fontsize = 14
        plt.setp(ax.get_yticklabels(), fontsize=fontsize)
        plt.setp(ax.get_xticklabels(), fontsize=fontsize)
        ax.set_xlabel("SHAP value", fontsize=fontsize + 2)

        if output:
            nes = shap_values.data.shape[0]
            shap_X = np.hstack([shap_values.data, shap_values.values, shap_values.base_values.reshape([nes, 1])])

            dy_labels = []
            for i in range(len(self.labels)):
                dy_labels.append("DY_D" + self.labels[i])

            cols = self.labels + dy_labels + ["Y_base"]
            shap_df = pd.DataFrame(shap_X, columns=cols)
            shap_df.to_csv(self.mname_header + mname + "_SHAP.csv")

        if not color_bar:
            plt.colorbar().ax.tick_params(direction="in", labelsize=0)

        if savefig:
            outfile = self.outfig_header + key
            if isinstance(figapp, str) and len(figapp) > 0:
                outfile += "_" + figapp
            outfile += "_shap.png"
            outfile = os.path.join(self.SAVE_PATH, outfile)
            plt.savefig(outfile, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def find_importance(self, X_test, y_test, key,
                        n_repeats=10, random_state=None, n_jobs=1,
                        Vertical=False, savefig=False, figapp=None):

        mname = key + self.keyapps[0]
        model = self.models[mname]
        rank = model.feature_importances_

        if self.Importance_mode is None:
            return rank
        else:
            nsubplot = len(self.Importance_mode)
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = ["Times New Roman"]
            fontsize = 16

            fig = plt.figure(figsize=(6 * nsubplot, 6))
            for iplot in range(len(self.Importance_mode)):
                plotid = self.Importance_mode[iplot]
                ax = plt.subplot(1, nsubplot, iplot + 1)
                if plotid == 0:
                    sorted_idx = np.argsort(rank)
                    pos = np.arange(sorted_idx.shape[0]) + 0.5
                    if Vertical:
                        plt.bar(pos, rank[sorted_idx], align="center")
                        plt.xticks(pos, np.array(self.labels)[sorted_idx])
                    else:
                        plt.barh(pos, rank[sorted_idx], align="center")
                        plt.yticks(pos, np.array(self.labels)[sorted_idx])
                    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
                    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
                    #plt.title("Feature Importance(MDI)_" + key, fontsize=fontsize+2)
                else:
                    result = permutation_importance(
                        model, X_test, y_test,
                        n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
                    sorted_idx = result.importances_mean.argsort()
                    plt.boxplot(
                        result.importances[sorted_idx].T,
                        vert=Vertical,
                        labels=np.array(self.labels)[sorted_idx],
                    )
                    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
                    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
                    #plt.title("Permutation Importance_" + key, fontsize=fontsize+2)
            fig.tight_layout()

            if savefig:
                outfile = self.outfig_header + key + self.keyapps[0]
                if figapp is None:
                    outfile = outfile + ".png"
                else:
                    outfile = outfile + "_" + figapp + ".png"
                outfile = os.path.join(self.SAVE_PATH, outfile)
                plt.savefig(outfile, bbox_inches='tight')
                plt.close()
                fig = None
            else:
                plt.show()
        return rank

    def pdp_plot_settings(self, features):
        subfeatures = []
        nfeature = len(features)
        finfo = copy.deepcopy(features)
        if nfeature % 2 == 1 and nfeature > 3 and nfeature != 9:
            finfo = np.append(finfo, finfo[0])
            nfeature += 1
        if nfeature <= 3:
            ncols = [1]
            subfeatures.append(finfo)
        elif nfeature % 2 == 0 and nfeature < 13:
            if nfeature >= 12:
                ncols = [4]
            elif nfeature >= 10:
                ncols = [5]
            elif nfeature >= 8:
                ncols = [4]
            elif nfeature >= 6:
                ncols = [2]
            elif nfeature > 3:
                ncols = [2]
            subfeatures.append(finfo)
        else:
            if nfeature > 18:
                raise ValueError("Too many feature to plot!")
            elif nfeature == 18:
                ncols = [3, 3]
                iend = 9
            elif nfeature == 17:
                ncols = [3, 4]
                iend = 9
            elif nfeature == 16:
                ncols = [4, 4]
                iend = 8
            elif nfeature == 15:
                ncols = [3, 2]
                iend = 9
            elif nfeature == 14:
                ncols = [4, 2]
                iend = 8
            elif nfeature >= 13:
                ncols = [3, 2]
                iend = 9
            elif nfeature >= 11:
                ncols = [1, 4]
                iend = 3
            elif nfeature >= 9:
                ncols = [3]
                iend = nfeature
            elif nfeature >= 7:
                ncols = [1, 2]
                iend = 3
            elif nfeature >= 5:
                ncols = [1, 1]
                iend = 3

            if len(ncols) == 1:
                subfeatures.append(finfo)
            else:
                subfeatures.append(finfo[0:iend])
                subfeatures.append(finfo[iend:nfeature])

        nrows = []
        for i in range(len(subfeatures)):
            nsub = len(subfeatures[i])
            ncol = ncols[i]
            nrows.append(int(nsub / ncol))
        return subfeatures, ncols, nrows

    def PDP_settings(self, ifeatures):
        self.common_params = {
            "subsample": self.subsample,
            "n_jobs": 1,
            "grid_resolution": 20,
            "random_state": 0,
        }

        self.features_info = {
            # features of interest
            "features": ifeatures,
            "feature_names": self.labels,
            # type of partial dependence plot
            "kind": "average",
            # information regarding categorical features
            "categorical_features": None,
        }

    def find_partialdependence(self, X_train, key, savefig=False, figapp=None, PDP_output=False):
        mname = key + self.keyapps[0]
        model = self.models[mname]

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        fontsize = 14
        basesize = 2.67

        if self.feature_type == "default":
            line_kw = {"linewidth": 2, "color": "tab:blue"}
        else:
            if X_train.shape[0] < 10000:
                line_kw = {"linewidth": 0, "color": "tab:blue", "marker": "o", "markersize": 8}
            else:
                line_kw = {"linewidth": 2, "color": "tab:blue"}

        if self.nsplit == 1:
            features4plot = []
            features4plot.append(self.features4plot)
        else:
            features4plot = []
            features4plot.append(np.arange(1))
            features4plot.append(np.arange(1, len(self.features4plot)))

        cols = []
        Xdf = []
        for isplit in range(len(features4plot)):
            feature4plot = features4plot[isplit]
            subfeatures, ncolss, nrowss = self.pdp_plot_settings(feature4plot)
            for iplot in range(len(subfeatures)):
                subfeature = subfeatures[iplot]
                nrows = nrowss[iplot]
                ncols = ncolss[iplot]
                self.PDP_settings(subfeature)

                _, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(1.25 * basesize * ncols, basesize * nrows),
                                     constrained_layout=True)

                display = PartialDependenceDisplay.from_estimator(
                    model,
                    X_train,
                    **self.features_info,
                    line_kw=line_kw,
                    ax=ax,
                    **self.common_params,
                )
                if self.feature_type == "default":
                    #xmaxs = 0.8 * np.ones(len(self.features))
                    #xmins = 0.1 * np.ones(len(self.features))
                    xmaxs = np.max(X_train, axis=0)
                    xmins = np.min(X_train, axis=0)
                else:
                    xmaxs = np.max(X_train, axis=0)
                    xmins = np.min(X_train, axis=0)

                thisax = display.axes_
                thisax = thisax.reshape([nrows, ncols])
                for i in range(len(subfeature)):
                    irow = int(i / ncols)
                    icol = i - irow * ncols
                    ifeat = subfeature[i]
                    if "CONC" in self.features[ifeat].upper():
                        xmin = 0.1
                        xmax = xmaxs[ifeat]
                    else:
                        xmin = xmins[ifeat]
                        xmax = xmaxs[ifeat]
                    thisax[irow, icol].set_xlim(xmin, xmax)
                    thisax[irow, icol].set_xlabel(self.labels[ifeat], fontsize=fontsize + 2)
                    thisax[irow, icol].yaxis.label.set_visible(False)
                    plt.setp(thisax[irow, icol].get_yticklabels(), fontsize=fontsize)
                    plt.setp(thisax[irow, icol].get_xticklabels(), fontsize=fontsize)

                    if PDP_output:
                        cols.append(self.labels[ifeat])
                        cols.append("DY_D" + self.labels[ifeat])
                        thislines = thisax[irow, icol].get_lines()
                        xdata = thislines[0].get_xdata()
                        ydata = thislines[0].get_ydata()
                        Xdf.append(xdata)
                        Xdf.append(ydata)

                '''
                display.axes_
                _ = display.figure_.suptitle(
                    ("Partial dependence of " + key + " on each feature"),
                    fontsize=fontsize+4,
                    )
                '''
                if savefig:
                    outfile = self.outfig_header + key + "_PDP" + str(isplit) + "_" + str(iplot)
                    if figapp is None:
                        outfile = outfile + ".png"
                    else:
                        outfile = outfile + "_" + figapp + ".png"
                    outfile = os.path.join(self.SAVE_PATH, outfile)
                    plt.savefig(outfile, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
        if PDP_output:
            Xdf = np.array(Xdf)
            #print(f"cols:{cols} shape:{Xdf.shape}")
            thisdf = pd.DataFrame(Xdf.T, columns=cols)
            thisdf.to_csv(self.mname_header + mname + "_PDP.csv")

    def find_interactive_pdp(self, X_train, key, savefig=False, figapp=None):
        mname = key + self.keyapps[0]
        model = self.models[mname]

        subfeatures, ncolss, nrowss = self.pdp_plot_settings(self.inter_features)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        fontsize = 14
        basesize = 2.67
        if self.feature_type == "default":
            line_kw = {"linewidth": 2, "color": "tab:blue"}
        else:
            if X_train.shape[0] < 10000:
                line_kw = {"linewidth": 0, "color": "tab:blue", "marker": "o", "markersize": 8}
            else:
                line_kw = {"linewidth": 2, "color": "tab:blue"}

        for iplot in range(len(subfeatures)):
            subfeature = subfeatures[iplot]
            nrows = nrowss[iplot]
            ncols = ncolss[iplot]

            thisinter = []
            for ipair in subfeature:
                thisinter.append(self.inter_features[ipair])
            self.PDP_settings(thisinter)

            _, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(1.25 * basesize * ncols, basesize * nrows),
                                 constrained_layout=True)

            display = PartialDependenceDisplay.from_estimator(
                model,
                X_train,
                **self.features_info,
                line_kw=line_kw,
                ax=ax,
                **self.common_params,
            )

            if self.feature_type == "default":
                xmaxs = 0.8 * np.ones(len(self.features))
                xmins = 0.1 * np.ones(len(self.features))
            else:
                xmaxs = np.max(X_train, axis=0)
                xmins = np.min(X_train, axis=0)
            thisax = display.axes_
            thisax = thisax.reshape([nrows, ncols])
            for i in range(len(subfeature)):
                irow = int(i / ncols)
                icol = i - irow * ncols
                isub = subfeature[i]
                ix = self.inter_features[isub][0]
                iy = self.inter_features[isub][1]
                if "CONC" in self.features[ix].upper():
                    xmin = 0.1
                    xmax = xmaxs[ix]
                else:
                    xmin = xmins[ix]
                    xmax = xmaxs[ix]
                thisax[irow, icol].set_xlim(xmin, xmax)
                thisax[irow, icol].set_xlabel(self.labels[ix], fontsize=fontsize + 2)
                if "CONC" in self.features[iy].upper():
                    xmin = 0.1
                    xmax = xmaxs[iy]
                else:
                    xmin = xmins[iy]
                    xmax = xmaxs[iy]
                thisax[irow, icol].set_ylim(xmin, xmax)
                thisax[irow, icol].set_ylabel(self.labels[iy], fontsize=fontsize + 2)
                #thisax[irow, icol].yaxis.label.set_visible(True)
                plt.setp(thisax[irow, icol].get_yticklabels(), fontsize=fontsize)
                plt.setp(thisax[irow, icol].get_xticklabels(), fontsize=fontsize)

            '''
            display.axes_
            _ = display.figure_.suptitle(
                ("Interactive partial dependence of " + key),
                fontsize=fontsize+4,
                )
            '''
            if savefig:
                outfile = self.outfig_header + key + "_interPDP" + str(iplot)
                if figapp is None:
                    outfile = outfile + ".png"
                else:
                    outfile = outfile + "_" + figapp + ".png"
                outfile = os.path.join(self.SAVE_PATH, outfile)
                plt.savefig(outfile, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

    def get_regression_model(self, X_train, y_train, key, savemodel=False):
        mname = key + self.keyapps[0]
        if self.modelname == "HGBR":
            model = ensemble.HistGradientBoostingRegressor(**self.params)
            model.fit(X_train, y_train)
        elif self.modelname == "GBR":
            model = ensemble.GradientBoostingRegressor(**self.params)
            model.fit(X_train, y_train)
        else:
            model = MLPRegressor(**self.params).fit(X_train, y_train)

        self.models[mname] = model
        if savemodel: self.save_model(key)
        return model

    def regression_models(self, random_state=None, n_repeats=10,
                          loadmodel=False, savemodel=False, savefig=False,
                          plot_predict=False, colorkey=None, figapp="",
                          Save_Pred2Data=False, plot_predict_all=False, SHAP_data="TEST",
                          SHAP_output=False, PDP_output=False):

        if not os.path.isdir(os.path.join(os.getcwd(), self.SAVE_PATH)):
            os.mkdir(os.path.join(os.getcwd(), self.SAVE_PATH))

        for key in self.keys:
            X_train, X_test, y_train, y_test, X, y = self.generate_data(key, random_state=random_state)
            if loadmodel:
                isValid = self.load_model(key)
            else:
                isValid = False
            if isValid:
                self.get_mse_mae(X_test, y_test, key,
                                 plot_predict=plot_predict, savefig=savefig, colorkey=colorkey, App="_test")
            else:
                self.get_regression_model(X_train, y_train, key, savemodel=savemodel)
                self.get_mse_mae(X_test, y_test, key,
                                 plot_predict=plot_predict, savefig=savefig, colorkey=colorkey, App="_test")

            if self.Visualize_Results:
                self.visualize_results(X_test, y_test, key)

            if self.SHAP_Plot:
                if SHAP_data.upper() == "ALL":
                    self.shap_plot(X, key, style=self.SHAP_style, max_display=self.max_display,
                                   figapp=figapp, savefig=savefig, output=SHAP_output)
                elif SHAP_data.upper() == "TRAIN":
                    self.shap_plot(X_train, key, style=self.SHAP_style, max_display=self.max_display,
                                   figapp=figapp, savefig=savefig, output=SHAP_output)
                else:
                    self.shap_plot(X_test, key, style=self.SHAP_style, max_display=self.max_display,
                                   figapp=figapp, savefig=savefig, output=SHAP_output)

            if self.Importance_mode is None:
                pass
            else:
                self.find_importance(X_test, y_test, key,
                                     n_repeats=n_repeats, random_state=random_state,
                                     Vertical=self.Vertical, savefig=savefig, figapp=figapp)
            if self.Find_PDP:
                self.find_partialdependence(X_train, key, savefig=savefig, figapp=figapp, PDP_output=PDP_output)
            if self.Find_interPDP:
                self.find_interactive_pdp(X_train, key, savefig=savefig, figapp=figapp)
            if Save_Pred2Data or plot_predict_all:
                self.save_predictions2data(key,
                                           predictions2data=Save_Pred2Data, plot_predict_all=plot_predict_all,
                                           savefig=savefig, colorkey=colorkey)

    def kfold_crossvalidation(self, n_splits, n_repeats=1, style="KFold", random_state=None,
                              find_outliers=False, thres4outliers=3.0):
        def get_outliers_index(standardized_resids, threshold=3.0):
            local_inds = np.arange(len(standardized_resids), dtype="int")
            out_inds = np.compress(np.abs(standardized_resids) > threshold, local_inds)
            if len(out_inds) == 0:
                return np.array([], dtype="int")
            else:
                return out_inds

        if not os.path.isdir(os.path.join(os.getcwd(), self.SAVE_PATH)):
            os.mkdir(os.path.join(os.getcwd(), self.SAVE_PATH))
        kfold_key_keys = ["imodel", "score",
                          "imax_abs", "abs_error", "compstr_abs",
                          "imax_rabs", "relative_error", "compstr_rabs",
                          "imax_std", "standardized_error", "compstr_std",
                          "outliers"]
        kfold_keys = ["key", "R2_AVG_ALL", "R2_STDEV"]
        kfold_df = pd.DataFrame(columns=kfold_keys)
        keys = []
        R2s = []
        R2stds = []
        print(f"length of original data:{len(self.data.gooddf)}")
        for key in self.keys:
            X = self.generate_X()
            y = self.data.gooddf[key].to_numpy()

            if style[0:3].upper() == "STR":
                rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                y4split = np.ones(len(y))
            else:
                rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                y4split = copy.deepcopy(y)

            imodel = 0
            thisR2s = []
            df = pd.DataFrame(columns=kfold_key_keys)
            for train, test in rkf.split(X, y4split):
                X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
                #if len(y_test) < 40:
                #    nadd = 40-len(y_test)
                #    X_test = np.append(X_train[0:nadd,:], X_test, axis=0)
                #    y_test = np.append(y_train[0:nadd], y_test)
                model = self.get_regression_model(X_train, y_train, key, savemodel=False)
                thisscore = model.score(X_test, y_test)

                thisdict = {}
                for kfoldkey in kfold_key_keys:
                    thisdict[kfoldkey] = "NA"
                thisdict = {"imodel": imodel, "score": thisscore}

                if imodel % 20 == 0:
                    print_results = True
                else:
                    print_results = False

                if find_outliers:
                    preds = model.predict(X_test)
                    resids = y_test - preds
                    resid_std = np.std(resids)
                    relative_resids = resids / (y_test + VERY_SMALL_VALUE)
                    standardized_resids = resids / (resid_std + VERY_SMALL_VALUE)
                    errorss = [resids, relative_resids, standardized_resids]
                    for ierror in range(len(errorss)):
                        errors = copy.deepcopy(errorss[ierror])
                        imax = np.argmax(np.abs(errors))
                        maxerror = errors[imax]
                        yhat = y_test[imax]
                        pred = preds[imax]
                        imax_data = test[imax]
                        compstr = self.data.gooddf.iloc[imax_data]["Composition"]
                        if ierror == 0:
                            thisdict["imax_abs"] = imax_data
                            thisdict["abs_error"] = maxerror
                            thisdict["compstr_abs"] = compstr
                            if print_results:
                                print(f"---- max abs_error ----")
                        elif ierror == 1:
                            thisdict["imax_rabs"] = imax_data
                            thisdict["relative_error"] = maxerror
                            thisdict["compstr_rabs"] = compstr
                            if print_results:
                                print(f"---- max relative_error ----")
                        elif ierror == 2:
                            thisdict["imax_std"] = imax_data
                            thisdict["standardized_error"] = maxerror
                            thisdict["compstr_std"] = compstr
                            if print_results:
                                print(f"---- max standardized_error ----")
                        if print_results:
                            print(f"imax: {imax} max error: {maxerror}")
                            print(f"target:{yhat} prediction: {pred}")
                            print(f"ind_data:{imax_data} compstr:{compstr}")

                    local_outliers = get_outliers_index(standardized_resids, threshold=thres4outliers)
                    if len(local_outliers) > 0:
                        outliers = test[local_outliers]
                    else:
                        outliers = np.array([], dtype="int")
                    thisdict["outliers"] = str(tuple(outliers))
                    if print_results:
                        print(f"outliers:{outliers}")

                df.loc[len(df)] = thisdict
                thisR2s.append(thisscore)
                if print_results:
                    print(f"key:{key} xtrain:{X_train.shape} xtest:{X_test.shape} imodel:{imodel} score:{thisscore}")
                imodel += 1

            fname = "KFOLD_" + self.mname_header + key + ".csv"
            df.to_csv(os.path.join(self.SAVE_PATH, fname), index=False)

            thisR2s = np.array(thisR2s)
            thismean = np.mean(thisR2s)
            thisstd = np.std(thisR2s)
            keys.append(key)
            R2s.append(thismean)
            R2stds.append(thisstd)
            print(f"features:{self.features}")
            print(f"key:{key}  R2_AVG_ALL:{thismean} R2_stdev:{thisstd}")
            print(f"--- finished Kfold for {key}! ---")
        kfold_df["key"] = keys
        kfold_df["R2_AVG_ALL"] = R2s
        kfold_df["R2_STDEV"] = R2stds
        fname = "KFOLD_" + self.mname_header + "ALL.csv"
        kfold_df.to_csv(os.path.join(self.SAVE_PATH, fname), index=False)
        return kfold_df



