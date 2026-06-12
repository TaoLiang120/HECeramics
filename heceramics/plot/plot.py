import copy
import itertools

# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from pylab import *
from matplotlib.font_manager import FontProperties
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error


def f1(x, a0, a1):
    y = a0 + a1 * x
    return y


def f2(x, a0, a1, a2):
    y = a0 + a1 * np.exp(-a2 * x)
    return y


def get_ticks_labels(ys, nyticks):
    vmax = np.max(ys)
    vmin = np.min(ys)
    vticks = np.linspace(vmin, vmax, nyticks)
    if vmax - vmin >= 10:
        ticklabels = [f'{int(round(v))}' for v in vticks]
    elif vmax - vmin >= 2:
        ticklabels = [f'{v:1.1f}' for v in vticks]
    else:
        ticklabels = [f'{v:1.2f}' for v in vticks]
    return vticks, ticklabels


def plot_linear_xys(x, ys, style="scatter", plot_xx=False,
                    Xlabel=None, Ylabel=None, Title=None, compute_R2=True, Label_R2=False,
                    curvefit=False, cmap="jet", colors=None, ShowColorbar=False, Comp_Labels=None, Label_Comp=False,
                    fontsize=16, nxticks=None, nyticks=None):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    def init_performance_df(n):
        keys = ["iplot", "mse", "rmse", "mae", "rmae", "r2", "p0_fit", "p1_fit", "r2_fit"]
        a = np.zeros([n, len(keys)])
        performance_df = pd.DataFrame(a, columns=keys)
        performance_df["iplot"] = np.arange(n, dtype=int)
        performance_df = performance_df.set_index("iplot")
        return performance_df

    def init_performance_dict():
        keys = ["mse", "rmse", "mae", "rmae", "r2", "p0_fit", "p1_fit", "r2_fit"]
        performance_dict = {}
        for key in keys:
            performance_dict[key] = 0.0
        return performance_dict

    def assign_dict2df(iplot, thisdict, performance_df):
        performance_df.loc[iplot] = thisdict
        return performance_df

    ShowColor = True
    if colors is None:
        colors = x[0:len(x)]
        ShowColor = False
    vmin = np.min(colors)
    vmax = np.max(colors)

    if len(ys.shape) == 1: ys = np.array([ys])
    nsubplot = len(ys)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update({'figure.autolayout': True})

    fig = plt.figure()
    axs = [None] * nsubplot
    font = FontProperties(family='times new roman', weight="bold", size=fontsize + 2)
    norm = Normalize(vmin=vmin, vmax=vmax)
    _map = ScalarMappable(norm=norm, cmap=cmap)
    cs = _map.to_rgba(colors)

    Label_compstr = False
    if Label_Comp:
        if isinstance(Comp_Labels, list) or isinstance(Comp_Labels, np.ndarray):
            if len(Comp_Labels) == len(x): Label_compstr = True

    performance_df = init_performance_df(nsubplot)
    xmean = np.mean(x)
    for iplot in range(nsubplot):
        y = ys[iplot, :]
        if iplot == 0:
            axs[iplot] = plt.subplot(nsubplot, 1, iplot + 1)
        else:
            axs[iplot] = plt.subplot(nsubplot, 1, iplot + 1, sharex=axs[0])

        if plot_xx:
            tmpx = np.linspace(np.min(x), np.max(x), 20)
            axs[iplot].plot(tmpx, tmpx, linewidth=1.5, color="k", linestyle="dotted", ms=0.0)
        if ShowColor:
            if style == "line":
                axs[iplot].plot(x, y, linewidth=1.5, color="k", linestyle="-", ms=20.0, mfc=cs, mew=0.0)
            else:
                axs[iplot].scatter(x, y, s=30, marker="o", c=cs)
        else:
            if style == "line":
                axs[iplot].plot(x, y, linewidth=1.5, color="k", linestyle="-", ms=20.0, mfc=cs, mew=0.0)
            else:
                axs[iplot].scatter(x, y, s=30, marker="o", c='k')

        if Label_compstr:
            for i in range(len(x)):
                axs[iplot].text(x[i], y[i], Comp_Labels[i], fontsize=fontsize - 4)

        thisdict = init_performance_dict()
        R2 = 0.0
        if compute_R2:
            yr = y - x
            diff = x - xmean

            R2 = 1.0 - np.sum(yr * yr) / (np.sum(diff * diff) + 1.0e-20)
            if not curvefit and Label_R2:
                label = "R2 = " + str(round(R2, 4))
                labelx = np.min(x) + 0.6 * (np.max(x) - np.min(x))
                labely = np.min(y) + 0.4 * (np.max(y) - np.min(y))
                axs[iplot].text(labelx, labely, label, fontsize=fontsize)

        mse = mean_squared_error(x, y)
        mse = np.sqrt(mse)
        rmse = 100.0 * mse / xmean
        mae = mean_absolute_error(x, y)
        rmae = 100.0 * mae / xmean
        print(f"iplot: {iplot}  R2: {R2}")
        print(f"MSE on test set: {mse} Relative MSE: {rmse}")
        print(f"MAE on test set: {mae} Relative MAE: {rmae}")
        thisdict["iplot"] = iplot
        thisdict["mse"] = mse
        thisdict["rmse"] = rmse
        thisdict["mae"] = mae
        thisdict["rmae"] = rmae
        thisdict["r2"] = R2
        if curvefit:
            popt, pcov = curve_fit(eval('f1'), x, y)
            xnew = np.linspace(np.min(x), np.max(x), 20)
            ynew = eval("f1(xnew,*popt)")
            axs[iplot].plot(xnew, ynew, color="k", linestyle='dashed', linewidth=1.0)
            yr = eval("f1(x,*popt)")
            yr = y - yr
            ydiff = y - np.mean(y)
            r2 = 1.0 - np.sum(yr * yr) / np.sum(ydiff * ydiff)
            if Label_R2:
                label = "R2 = " + str(round(r2, 4))
                labelx = np.min(x) + 0.6 * (np.max(x) - np.min(x))
                labely = np.min(y) + 0.3 * (np.max(y) - np.min(y))
                axs[iplot].text(labelx, labely, label, fontsize=fontsize - 2)
            print(f"iplot:{iplot} popt:{popt} R2:{r2}")
            thisdict["p0_fit"] = popt[0]
            thisdict["p1_fit"] = popt[1]
            thisdict["r2_fit"] = r2

        performance_df = assign_dict2df(iplot, thisdict, performance_df)
        if iplot == nsubplot - 1:
            if Xlabel is not None: axs[iplot].set_xlabel(Xlabel, fontsize=fontsize + 2)
            if isinstance(nxticks, int):
                xticks, xlabels = get_ticks_labels(x, nxticks)
                axs[iplot].set_xticks(xticks, labels=xlabels)

        if isinstance(nyticks, int):
            yticks, ylabels = get_ticks_labels(y, nyticks)
            axs[iplot].set_yticks(yticks, labels=ylabels)

        if Ylabel is not None:
            if isinstance(Ylabel, list):
                axs[iplot].set_ylabel(Ylabel[iplot], fontsize=fontsize + 2)
            else:
                axs[iplot].set_ylabel(Ylabel, fontsize=fontsize + 2)

    plt.setp(axs[iplot].get_yticklabels(), fontsize=fontsize)
    plt.setp(axs[iplot].get_xticklabels(), fontsize=fontsize)
    if Title is not None: plt.title(Title, fontsize=fontsize + 4)
    if ShowColor and ShowColorbar:
        font = FontProperties(family='times new roman', size=fontsize)
        _map.set_array(colors)
        cbar = plt.colorbar(_map, ax=plt.gca())
        cbar.set_label("", rotation=-90, ha="left", va="center", fontproperties=font)
        cbar.ax.tick_params(labelsize=fontsize)
    return fig, performance_df


def plot_xss_yss_lines(xss, yss, style="scatter", labels=None, show_legend=True,
                       Label_compstr=False, Comp_Labels=None, ncol=None,
                       savefig=False, outfile="xss_yss.svg", format="svg"):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update({'figure.autolayout': True})
    fontsize = 14
    markers = ["s", "o", "d", "x", "*", "<", ">"]
    colors = ["r", "b", "g", "m", "k"]
    nsubplot = 1
    fig = plt.figure()
    axs = [None] * nsubplot
    iplot = 0
    nline = len(xss)
    lines = [None] * nline
    for iline in range(nline):
        x = copy.deepcopy(xss[iline])
        y = copy.deepcopy(yss[iline])
        imark = iline % len(markers)
        thismarker = markers[imark]
        icolor = iline % len(colors)
        thiscolor = colors[icolor]
        axs[iplot] = plt.subplot(nsubplot, 1, iplot + 1)
        if style == "scatter":
            lines[iline] = axs[iplot].scatter(x, y, s=30, marker=thismarker, c=thiscolor)
        else:
            lines[iline], = axs[iplot].plot(x, y, linewidth=1.5, color=thiscolor, linestyle="-", ms=20.0, mfc=thiscolor,
                                           mew=0.0)
        if Label_compstr:
            for i in range(len(x)):
                axs[iplot].text(x[i], y[i], Comp_Labels[iline][i], fontsize=fontsize - 4)

        plt.setp(axs[iplot].get_yticklabels(), fontsize=fontsize)
        plt.setp(axs[iplot].get_xticklabels(), fontsize=fontsize)

    if isinstance(xss, np.ndarray):
        ndata = xss.shape[1]
    else:
        ndata = 4

    if isinstance(ncol, int):
        pass
    else:
        if ndata > 1:
            ncol = 4
        else:
            ncol = 1

    if show_legend:
        if labels is None:
            labels = ["NA"] * len(lines)
        elif len(labels) != len(lines):
            labels = ["NA"] * len(lines)
        if style == "scatter":
            plt.legend(lines, labels, scatterpoints=1, ncol=ncol, fontsize=fontsize - 4)
        else:
            plt.legend(lines, labels, ncol=ncol, fontsize=fontsize - 4)


    if savefig:
        plt.savefig(outfile, bbox_inches='tight', format=format)
        plt.close()
    else:
        plt.show()


def bar_plot(xkeys, ys, horizontal=True, fontsize=14,
             figsize=(6,6), nyticks=4, rotation=90,
             savefig=False, outfile=None):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update({'figure.autolayout': True})
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        ax.barh(xkeys, ys)
        ax.set_xticks(np.linspace(round(np.min(ys),3), round(np.max(ys),3), nyticks))
        labels = ax.get_xticklabels()
        plt.setp(labels, fontsize=fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    else:
        ax.bar(xkeys, ys)
        ax.set_yticks(np.linspace(round(np.min(ys),3), round(np.max(ys),3), nyticks))
        labels = ax.get_yticklabels()
        plt.setp(labels, fontsize=fontsize)
        labels = ax.get_xticklabels()
        plt.setp(labels, fontsize=fontsize-2, rotation=rotation)

    if savefig:
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def get_3d_bar(xs, ys, zs, yticks=["AMO", "FCC", "MIX", "BCC"], isstr=False):
    yticks = list(yticks)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = ['r', 'g', 'b', 'y']
    markers = ['o', '^', '*', 'd']

    xss = []
    zss = []
    xlabels = []
    for i in range(len(yticks)):
        ykey = yticks[i]
        inds = np.arange(len(xs), dtype=int)
        inds = np.compress(ys == ykey, inds)
        thisx = xs[inds]
        thisz = zs[inds]
        xlabels.append(thisx)
        if isstr:
            thisx = np.arange(len(xlabels), dtype=int)
        print(f"i:{i} ykey:{ykey} length:{len(thisz)}")
        xss.append(thisx)
        zss.append(thisz)

    for c, k in zip(colors, yticks):
        ik = yticks.index(k)
        thisx = xss[ik]
        thisz = zss[ik]
        thisy = [ik] * len(thisx)
        thisxlabel = xlabels[ik]
        cs = [c] * len(thisx)
        m = markers[ik]

        ax.bar(thisxlabel, thisz, zs=ik, zdir='y', color=cs, alpha=0.8)

    if not isstr:
        xmax = np.max(xs)
        xmin = np.min(xs)
        if xmax - xmin > 10:
            ndec = 0
        elif xmax - xmin > 1:
            ndec = 1
        elif xmax - xmin > 0.1:
            ndec = 2
        else:
            ndec = 3
        xticks = np.linspace(xmin, xmax, 5)
        xticks = np.around(xticks, decimals=ndec)
        ax.set_xticks(xticks)
        plt.setp(ax.get_xticklabels(), fontsize=14)

    xmax = np.max(zs)
    xmin = np.min(zs)
    if xmax - xmin > 10:
        ndec = 0
    elif xmax - xmin > 1:
        ndec = 1
    elif xmax - xmin > 0.1:
        ndec = 2
    else:
        ndec = 3
    zticks = np.linspace(xmin, xmax, 5)
    zticks = np.around(zticks, decimals=ndec)
    ax.set_zticks(zticks)
    plt.setp(ax.get_zticklabels(), fontsize=14)
    #ax.set_yticklabels(yticks)
    plt.setp(ax.get_yticklabels(), visible=False)
    return fig


def quartiles_plot(df, xkey, ykey, xbins=None, nbin=9,
                   vert=True, showmeans=True, meanline=True, showfliers=False):
    fontsize = 16
    ys = df[ykey].to_numpy()
    xs = df[xkey].to_numpy()
    if xbins is None:
        xmin = np.min(xs)
        xmax = np.max(xs)
        xbins = np.arange(xmin, xmax, nbin + 1)
    else:
        nbin = len(xbins) + 1

    ymeans = []
    yboxes = []
    #lolims = []
    #uplims = []
    for i in range(1, len(xbins)):
        inds = []
        for ii in range(len(df)):
            x = xs[ii]
            if x >= xbins[i - 1] and x < xbins[i]:
                inds.append(ii)
        inds = np.array(inds).astype(int)
        thisys = ys[inds]
        yboxes.append(thisys)

        ymean = np.mean(thisys)
        ymeans.append(ymean)
        print(f"ibin:{i} xrange:{xbins[i - 1], xbins[i]} length y:{len(thisys)}")

    xbins = xbins[1:len(xbins)]
    ymeans = np.array(ymeans)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 1, 1)
    font = FontProperties(family='times new roman', size=fontsize)
    plt.boxplot(
        yboxes,
        vert=vert,
        labels=xbins,
        showmeans=showmeans,
        meanline=meanline,
        showfliers=showfliers,
    )

    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)

    return fig
