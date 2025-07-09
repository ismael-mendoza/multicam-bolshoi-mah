#!/usr/bin/env python3
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict

import click
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d

from multicam import plotting as rxplots
from multicam.correlations import (
    add_box_indices,
    get_2d_corr,
    get_opt_corr,
    spearmanr,
    vol_jacknife_err,
)
from multicam.cosmo import get_a_from_t, get_fractional_tdyn, get_t_from_a
from multicam.fits import alpha_analysis, get_early_late
from multicam.mah import get_an_from_am, get_mah
from multicam.models import opcam_dict, prepare_datasets, training_suite
from multicam.plotting import CB_COLORS, LATEX_PARAMS, MARKS, set_rc

plt.ioff()

# directories
ROOT = Path(__file__).absolute().parent.parent
FIGS_DIR = ROOT.joinpath("results", "figs")
CACHE_DIR = ROOT.joinpath("results", "figs", "cache")
MAH_DIR = ROOT.joinpath("data", "processed", "bolshoi_m12")

# create if necessary
FIGS_DIR.mkdir(exist_ok=True, parents=False)
CACHE_DIR.mkdir(exist_ok=True, parents=False)

rho_latex = r"\rho_{\rm sp}"


class Figure(ABC):
    cache_name = ""
    params = ()
    bbox_inches = "tight"
    pad_inches = 0

    def __init__(self, overwrite=False, ext="png", rng=None) -> None:
        self.cache_file = CACHE_DIR.joinpath(self.cache_name).with_suffix(".npy")
        self.ext = "." + ext
        self.overwrite = overwrite
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    @abstractmethod
    def _set_rc(self):
        """Specify global plotting parameters."""

    @abstractmethod
    def get_data(self):
        """Return data as  for plotting."""

    @abstractmethod
    def get_figures(self, data: Dict[str, np.ndarray]) -> Dict[str, mpl.figure.Figure]:
        """Return dictionary with matplotlib figures."""

    def save(self):
        if self.overwrite or not self.cache_file.exists():
            data = self.get_data()
            np.save(self.cache_file, data)
        data = np.load(self.cache_file, allow_pickle=True)
        self._set_rc()
        figs = self.get_figures(data.item())
        for name, fig in figs.items():
            fig.savefig(
                FIGS_DIR.joinpath(name).with_suffix(self.ext),
                bbox_inches=self.bbox_inches,
                pad_inches=self.pad_inches,
            )


class CorrelationMAH(Figure):
    cache_name = "correlations_mah"
    params = (
        "cvir",
        "vmax/vvir",
        "voff/vvir",
        "x0",
        "t/|u|",
        "spin",
        "spin_bullock",
        "c_to_a",
        "b_to_a",
        "q",
        "r200m/rvir",
        "r500c/rvir",
    )
    fig_params = ("cvir", "vmax/vvir", "x0", "t/|u|", "spin_bullock", "c_to_a")
    # add lambda peebles, b/a, voff to vvir ratio, r500c to rvir ratio, r200m to rvir ratio
    lss = np.array(["-", ":"])  # pos vs neg correlations

    def _set_rc(self):
        set_rc(figsize=(7, 7), fontsize=24, lgsize="small", lgloc="best")

    def get_data(self):
        mah_data = get_mah(MAH_DIR, cutoff_missing=0.05, cutoff_particle=0.05)

        # load data
        cat = mah_data["cat"]
        scales = mah_data["scales"]
        am = mah_data["am"]
        ma = mah_data["ma_peak"]
        mass_bins = mah_data["mass_bins"]
        add_box_indices(cat)

        # remove last index to avoid correlation warning.
        ma = ma[:, :-1]
        scales = scales[:-1]
        am = am[:, :-1]
        mass_bins = mass_bins[:-1]

        tdyn = np.mean(cat["tdyn"]) / 10**9  # Gyr which astropy also returns by default
        ibox = cat["ibox"]
        ma_data = {}
        am_data = {}
        ma_max_dict = {}
        am_max_dict = {}
        for param in self.params:
            pvalue = cat[param]
            ma_corr, ma_err = get_2d_corr(ma, pvalue, ibox)
            ma_data[param] = (ma_corr, ma_err)
            max_indx, opt_scale, opt_err_scale = get_opt_corr(ma, pvalue, scales, ibox)
            opt_err_scale = max(opt_err_scale, scales[max_indx] - scales[max_indx - 1])
            ma_max_dict[param] = opt_scale, opt_err_scale, ma_corr[max_indx], ma_err[max_indx]

            # am
            am_corr, am_err = get_2d_corr(am, pvalue, ibox)
            am_data[param] = am_corr, am_err
            max_indx, opt_mbin, opt_err_mbin = get_opt_corr(am, pvalue, mass_bins, ibox)
            opt_err_mbin = max(opt_err_mbin, mass_bins[max_indx] - mass_bins[max_indx - 1])
            am_max_dict[param] = opt_mbin, opt_err_mbin, am_corr[max_indx], am_err[max_indx]

        return {
            "tdyn": tdyn,
            "ma_data": ma_data,
            "am_data": am_data,
            "ma_max_dict": ma_max_dict,
            "am_max_dict": am_max_dict,
            "scales": scales,
            "mass_bins": mass_bins,
        }

    def get_latex_table(self, data):
        table = (
            r"\begin{table*}[ht]" + "\n"
            r"\centering" + "\n"
            r"\begin{tabular}{|c|c|c|c|c|}" + "\n"
            r"\hline" + "\n"
            rf"$X$ & $a_{{\rm opt}}$ & ${rho_latex}\left(X, m(a_{{\rm opt}})\right)$"
            rf" & $m_{{\rm opt}}$ & ${rho_latex}\left(X, a(m_{{\rm opt}})\right)$ \\ [0.5ex]" + "\n"
            r"\hline\hline" + "\n"
        )
        for param in self.params:
            latex_param = rxplots.LATEX_PARAMS[param]
            scale, err_scale, val_ma, err_ma = data["ma_max_dict"][param]
            mass_bin, err_mbin, val_am, err_am = data["am_max_dict"][param]
            table += rf"{latex_param} & ${scale:.3f} \pm {err_scale:.3f}$ & ${val_ma:.3f} \pm {err_ma:.3f}$"
            table += rf" & ${mass_bin:.3f} \pm {err_mbin:.3f}$ & ${val_am:.3f} \pm {err_am:.3f}$\\ \hline"
            table += "\n"

        table += r"\end{tabular}" + "\n" + r"\caption{}" + "\n" + r"\end{table*}"

        with open(FIGS_DIR.joinpath("max_corrs_table.txt"), "w") as fp:
            print(table.strip(), file=fp)

    def get_ma_figure(self, data):
        """Get correlations with m(a) figure"""
        scales = data["scales"]
        tdyn = data["tdyn"]
        ma_data = data["ma_data"]
        max_dict = data["ma_max_dict"]

        fig, ax = plt.subplots(1, 1)

        for j, param in enumerate(self.fig_params):
            corr, err = ma_data[param]
            latex_param = rxplots.LATEX_PARAMS[param]
            color = CB_COLORS[j]
            pos = corr > 0
            neg = ~pos
            _corr = abs(corr)

            # plot positive corr and negative corr with different markers.
            if sum(pos) > 0:
                label = f"${latex_param}$" if sum(pos) > sum(neg) else None
                ax.plot(scales[pos], _corr[pos], color=color, ls=self.lss[0], label=label)

            if sum(neg) > 0:
                label = f"${latex_param}$" if sum(pos) < sum(neg) else None
                ax.plot(scales[neg], _corr[neg], color=color, ls=self.lss[1], label=label)

            ax.fill_between(scales, _corr - err, _corr + err, color=color, alpha=0.3)

        text = ""
        for j, param in enumerate(self.params):
            scale, err_scale, corr, err = max_dict[param]
            text += f"{param}: Max corr is {corr:.3f} +- {err:.3f} at scale {scale:.3f} +- {err_scale:.5f}\n"
            if param == "vmax/vvir":
                assert j == 1
                ax.axvline(scale, color=CB_COLORS[1], ls="--")
                ax.text(scale + 0.01, abs(corr) + 0.05, r"$a_{\rm opt}$", color=CB_COLORS[1])

        # additional saving of max correlations for table
        with open(FIGS_DIR.joinpath("max_corrs_ma.txt"), "w") as fp:
            print(text.strip(), file=fp)

        ax.set_ylim(0, 0.8)
        ax.set_xlim(0, 1.0)
        ax.set_ylabel(rf"${rho_latex}\left(X, m(a)\right)$")
        ax.set_xlabel(r"\rm Expansion factor $a=1/(1+z)$")

        # add additional x-axis with tydn fractional scale
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ax.get_xticks())

        xticks = ax.get_xticks()[1:]
        fractional_tdyn = get_fractional_tdyn(xticks, tdyn, sim_name="Bolshoi")
        fractional_tdyn = [f"${x/10**9:.2g}$" for x in fractional_tdyn]
        ax2.set_xticklabels([np.nan] + fractional_tdyn)
        ax2.set_xlabel(r"$ \Delta t / t_{\rm dyn}$", labelpad=10)

        ax.set_xlim(0.15, 1)
        ax2.set_xlim(0.15, 1)
        ax2.grid(None)
        return fig

    def get_am_figure(self, data):
        """Get correlations with a(m) figure"""
        mass_bins = data["mass_bins"]
        am_data = data["am_data"]
        max_dict = data["am_max_dict"]

        fig, ax = plt.subplots(1, 1)

        for j, param in enumerate(self.fig_params):
            corr, err = am_data[param]
            latex_param = rxplots.LATEX_PARAMS[param]
            color = CB_COLORS[j]
            pos = corr >= 0
            neg = ~pos
            _corr = abs(corr)

            # plot positive corr and negative corr with different markers.
            if sum(pos) > 0:
                label = f"${latex_param}$" if sum(pos) > sum(neg) else None
                ax.plot(mass_bins[pos], _corr[pos], color=color, ls=self.lss[0], label=label)

            if sum(neg) > 0:
                label = f"${latex_param}$" if sum(pos) < sum(neg) else None
                ax.plot(mass_bins[neg], _corr[neg], color=color, ls=self.lss[1], label=label)

            ax.fill_between(mass_bins, _corr - err, _corr + err, alpha=0.3)

        # draw a vertical line at max scales, output table.
        text = ""
        for j, param in enumerate(self.params):
            mbin, mbin_err, corr, err = max_dict[param]
            text += f"{param}: Max corr is {corr:.3f} +- {err:.3f} at mass bin {mbin:.3f} +- {mbin_err:.5f}\n"

        with open(FIGS_DIR.joinpath("max_corrs_am.txt"), "w") as fp:
            print(text.strip(), file=fp)

        ax.set_ylim(0, 0.8)
        ax.set_xlim(0.01, 1.0)
        ax.set_ylabel(rf"${rho_latex}\left(X, a(m)\right)$")
        ax.set_xlabel(r"\rm Mass fraction $m=M/M(z=0)$")
        ax.tick_params(axis="both", which="major")
        ax.tick_params(axis="x", which="minor")

        # add additional x-axis so figures are aligned in paper
        ax2 = ax.twiny()
        ax2.set_xlabel(r"$ \Delta t / t_{\rm dyn}$", labelpad=10)
        ax2.set_xlim(0.01, 1.0)
        ax2.set_xlabel(r"$m=M/M(z=0)$")
        ax2.tick_params(axis="x", which="minor")

        ax.legend(loc="best")
        return fig

    def get_figures(self, data: Dict[str, np.ndarray]) -> Dict[str, mpl.figure.Figure]:
        self.get_latex_table(data)
        return {"ma_corr": self.get_ma_figure(data), "am_corr": self.get_am_figure(data)}


class TriangleSamples(Figure):
    cache_name = "triangle"
    params = ("cvir", "t/|u|", "x0", "spin_bullock", "c_to_a")
    which_log = [True, True, True, True, False]
    subset_params = [2, 3, 4]
    bbox_inches = "tight"
    pad_inches = 0.05

    def _set_rc(self):
        set_rc(fontsize=24)
        mpl.rcParams.update({"axes.grid": False, "figure.autolayout": False})

    def get_data(self):
        mah_data = get_mah(MAH_DIR, cutoff_missing=0.05, cutoff_particle=0.05)

        cat = mah_data["cat"]
        am = mah_data["am"]
        mass_bins = mah_data["mass_bins"]

        # prepare catalog with all a_m
        am_names = [f"am_{ii}" for ii in range(len(mass_bins))]
        for ii in range(len(mass_bins)):
            cat.add_column(am[:, ii], name=am_names[ii])

        # dataset preparation
        info = {"all": {"x": am_names, "y": self.params}}
        datasets, _, _ = prepare_datasets(cat, info, self.rng)

        # models to use
        n_targets = len(self.params)
        data = {
            "multicam": {
                "xy": datasets["all"]["train"],
                "n_features": 100,
                "n_targets": n_targets,
                "model": "gaussian",
                "kwargs": {"rng": self.rng},
            },
            "optcam": {
                "xy": datasets["all"]["train"],
                "n_features": 100,
                "n_targets": n_targets,
                "model": "mixed_cam",
                "kwargs": {
                    "mass_bins": mass_bins,
                    "opt_mbins": [opcam_dict[param]["mbin"] for param in self.params],
                    "cam_orders": [opcam_dict[param]["order"] for param in self.params],
                },
            },
        }
        joint_models = training_suite(data)

        x_test = datasets["all"]["test"][0]
        samples_multigauss = joint_models["multicam"].sample(x_test).reshape(-1, n_targets)
        samples_linear = joint_models["multicam"].predict(x_test).reshape(-1, n_targets)
        samples_cam = joint_models["optcam"].predict(x_test).reshape(-1, n_targets)

        return {
            "truth": datasets["all"]["test"][1],
            "multigauss": samples_multigauss,
            "lr": samples_linear,
            "cam": samples_cam,
        }

    def _transform(self, y):
        y_new = np.zeros_like(y)
        for ii in range(y.shape[1]):
            y_new[:, ii] = np.log10(y[:, ii]) if self.which_log[ii] else y[:, ii]
        return y_new

    def get_latex_table(self, corrs):
        nice_names = {
            "truth": r"\rm True",
            "cam": r"\rm CAM $a(m_{\rm opt})$",
            "lr": r"\rm MultiCAM (no scatter)",
            "multigauss": r"\rm MultiCAM (with scatter)",
        }
        latex_name_tuples = [
            rf"{rxplots.LATEX_PARAMS[p1]},\,{rxplots.LATEX_PARAMS[p2]}"
            for (p1, p2) in corrs["truth"]
        ]
        name_row = "Model & " + " & ".join(latex_name_tuples)
        table = (
            r"\begin{table*}[ht]" + "\n"
            r"\centering" + "\n"
            r"\begin{tabular}{|c|c|c|c|}" + "\n"
            r"\hline" + "\n"
            rf"{name_row} \\ [0.5ex]" + "\n"
            r"\hline\hline" + "\n"
        )

        for model in nice_names:
            latex_model = nice_names[model]
            corr_values = corrs[model].values()
            corr_row = f"{latex_model} & " + " & ".join([f"{corr:.2f}" for corr in corr_values])
            table += rf"{corr_row} \\" + "\n"
            table += r"\hline" + "\n"

        table += r"\end{tabular}" + "\n" + r"\caption{}" + "\n" + r"\end{table*}"

        with open(FIGS_DIR.joinpath("triangle_corrs_table.txt"), "w") as fp:
            print(table.strip(), file=fp)

    def get_figures(self, data: Dict[str, np.ndarray]) -> Dict[str, mpl.figure.Figure]:
        figs = {}

        # (1) multicam gaussian samples on all params.
        labels = [rxplots.LATEX_PARAMS[param] for param in self.params]
        labels = [
            rf"$\log_{{10}} \left({label[1:-1]}\right)$" if self.which_log[ii] else label
            for ii, label in enumerate(labels)
        ]
        y_true = data.pop("truth")
        y1 = self._transform(y_true)
        y2 = self._transform(data["multigauss"])
        levels = [1 - np.exp(-(x**2) / 2) for x in [1, 2, 3]]
        ranges = [(0.3, 1.4), (-0.40, 0), (-2.7, -0.3), (-2.5, -0.4), (0.15, 1.00)]
        # ranges = [0.999, 0.999, 0.999, 0.999, 0.999]
        n_bins = 15
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        fig = corner.corner(
            y1,
            labels=labels,
            max_n_ticks=3,
            color=CB_COLORS[1],
            plot_datapoints=False,
            levels=levels,
            range=ranges,
            bins=n_bins,
            labelpad=0.125,
            fig=fig,
        )
        fig = corner.corner(
            y2,
            max_n_ticks=3,
            color=CB_COLORS[2],
            plot_datapoints=False,
            levels=levels,
            range=ranges,
            bins=n_bins,
            fig=fig,
        )
        axes[0, 0].annotate(
            r"\rm \textbf{True samples}",
            xy=(0.275, 0.925),
            xycoords="figure fraction",
            color=CB_COLORS[1],
            fontsize=18,
        )
        axes[0, 0].annotate(
            "\\rm \\textbf{MultiCAM} \n \\rm \\textbf{(with scatter)}",
            xy=(0.275, 0.87),
            xycoords="figure fraction",
            color=CB_COLORS[2],
            fontsize=18,
        )
        fig.subplots_adjust(wspace=0.15, hspace=0.15)
        figs["multigauss_triangle"] = fig

        # (2) Now a subset set of 3 triangle plots without histograms.
        ndim = len(self.subset_params)
        annotations = {
            "cam": r"\rm \textbf{CAM} $\boldsymbol{a(m_{\rm opt})}$",
            "lr": "\\rm \\textbf{MultiCAM} \n \\rm \\textbf{(no scatter)}",
            "multigauss": "\\rm \\textbf{MultiCAM} \n \\rm \\textbf{(with scatter)}",
        }
        for name, yest in data.items():
            _ranges = [(-2.7, -0.3), (-2.5, -0.4), (0.15, 1.00)]
            y2 = self._transform(yest)
            _y1 = y1[:, self.subset_params]
            _y2 = y2[:, self.subset_params]
            _labels = [labels[ii] for ii in self.subset_params]
            fig = corner.corner(
                _y1,
                color=CB_COLORS[1],
                max_n_ticks=3,
                plot_datapoints=False,
                labels=_labels,
                labelpad=0.1,
                range=_ranges,
                bins=n_bins,
                levels=levels,
            )
            fig = corner.corner(
                _y2,
                fig=fig,
                color=CB_COLORS[2],
                max_n_ticks=3,
                plot_datapoints=False,
                bins=n_bins,
                range=_ranges,
                levels=levels,
            )

            # remove histograms
            axes = np.array(fig.axes).reshape((ndim, ndim))
            if name == "cam":
                axes[1, 0].annotate(
                    r"\rm \textbf{True samples}",
                    xy=(0.425, 0.56),
                    xycoords="figure fraction",
                    color=CB_COLORS[1],
                )
            axes[1, 0].annotate(
                annotations[name], xy=(0.425, 0.50), xycoords="figure fraction", color=CB_COLORS[2]
            )
            # axes[1, 0].annotate(annotations[name], xy=(0.3, 0.68), xycoords="figure fraction")
            for ii in range(ndim):
                ax: plt.Axes = axes[ii, ii]
                ax.set_xmargin(0)
                ax.set_ymargin(0)
                ax.set_axes_locator(plt.NullLocator())
                ax.set_axis_off()
                ax.remove()
            fig.subplots_adjust(wspace=0.2, hspace=0.2)
            figs[f"{name}_subset_triangle"] = fig

        # (3) Print out correlations explicitly so that they can be reported in a table.
        data["truth"] = y_true  # now we put it back since we popped it
        corrs = {k: {} for k in data}
        for model in data:
            for p1 in self.subset_params:
                for p2 in self.subset_params:
                    param1 = self.params[p1]
                    param2 = self.params[p2]
                    if param1 == param2 or (param2, param1) in corrs[model]:
                        continue
                    idx1 = self.params.index(param1)
                    idx2 = self.params.index(param2)
                    y = data[model]
                    corrs[model][(param1, param2)] = spearmanr(y[:, idx1], y[:, idx2])
        self.get_latex_table(corrs)
        return figs


class PredictMAH(Figure):
    cache_name = "predict_mah"
    params = ("cvir", "t/|u|", "x0", "spin_bullock", "c_to_a")

    def _set_rc(self):
        set_rc(fontsize=28, lgsize=22, figsize=(8, 8))

    def get_data(self):
        mah_data = get_mah(MAH_DIR, cutoff_missing=0.05, cutoff_particle=0.05)
        cat = mah_data["cat"]
        ma = mah_data["ma_peak"]
        am = mah_data["am"]
        mass_bins = mah_data["mass_bins"][:-1]  # remove last bin to avoid spearman error.
        scales = mah_data["scales"][:-1]  # same for scales.
        n_mbins = len(mass_bins)
        n_scales = len(scales)
        add_box_indices(cat)

        # prepare catalog with all m_a
        ma_names = [f"ma_{ii}" for ii in range(len(scales))]
        for ii in range(len(scales)):
            cat.add_column(ma[:, ii], name=ma_names[ii])

        # prepare catalog with all a(m)
        am_names = [f"am_{ii}" for ii in range(len(mass_bins))]
        for ii in range(len(mass_bins)):
            cat.add_column(am[:, ii], name=am_names[ii])

        # prepare datasets
        info = {
            "cvir_only": {"x": ("cvir",), "y": am_names + ma_names},
            "x0_only": {"x": ("x0",), "y": am_names + ma_names},
            "tu_only": {"x": ("t/|u|",), "y": am_names + ma_names},
            "all": {"x": self.params, "y": am_names + ma_names},
        }
        datasets, _, test_idx = prepare_datasets(cat, info, self.rng)

        # train models
        data = {
            "linear_cvir": {
                "xy": datasets["cvir_only"]["train"],
                "n_features": 1,
                "n_targets": n_mbins + n_scales,
                "model": "linear",
                "kwargs": {},
            },
            "linear_x0": {
                "xy": datasets["x0_only"]["train"],
                "n_features": 1,
                "n_targets": n_mbins + n_scales,
                "model": "linear",
                "kwargs": {},
            },
            "linear_tu": {
                "xy": datasets["tu_only"]["train"],
                "n_features": 1,
                "n_targets": n_mbins + n_scales,
                "model": "linear",
                "kwargs": {},
            },
            "linear_all": {
                "xy": datasets["all"]["train"],
                "n_features": len(self.params),
                "n_targets": n_mbins + n_scales,
                "model": "linear",
                "kwargs": {},
            },
        }
        models = training_suite(data)

        corrs_am = defaultdict(lambda: np.zeros(n_mbins))
        errs_am = defaultdict(lambda: np.zeros(n_mbins))
        corrs_ma = defaultdict(lambda: np.zeros(n_scales))
        errs_ma = defaultdict(lambda: np.zeros(n_scales))
        dataset_names = ["cvir_only", "x0_only", "tu_only", "all"]
        mdl_names = ["linear_cvir", "linear_x0", "linear_tu", "linear_all"]
        ibox = cat[test_idx]["ibox"]  # need it for errors.
        for dataset_names, mdl_name in zip(dataset_names, mdl_names):
            model = models[mdl_name]
            x_test, y_test = datasets[dataset_names]["test"]
            y_pred = model.predict(x_test)

            for jj in range(n_mbins):
                y1 = y_pred[:, jj]
                y2 = y_test[:, jj]
                corrs_am[mdl_name][jj] = spearmanr(y1, y2)
                errs_am[mdl_name][jj] = vol_jacknife_err(spearmanr, ibox, y1, y2)

            for jj in range(n_scales):
                y1 = y_pred[:, jj + n_mbins]
                y2 = y_test[:, jj + n_mbins]
                corrs_ma[mdl_name][jj] = spearmanr(y1, y2)
                errs_ma[mdl_name][jj] = vol_jacknife_err(spearmanr, ibox, y1, y2)

        return {
            "corrs_am": dict(corrs_am),
            "errs_am": dict(errs_am),
            "corrs_ma": dict(corrs_ma),
            "errs_ma": dict(errs_ma),
            "mass_bins": mass_bins,
            "scales": scales,
        }

    def get_figures(self, data: Dict[str, np.ndarray]) -> Dict[str, mpl.figure.Figure]:
        corrs_am, errs_am, corrs_ma, errs_ma, mass_bins, scales = data.values()
        mdl_names = list(corrs_am.keys())
        nice_names = [
            r"\rm $c_{\rm vir}$ only (CAM)",
            r"\rm $x_{\rm off}$ only (CAM)",
            r"\rm $T / |U|$ only (CAM)",
            r"\rm All properties",
        ]

        # (1) Correlation m(a) vs m_pred(a) figure
        fig1, ax = plt.subplots(1, 1)
        for jj, (nice_name, mdl_name) in enumerate(zip(nice_names, mdl_names)):
            corr = corrs_ma[mdl_name]
            err = errs_ma[mdl_name]
            ax.plot(scales, corr, label=nice_name, color=CB_COLORS[jj])
            ax.fill_between(scales, corr - err, corr + err, color=CB_COLORS[jj], alpha=0.5)
        ax.set_xlabel("$a$")
        ax.set_ylabel(rf"${rho_latex}\left(m(a), m_{{\rm pred}}(a)\right)$")
        ax.set_yticks(np.arange(0.0, 0.9, 0.1))
        ax.set_xticks(np.arange(0.2, 1.1, 0.1))
        ax.set_ylim(0.0, 0.8)
        ax.set_xlim(0.2, 1.0)

        # (2) Correlation a(m) vs a_pred(m) figure
        fig2, ax = plt.subplots(1, 1)
        for jj, (nice_name, mdl_name) in enumerate(zip(nice_names, mdl_names)):
            corr = corrs_am[mdl_name]
            err = errs_am[mdl_name]
            ax.plot(mass_bins, corr, label=nice_name, color=CB_COLORS[jj])
            ax.fill_between(mass_bins, corr - err, corr + err, color=CB_COLORS[jj], alpha=0.5)
        ax.set_xlabel("$m$")
        ax.set_ylabel(rf"${rho_latex}\left(a(m), a_{{\rm pred}}(m)\right)$")
        ax.set_yticks(np.arange(0.0, 0.9, 0.1))
        ax.set_ylim(0.0, 0.8)
        ax.set_xticks(np.arange(0.0, 1.1, 0.1))
        ax.set_xlim(0.0, 1.0)
        ax.set_yticklabels(f"${x:.1f}$" for x in ax.get_yticks())
        ax.legend(loc="best")

        return {"corr_pred_mah_ma": fig1, "corr_pred_mah_am": fig2}


class InvPredMetrics(Figure):
    cache_name = "inv_pred_metrics"
    params = ("a2", "alpha", "mdyn", "tau_c", "alpha_late", "alpha_early")

    def _set_rc(self):
        set_rc(fontsize=28, lgsize=20, lgloc="lower left", figsize=(8, 8))

    def get_data(self):
        mah_data = get_mah(MAH_DIR, cutoff_missing=0.05, cutoff_particle=0.05)
        cat = mah_data["cat"]
        ma = mah_data["ma"]  # for alpha fit
        ma_peak = mah_data["ma_peak"]  # for m_tdyn
        am = mah_data["am"]
        scales = mah_data["scales"]
        mass_bins = mah_data["mass_bins"]
        add_box_indices(cat)

        # extract m(a_{tdyn}) = \dot{M}_dynamical
        t0 = get_t_from_a(1)
        tdyn = np.mean(cat["tdyn"])
        t = (t0 - tdyn) * u.Gyr
        a_dyn = get_a_from_t(t)
        indx_dyn = np.where(scales > a_dyn)[0][0]
        mdyn = ma_peak[:, indx_dyn].reshape(-1, 1)

        # extract alpha fits
        alpha_file = ROOT.joinpath("data", "processed", "alpha_fits.npy")
        alphas, _, _ = alpha_analysis(ma, scales, mass_bins, alpha_file=alpha_file)

        # extract a_{1/2} and a_{3/4}
        a2 = get_an_from_am(am, mass_bins, mbin=0.5)
        a4 = get_an_from_am(am, mass_bins, mbin=0.75)

        # Ddiffmah parameters
        pars = np.load(ROOT.joinpath("data", "processed", "pbest_diffmah.npy"))
        logtc, ue, ul = pars[:, 0], pars[:, 1], pars[:, 2]
        early, late = get_early_late(ue, ul)

        # add everything extracted to catalog
        cat.add_column(10**logtc, name="tau_c")
        cat.add_column(early, name="alpha_early")
        cat.add_column(late, name="alpha_late")
        cat.add_column(alphas, name="alpha")
        cat.add_column(a2, name="a2")
        cat.add_column(a4, name="a4")
        cat.add_column(mdyn, name="mdyn")

        x_params = ("cvir", "t/|u|", "x0", "spin_bullock", "c_to_a")
        n_params = len(self.params)
        info = {
            "cvir_only": {
                "x": ("cvir",),
                "y": self.params,
            },
            "x0_only": {
                "x": ("x0",),
                "y": self.params,
            },
            "tu_only": {
                "x": ("t/|u|",),
                "y": self.params,
            },
            "all": {
                "x": x_params,
                "y": self.params,
            },
        }
        datasets, _, test_idx = prepare_datasets(cat, info, self.rng)

        data = {
            "linear_cvir": {
                "xy": datasets["cvir_only"]["train"],
                "n_features": 1,
                "n_targets": len(self.params),
                "model": "linear",
                "kwargs": {},
            },
            "linear_x0": {
                "xy": datasets["x0_only"]["train"],
                "n_features": 1,
                "n_targets": len(self.params),
                "model": "linear",
                "kwargs": {},
            },
            "linear_tu": {
                "xy": datasets["tu_only"]["train"],
                "n_features": 1,
                "n_targets": len(self.params),
                "model": "linear",
                "kwargs": {},
            },
            "linear_all": {
                "xy": datasets["all"]["train"],
                "n_features": len(x_params),
                "n_targets": len(self.params),
                "model": "linear",
                "kwargs": {},
            },
        }
        models = training_suite(data)

        mdl_names = ["linear_cvir", "linear_x0", "linear_tu", "linear_all"]
        ds_names = ["cvir_only", "x0_only", "tu_only", "all"]
        nice_names = [
            r"\rm $c_{\rm vir}$ only (CAM)",
            r"\rm $x_{\rm off}$ only (CAM)",
            r"\rm $T/|U|$ only (CAM)",
            r"\rm All parameters",
        ]

        output = {}
        ibox = cat[test_idx]["ibox"]
        for ds, mdl in zip(ds_names, mdl_names):
            d = defaultdict(lambda: np.zeros(n_params))
            x_test, y_test = datasets[ds]["test"]
            y_est = models[mdl].predict(x_test)
            for jj in range(n_params):
                y1, y2 = y_test[:, jj], y_est[:, jj]
                d["val"][jj] = spearmanr(y1, y2)
                d["err"][jj] = vol_jacknife_err(spearmanr, ibox, y1, y2)
            output[mdl] = dict(d)

        return {
            "mdl_names": mdl_names,
            "nice_names": nice_names,
            "output": output,
        }

    def get_figures(self, data: Dict[str, np.ndarray]) -> Dict[str, mpl.figure.Figure]:
        fig, ax = plt.subplots(1, 1)
        mdl_names, nice_names, output = data.values()
        x_bias = -0.2
        for ii, (mdl, label) in enumerate(zip(mdl_names, nice_names)):
            m, c = MARKS[ii], CB_COLORS[ii]
            mval, merr = output[mdl]["val"], output[mdl]["err"]
            plot_params = [LATEX_PARAMS[param] for param in self.params]
            rxplots.metrics_plot(ax, mval, merr, plot_params, label, x_bias, m, c)
            x_bias += 0.1
        ax.set_ylim(-0.05, 0.80)
        ax.set_xlim(-0.5, len(self.params) - 0.5)
        ax.axvline(len(self.params) - 0.5 - 0.02, ymin=0.0, ymax=1.0, color="k", lw=1.3)
        ax.set_ylabel(rf"${rho_latex}\left(y_{{\rm pred}}, y_{{\rm true}}\right)$")
        ax.legend()
        return {"inv_pred_metrics": fig}


class ForwardPredMetrics(Figure):
    cache_name = "forward_pred_metrics"
    params = ("cvir", "t/|u|", "x0", "spin_bullock", "c_to_a")

    def _set_rc(self):
        return set_rc(figsize=(8, 8), fontsize=28, lgsize=16, lgloc="lower left")

    def get_data(self):
        mah_data = get_mah(MAH_DIR, cutoff_missing=0.05, cutoff_particle=0.05)
        cat = mah_data["cat"]
        ma_peak = mah_data["ma_peak"]  # for alpha fits and gradients.
        am = mah_data["am"]
        scales = mah_data["scales"]
        mass_bins = mah_data["mass_bins"]
        add_box_indices(cat)

        # prepare catalog with all m_a
        ma_names = [f"ma_{ii}" for ii in range(len(scales))]
        for ii in range(len(scales)):
            cat.add_column(ma_peak[:, ii], name=ma_names[ii])

        # prepare catalog with all a_m
        am_names = [f"am_{ii}" for ii in range(len(mass_bins))]
        for ii in range(len(mass_bins)):
            cat.add_column(am[:, ii], name=am_names[ii])

        # load alpha fits
        alpha_file = ROOT.joinpath("data", "processed", "alpha_fits.npy")
        alphas, _, _ = alpha_analysis(ma_peak, scales, mass_bins, alpha_file=alpha_file)
        cat.add_column(alphas, name="alpha")

        # load diffmah parameters of best fits.
        pars = np.load(ROOT.joinpath("data", "processed", "pbest_diffmah.npy"))
        logtc, ue, ul = pars[:, 0], pars[:, 1], pars[:, 2]
        early, late = get_early_late(ue, ul)
        cat.add_column(10**logtc, name="tau_c")
        cat.add_column(early, name="alpha_early")
        cat.add_column(late, name="alpha_late")

        # load diffmah fits curves
        # note that they are already mpeak by virtual of parametrization of diffmah
        diffmah_curves = np.load(ROOT.joinpath("data", "processed", "diffmah_fits.npy"))
        ma_diffmah_names = [f"ma_diffmah_{ii}" for ii in range(len(scales))]
        for ii in range(len(scales)):
            cat.add_column(diffmah_curves[:, ii], name=ma_diffmah_names[ii])

        # add a_{1/2} also as alternative parametrization
        cat.add_column(get_an_from_am(am, mass_bins, 0.5), name="a2")

        info = {
            "ma": {
                "x": ma_names,
                "y": self.params,
            },
            "am": {
                "x": am_names,
                "y": self.params,
            },
            "ma_diffmah": {
                "x": ma_diffmah_names,
                "y": self.params,
            },
            "params_diffmah": {
                "x": ("tau_c", "alpha_early", "alpha_late"),
                "y": self.params,
            },
            "alpha": {
                "x": ("alpha",),
                "y": self.params,
            },
        }
        datasets, _, test_idx = prepare_datasets(cat, info, self.rng)
        n_params = len(self.params)

        data = {
            "multicam_ma": {
                "xy": datasets["ma"]["train"],
                "n_features": 165,
                "n_targets": n_params,
                "model": "linear",
                "kwargs": {},
            },
            "multicam_ma_diffmah": {
                "xy": datasets["ma_diffmah"]["train"],
                "n_features": 165,
                "n_targets": n_params,
                "model": "linear",
                "kwargs": {},
            },
            "multicam_params_diffmah": {
                "xy": datasets["params_diffmah"]["train"],
                "n_features": 3,
                "n_targets": n_params,
                "model": "linear",
                "kwargs": {},
            },
            "optcam": {
                "xy": datasets["am"]["train"],
                "n_features": 100,
                "n_targets": n_params,
                "model": "mixed_cam",
                "kwargs": {
                    "mass_bins": mass_bins,
                    "opt_mbins": [opcam_dict[param]["mbin"] for param in self.params],
                    "cam_orders": [opcam_dict[param]["order"] for param in self.params],
                },
            },
        }

        models = training_suite(data)

        mdl_names = ("multicam_ma", "multicam_ma_diffmah", "multicam_params_diffmah", "optcam")
        ds_names = ("ma", "ma_diffmah", "params_diffmah", "am")

        output = {}
        ibox = cat[test_idx]["ibox"]
        for ds, mdl in zip(ds_names, mdl_names):
            d = defaultdict(lambda: np.zeros(n_params))
            x_test, y_test = datasets[ds]["test"]
            y_est = models[mdl].predict(x_test)
            for jj in range(n_params):
                y1, y2 = y_test[:, jj], y_est[:, jj]
                d["val"][jj] = spearmanr(y1, y2)
                d["err"][jj] = vol_jacknife_err(spearmanr, ibox, y1, y2)
            output[mdl] = dict(d)

        return {
            "mdl_names": mdl_names,
            "output": output,
        }

    def get_figures(self, data: Dict[str, np.ndarray]) -> Dict[str, mpl.figure.Figure]:
        nice_names = (
            r"\rm MultiCAM $m(a)$",
            r"\rm MultiCAM Diffmah $m(a)$ curves",
            r"\rm MultiCAM Diffmah parameters",
            r"\rm CAM $a(m_{\rm opt})$",
        )
        fig, ax = plt.subplots(1, 1)
        mdl_names, output = data.values()
        x_bias = -0.2
        for ii, (mdl, label) in enumerate(zip(mdl_names, nice_names)):
            m, c = MARKS[ii], CB_COLORS[ii]
            mval, merr = output[mdl]["val"], output[mdl]["err"]
            plot_params = [LATEX_PARAMS[param] for param in self.params]
            rxplots.metrics_plot(ax, mval, merr, plot_params, label, x_bias, m, c)
            x_bias += 0.1
        ax.set_ylim(0.3, 0.8)
        ax.axvline(len(self.params) - 0.5 - 0.02, ymin=0.0, ymax=1.0, color="k", lw=1.3)
        ax.set_xlim(-0.5, len(self.params) - 0.5)
        ax.set_ylabel(rf"${rho_latex}\left(y_{{\rm pred}}, y_{{\rm true}}\right)$")
        ax.legend(loc="upper right")
        return {"forward_pred_metrics": fig}


class CovarianceAm(Figure):
    cache_name = "covariance_am"

    def _set_rc(self):
        set_rc(fontsize=24, cmap="tab10")
        mpl.rcParams.update({"grid.color": "0.85"})

    def get_data(self):
        mahdir = ROOT.joinpath("data", "processed", "bolshoi_m12")
        mah_data = get_mah(
            mahdir, cutoff_missing=0.05, cutoff_particle=0.05, log_mbin_spacing=False
        )
        scales = mah_data["scales"][:-1]
        ma = mah_data["ma_peak"][:, :-1]
        mass_bins = mah_data["mass_bins"]
        am = mah_data["am"]
        n_mbins = mass_bins.shape[0]
        n_haloes = ma.shape[0]

        # use interpolation to ensure linear spacing in m(a)
        new_scales = np.linspace(min(scales), max(scales), 100)
        new_ma = np.zeros((ma.shape[0], len(new_scales)))
        n_scales = new_scales.shape[0]
        for ii in range(n_haloes):
            f = interp1d(scales, ma[ii, :], bounds_error=False, fill_value=np.nan)
            new_ma[ii, :] = f(new_scales)

        assert np.sum(np.isnan(new_ma)) == 0

        corr_matrix_ma = np.zeros((n_scales, n_scales))
        for ii in range(n_scales):
            for jj in range(n_scales):
                corr_matrix_ma[ii, jj] = spearmanr(new_ma[:, -ii - 1], new_ma[:, jj])

        corr_matrix_am = np.zeros((n_mbins, n_mbins))
        for ii in range(n_scales):
            for jj in range(n_scales):
                corr_matrix_am[ii, jj] = spearmanr(am[:, -ii - 1], am[:, jj])

        return {
            "corr_am": corr_matrix_am,
            "corr_ma": corr_matrix_ma,
            "mass_bins": mass_bins,
            "scales": new_scales,
        }

    def get_figures(self, data: Dict[str, np.ndarray]) -> Dict[str, mpl.figure.Figure]:
        # (1) ma covariance figure
        corr = data["corr_ma"]
        scales = data["scales"]
        scale_labels = np.linspace(min(scales), max(scales), 5)
        scale_bin_labels = [rf"${x:.1f}$" for x in scale_labels]

        fig1, ax = plt.subplots(1, 1)
        im = ax.imshow(corr, vmin=0, vmax=1, extent=(-0.01, 100, -0.01, 100))

        ax.set_xlabel(r"$a_{i}$")
        ax.set_ylabel(r"$a_{j}$")
        ax.set_title(rf"${rho_latex}\left(m(a_{{i}}), m(a_{{j}}) \right)$", pad=15.0)

        new_xticks = np.linspace(0, 100, 5)
        ax.set_xticks(ticks=new_xticks, labels=scale_bin_labels)
        ax.set_yticks(ticks=new_xticks, labels=scale_bin_labels)

        # colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        fig1.colorbar(im, cax=cax, orientation="vertical")

        # (2) am covariance figure
        corr = data["corr_am"]
        mass_bins = data["mass_bins"]
        mass_bin_labels = np.linspace(mass_bins.min(), mass_bins.max(), 6)
        mass_bin_labels = [rf"${x:.1f}$" for x in mass_bin_labels]

        fig2, ax = plt.subplots(1, 1)
        im = ax.imshow(corr, vmin=0, vmax=1, extent=(-0.01, 100, -0.01, 100))

        ax.set_xlabel(r"$m_{i}$")
        ax.set_ylabel(r"$m_{j}$")
        ax.set_title(rf"${rho_latex}\left( a(m_{{i}}), a(m_{{j}}) \right)$", pad=15.0)
        ax.set_xticks(ticks=ax.get_xticks()[1:], labels=mass_bin_labels)
        ax.set_yticks(ticks=ax.get_yticks()[1:], labels=mass_bin_labels)

        # colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        fig2.colorbar(im, cax=cax, orientation="vertical")

        return {"ma_corr_matrix": fig1, "am_corr_matrix": fig2}


@click.command()
@click.option("--overwrite", "-o", is_flag=True, default=False)
@click.option("--ext", default="png", type=str)
@click.option("--seed", default=42, type=int)
def main(overwrite, ext, seed):
    rng = np.random.default_rng(seed=seed)
    CorrelationMAH(overwrite, ext, rng).save()
    PredictMAH(overwrite, ext, rng).save()
    InvPredMetrics(overwrite, ext, rng).save()
    ForwardPredMetrics(overwrite, ext, rng).save()
    CovarianceAm(overwrite, ext, rng).save()
    TriangleSamples(overwrite, ext, rng).save()  # FIXME: always last (bolding issue)


if __name__ == "__main__":
    main()
