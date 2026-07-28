import argparse
import os
import sys

import h5py
import hist
import numpy as np


def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


wremnants_base = os.environ.get("WREM_BASE", None)
if wremnants_base is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wremnants_base = os.path.abspath(os.path.join(script_dir, "../.."))
    if os.path.exists(os.path.join(wremnants_base, "wums")):
        sys.path.insert(0, wremnants_base)

from rabbit import tensorwriter
from wremnants.postprocessing.scetlib_np import response_matrix as scetlib_np_response
from wremnants.postprocessing.theory_variation_labels import (
    LATTICE_CORRELATED_NP_UNCERTAINTIES,
    TNP_UNCERTAINTIES,
    TRANSITION_FO_UNCERTAINTIES,
)

try:
    from wremnants.postprocessing.theory_variation_labels import (
        LATTICE_GAMMA_NP_UNCERTAINTIES,
    )
except ImportError:
    # The list is being removed upstream in the param-model era (the model
    # computes the gamma_nu NP on the fly). Keep a local copy so the legacy
    # (non --scetlibNPParamModel) tensor stays reproducible.
    LATTICE_GAMMA_NP_UNCERTAINTIES = [
        [
            "lambda2_nu0.0696-lambda4_nu0.0122-lambda_inf_nu1.1Ext",
            "lambda2_nu0.1044-lambda4_nu0.0026-lambda_inf_nu2.1Ext",
            "scetlibNPgammaEigvar1",
        ],
        [
            "lambda2_nu0.1153-lambda4_nu0.0032-lambda_inf_nu1.6Ext",
            "lambda2_nu0.0587-lambda4_nu0.0116-lambda_inf_nu1.6Ext",
            "scetlibNPgammaEigvar2",
        ],
        [
            "lambda2_nu0.0873-lambda4_nu0.0092",
            "lambda2_nu0.0867-lambda4_nu0.0056",
            "scetlibNPgammaEigvar3",
        ],
    ]
from wremnants.postprocessing.datagroups.datagroups import Datagroups
from wremnants.utilities import common, theory_utils
from wremnants.utilities.io_tools import base_io
from wums import output_tools

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="Input HDF5 file with histograms")
parser.add_argument("-o", "--output", default="./", help="output directory")
parser.add_argument("--outname", default="my_tensor", help="output file name")
parser.add_argument(
    "--histName", default="ptll", help="Histogram name to use (default: ptll)"
)
parser.add_argument(
    "--procFilters",
    nargs="*",
    default=[],
    help="Processes to include (default: all processes in the file; "
    "e.g. --procFilters Zmumu for a signal-only fit)",
)
parser.add_argument(
    "--sparse", default=False, action="store_true", help="Make sparse tensor"
)
parser.add_argument(
    "--systematicType",
    choices=["log_normal", "normal"],
    default="log_normal",
    help="probability density for systematic variations",
)
parser.add_argument(
    "--rebin",
    nargs="*",
    default=[],
    metavar="AXIS=N",
    help="Merge N adjacent bins along AXIS of the fit observable, e.g. "
    "--rebin cosThetaStarll=2 phiStarll=4; applied consistently to data, "
    "MC and all variation templates (for binning scans without re-running "
    "the histmaker)",
)
parser.add_argument(
    "--factorizeSystAxes",
    nargs="*",
    default=[],
    metavar="AXIS",
    help="Flatten the theory systematic templates over these (trailing) axes "
    "of the fit observable: the variation/nominal ratio is computed "
    "inclusively in them and broadcast onto the nominal, removing e.g. "
    "acceptance-induced angular structure from templates whose underlying "
    "corrections depend only on (ptV, yV). Does not affect experimental "
    "systematics.",
)
parser.add_argument(
    "--pseudoData",
    type=str,
    default=None,
    metavar="HISTNAME",
    help="Bias-test mode: replace the Asimov data with this histogram from "
    "the signal process (e.g. an alternative-model template like "
    "ptll_..._FlavDepNP), projected onto the fit axes. The fit of the "
    "unchanged nominal model against it measures the alpha_s bias.",
)
parser.add_argument(
    "--pseudoDataEntry",
    type=str,
    default=None,
    metavar="LABEL",
    help="Entry of the 'vars' axis of the --pseudoData histogram to use "
    "(e.g. lambda20.5 or pdfCT18ZNNLO_as_0120)",
)
parser.add_argument(
    "--ewHistsFile",
    type=str,
    default=None,
    help="Optional histmaker output providing the EW ptll_<gen>_Corr hists "
    "when the main infile predates the EW wiring: the per-bin EW/nominal "
    "weight ratios from this file are applied to the main file's nominal "
    "templates (ratios are smooth and MC-stat robust, so a partial-files "
    "run is sufficient)",
)
parser.add_argument(
    "--scetlibNPParamModel",
    action="store_true",
    help="Prepare the tensor for rabbit_fit --paramModel "
    "'wremnants.postprocessing.scetlib_np.SCETlibNPParamModel' (PR #701): "
    "DROP the 6 resumNonpert lambda nuisances (the model computes the lambda "
    "variations on the fly - keeping them would double count) and embed the "
    "scetlib_np auxiliary (response matrix R + gen-total N_gen from the "
    "histmaker's nominal_prefsr_yieldsUnfolding/prefsr hists)",
)
parser.add_argument(
    "--lumiScale",
    type=float,
    default=1.0,
    help="Scale the MC normalization luminosity by this factor, for luminosity "
    "projection studies (same meaning as setupRabbit.py --lumiScale). The "
    "expected yields scale by the factor, so an Asimov fit sees the data "
    "statistical uncertainty shrink by sqrt(factor). Default 1.0 = no scaling.",
)
parser.add_argument(
    "--lumiScaleVarianceLinearly",
    action="store_true",
    help="With --lumiScale, scale the MC-stat variance linearly instead of "
    "quadratically, i.e. pretend there are really proportionally more MC "
    "events so the bin-by-bin (binByBinStat) uncertainty also improves by "
    "sqrt(factor). Default (quadratic) keeps the RELATIVE MC-stat uncertainty "
    "unchanged, i.e. the same MC sample simply reweighted.",
)
args = parser.parse_args()

# Load data from HDF5 file
infile_path = os.path.abspath(args.infile)
if not os.path.exists(infile_path):
    raise FileNotFoundError(f"Input file not found: {infile_path}")

h5file = h5py.File(infile_path, "r")
results = base_io.load_results_h5py(h5file)

hist_name = args.histName
all_procs = [p for p in results.keys() if p != "meta_info"]
print_flush(f"All processes found in file: {all_procs}")

if args.procFilters:
    procs_to_use = [p for p in all_procs if any(filt in p for filt in args.procFilters)]
else:
    procs_to_use = all_procs

print_flush(f"Processes after filtering: {procs_to_use}")

data_procs = [p for p in all_procs if "SingleMuon" in p or p.startswith("Data")]
mc_procs = [p for p in procs_to_use if p not in data_procs]
print_flush(f"MC processes found: {mc_procs}")

# Absolute normalization: the histmaker fills MC with sign(genWeight) only,
# so every MC histogram must be scaled by xsec * lumi / genWeightSum before
# entering the tensor (applied centrally in _get_hist and the nominal load
# below; the same factor multiplies nominal and variations, so ratios are
# unaffected but yields and MC-stat variances become physical)
if not data_procs or "lumi" not in results[data_procs[0]]:
    raise RuntimeError("No data process with lumi found; cannot normalize MC")
lumi_pb = results[data_procs[0]]["lumi"] * 1000.0
if args.lumiScale != 1.0:
    print_flush(
        f"--lumiScale {args.lumiScale}: {lumi_pb:.3f}/pb -> "
        f"{lumi_pb * args.lumiScale:.3f}/pb "
        f"(MC-stat variance scaled "
        f"{'LINEARLY' if args.lumiScaleVarianceLinearly else 'quadratically'})"
    )
    lumi_pb *= args.lumiScale
proc_scale = {}
for proc in mc_procs:
    xsec = results[proc].get("dataset", {}).get("xsec")
    wsum = results[proc].get("weight_sum")
    if not xsec or not wsum:
        raise RuntimeError(f"Cannot normalize {proc}: xsec={xsec}, weight_sum={wsum}")
    proc_scale[proc] = xsec * lumi_pb / wsum
    print_flush(
        f"Normalization {proc}: xsec={xsec} pb, weight_sum={wsum:.6g}, "
        f"lumi={lumi_pb:.3f}/pb -> scale={proc_scale[proc]:.6g}"
    )


def _scale_mc(h, scale):
    """Apply the xsec*lumi normalization to a MC histogram.

    `h * scale` scales the value by `scale` and the variance by `scale**2`, so
    the RELATIVE MC-stat uncertainty is invariant -- correct for "the same MC
    sample, reweighted". With --lumiScaleVarianceLinearly we additionally divide
    the variance by the lumi factor, so it scales linearly overall and the
    relative MC-stat uncertainty improves by sqrt(factor), i.e. "there really
    are proportionally more MC events".
    """
    hs = h * scale
    if args.lumiScaleVarianceLinearly and args.lumiScale != 1.0:
        view = hs.view(flow=True)
        if hasattr(view, "variance"):
            view.variance /= args.lumiScale
    return hs


# Load MC histograms (xsec*lumi normalized)
h_mc_dict = {}
for proc in mc_procs:
    if "output" in results[proc] and hist_name in results[proc]["output"]:
        h_proxy = results[proc]["output"][hist_name]
        h = h_proxy.get() if hasattr(h_proxy, "get") else h_proxy
        h_mc_dict[proc] = _scale_mc(h, proc_scale[proc])

# Identify signal processes before closing file
signal_procs = [p for p in mc_procs if "Zmumu" in p or "Ztautau" in p]


# EW/FSR corrections (borrowed 13 TeV ratio files, filled by mz_5TeV.py)
EW_CORR_TAGS = (
    "powhegFOEW",
    "pythiaew_ISR",
    "horaceqedew_FSR",
    "horacelophotosmecoffew_FSR",
)


def _corr_hist_names(results, proc):
    """Match histmaker output <base>_<generator>_Corr and classify by type."""
    names = {"pdfas": None, "pdfvars": None, "central": None}
    if proc not in results or "output" not in results[proc]:
        return names
    for name in results[proc]["output"]:
        if not name.endswith("_Corr"):
            continue
        if any(tag in name for tag in EW_CORR_TAGS):
            continue
        if "pdfas" in name:
            names["pdfas"] = name
        elif "pdfvars" in name:
            names["pdfvars"] = name
        else:
            names["central"] = name
    return names


def _get_hist(results, proc, name):
    h_proxy = results[proc]["output"][name]
    h = h_proxy.get() if hasattr(h_proxy, "get") else h_proxy
    return _scale_mc(h, proc_scale[proc]) if proc in proc_scale else h


h_theory_corr_pdfas_dict = {}
h_theory_corr_pdfvars_dict = {}
h_theory_corr_central_dict = {}
h_qcd_helicity_dict = {}
h_ew_corr_dict = {}  # proc -> {generator: hist}

for proc in signal_procs:
    names = _corr_hist_names(results, proc)
    if names["pdfas"]:
        h_theory_corr_pdfas_dict[proc] = _get_hist(results, proc, names["pdfas"])
        print_flush(f"{proc}: found alpha_s variation hist {names['pdfas']}")
    else:
        print_flush(f"Warning: no pdfas theory hist for {proc}")
    if names["pdfvars"]:
        h_theory_corr_pdfvars_dict[proc] = _get_hist(results, proc, names["pdfvars"])
        print_flush(f"{proc}: found PDF variation hist {names['pdfvars']}")
    else:
        print_flush(f"Warning: no pdfvars theory hist for {proc}")
    if names["central"]:
        h_theory_corr_central_dict[proc] = _get_hist(results, proc, names["central"])
        print_flush(f"{proc}: found SCETlib theory variation hist {names['central']}")
    for name in results[proc]["output"]:
        if name.endswith("_qcdScaleByHelicity"):
            h_qcd_helicity_dict[proc] = _get_hist(results, proc, name)
            print_flush(f"{proc}: found qcdScaleByHelicity hist {name}")
        for tag in EW_CORR_TAGS:
            if name.endswith(f"{tag}_Corr"):
                h_ew_corr_dict.setdefault(proc, {})[tag] = _get_hist(
                    results, proc, name
                )
                print_flush(f"{proc}: found EW correction hist {name}")

# EW hists from an auxiliary histmaker file (--ewHistsFile): apply the
# per-bin EW/nominal weight ratios from that file to this file's nominal
# templates, for productions that predate the EW wiring
if args.ewHistsFile:
    ew_results = base_io.load_results_h5py(
        h5py.File(os.path.abspath(args.ewHistsFile), "r")
    )
    for proc in signal_procs:
        if proc in h_ew_corr_dict:
            continue  # main file already carries EW hists for this proc
        if proc not in ew_results or "output" not in ew_results[proc]:
            print_flush(f"Warning: {proc} not in --ewHistsFile, no EW systematics")
            continue
        ew_out = ew_results[proc]["output"]
        if hist_name not in ew_out:
            continue
        h_nom_aux = ew_out[hist_name]
        h_nom_aux = h_nom_aux.get() if hasattr(h_nom_aux, "get") else h_nom_aux
        nom_aux_vals = h_nom_aux.values()
        for name in ew_out:
            for tag in EW_CORR_TAGS:
                if not name.endswith(f"{tag}_Corr"):
                    continue
                h_aux = ew_out[name]
                h_aux = h_aux.get() if hasattr(h_aux, "get") else h_aux
                ratio = np.ones_like(h_aux.values())
                np.divide(
                    h_aux.values(),
                    nom_aux_vals[..., None],
                    out=ratio,
                    where=nom_aux_vals[..., None] != 0,
                )
                h_ew = h_aux.copy()
                new_vals = ratio * h_mc_dict[proc].values()[..., None]
                view = h_ew.view(flow=False)
                if view.dtype.fields:
                    view["value"] = new_vals
                    view["variance"] = 0.0
                else:
                    view[...] = new_vals
                h_ew_corr_dict.setdefault(proc, {})[tag] = h_ew
                print_flush(
                    f"{proc}: EW hist {name} transferred from --ewHistsFile "
                    f"via per-bin ratio"
                )

# muon momentum scale/resolution variations (scarekit bootstrap = stat,
# window-variation spread = syst), filled by mz_5TeV.py --muonCorr scarekit
# for all MC processes
MUON_VAR_HISTS = [
    "ptll_muonScaleUp",
    "ptll_muonScaleDown",
    "ptll_muonResUp",
    "ptll_muonResDown",
    "ptll_muonScaleSystUp",
    "ptll_muonScaleSystDown",
    "ptll_muonResSystUp",
    "ptll_muonResSystDown",
]
h_muon_var_dict = {}
for proc in mc_procs:
    if proc not in results or "output" not in results[proc]:
        continue
    found = {
        nm: _get_hist(results, proc, nm)
        for nm in MUON_VAR_HISTS
        if nm in results[proc]["output"]
    }
    if found:
        h_muon_var_dict[proc] = found
        print_flush(f"{proc}: found muon calibration variation hists {sorted(found)}")

# Z boson mass variations (MiNNLO Breit-Wigner reweighting, MEParamWeight),
# filled by mz_5TeV.py for Z MC; the +-2.1 MeV (PDG) entries give the mZ
# uncertainty nuisance
MASSWEIGHT_HIST = "ptll_massWeightZ"
h_massweight_dict = {}
for proc in mc_procs:
    if (
        proc in results
        and "output" in results[proc]
        and MASSWEIGHT_HIST in results[proc]["output"]
    ):
        h_massweight_dict[proc] = _get_hist(results, proc, MASSWEIGHT_HIST)
        print_flush(f"{proc}: found Z mass variation hist {MASSWEIGHT_HIST}")

# Z width and sin2theta variations (same MiNNLO reweighting machinery)
WIDTHWEIGHT_HIST = "ptll_widthWeightZ"
SIN2THETAWEIGHT_HIST = "ptll_sin2thetaWeightZ"
h_widthweight_dict = {}
h_sin2thetaweight_dict = {}
for proc in mc_procs:
    if proc not in results or "output" not in results[proc]:
        continue
    if WIDTHWEIGHT_HIST in results[proc]["output"]:
        h_widthweight_dict[proc] = _get_hist(results, proc, WIDTHWEIGHT_HIST)
        print_flush(f"{proc}: found Z width variation hist {WIDTHWEIGHT_HIST}")
    if SIN2THETAWEIGHT_HIST in results[proc]["output"]:
        h_sin2thetaweight_dict[proc] = _get_hist(results, proc, SIN2THETAWEIGHT_HIST)
        print_flush(f"{proc}: found sin2theta variation hist {SIN2THETAWEIGHT_HIST}")

# b,c quark mass variation hists (MSHT20 mbrange/mcrange LHE weights,
# renormalized to the range set's own central in the histmaker)
BCMASS_HISTS = {
    "pdfMSHT20mbrange": "ptll_pdfMSHT20mbrange",
    "pdfMSHT20mcrange": "ptll_pdfMSHT20mcrange",
}
h_bcmass_dicts = {key: {} for key in BCMASS_HISTS}
for proc in mc_procs:
    if proc not in results or "output" not in results[proc]:
        continue
    for key, hname in BCMASS_HISTS.items():
        if hname in results[proc]["output"]:
            h_bcmass_dicts[key][proc] = _get_hist(results, proc, hname)
            print_flush(f"{proc}: found b,c quark mass variation hist {hname}")

# pseudodata for bias tests: load the alternative-model histogram from the
# (first) signal process before the file is closed
h_pseudo_raw = None
if args.pseudoData:
    pseudo_proc = signal_procs[0] if signal_procs else mc_procs[0]
    if args.pseudoData not in results[pseudo_proc]["output"]:
        raise RuntimeError(
            f"--pseudoData histogram {args.pseudoData} not found for process "
            f"{pseudo_proc}; available: {sorted(results[pseudo_proc]['output'])}"
        )
    h_pseudo_raw = _get_hist(results, pseudo_proc, args.pseudoData)

h5file.close()

if h_mc_dict:
    # Asimov data = sum of ALL MC processes (signal + backgrounds), so the
    # stored data matches the model expectation exactly and fitting it
    # directly (-t 0, e.g. for pseudodata bias tests) closes at zero
    h_data = None
    for p, hmc in h_mc_dict.items():
        hn = hmc[{"vars": 0}] if "vars" in hmc.axes.name else hmc
        h_data = hn.copy() if h_data is None else h_data + hn
    print_flush(
        f"Using sum of MC processes {list(h_mc_dict.keys())} as expected data (Asimov)"
    )
else:
    raise RuntimeError("No MC processes found to use as data")

# Handle vars axis if present
axis_names = [ax.name for ax in h_data.axes] if hasattr(h_data, "axes") else []
h_data_base = h_data[{"vars": 0}] if "vars" in axis_names else h_data
h_mc_base = {
    proc: (h[{"vars": 0}] if "vars" in h.axes.name else h)
    for proc, h in h_mc_dict.items()
}

rebin_factors = {}
for spec in args.rebin:
    ax_name, sep, factor = spec.partition("=")
    if not sep or not factor.isdigit():
        raise ValueError(f"Invalid --rebin spec '{spec}', expected AXIS=N")
    rebin_factors[ax_name] = int(factor)


def _rebin(h):
    for ax_name, factor in rebin_factors.items():
        if ax_name in h.axes.name:
            h = h[{ax_name: slice(None, None, hist.rebin(factor))}]
    return h


if rebin_factors:
    h_data_base = _rebin(h_data_base)
    h_mc_base = {proc: _rebin(h) for proc, h in h_mc_base.items()}
    print_flush(f"Rebinned fit observable with {rebin_factors}")

if h_pseudo_raw is not None:
    hp = h_pseudo_raw
    if args.pseudoDataEntry is not None:
        if "vars" not in hp.axes.name:
            raise RuntimeError(
                f"--pseudoDataEntry given but {args.pseudoData} has no 'vars' axis"
            )
        hp = hp[{"vars": args.pseudoDataEntry}]
    elif "vars" in hp.axes.name:
        raise RuntimeError(
            f"{args.pseudoData} has a 'vars' axis; select an entry with "
            f"--pseudoDataEntry (available: {list(hp.axes['vars'])})"
        )
    hp = hp.project(*h_data_base.axes.name)
    hp = _rebin(hp)
    # replace only the signal component of the Asimov sum; the other
    # processes stay at their nominal expectation
    pseudo_vals = hp.values() + (h_data_base.values() - h_mc_base[pseudo_proc].values())
    # keep the data histogram storage type (values are what matters;
    # pseudodata is a smooth template, Poisson errors come from the fit)
    h_pseudo = h_data_base.copy()
    view = h_pseudo.view(flow=False)
    if view.dtype.fields:
        view["value"] = pseudo_vals
        view["variance"] = pseudo_vals
    else:
        view[...] = pseudo_vals
    h_data_base = h_pseudo
    print_flush(
        f"PSEUDODATA: Asimov data replaced by {args.pseudoData}"
        + (f" [vars={args.pseudoDataEntry}]" if args.pseudoDataEntry else "")
        + f" (yield {float(h_data_base.values().sum()):.1f})"
    )


def _hist_yield(h):
    s = h.sum()
    return s.value if hasattr(s, "value") else float(s)


# Asimov: MC stays at its absolute lumi x xsec normalization, so the
# luminosity uncertainty applies (see below). mc_scale is kept as a hook
# used by the systematic templates.
mc_scale = 1.0

# Build tensor
writer = tensorwriter.TensorWriter(
    sparse=args.sparse,
    systematic_type=args.systematicType,
)

channel_name = "ch0"
writer.add_channel(h_data_base.axes, channel_name)
writer.add_data(h_data_base, channel_name)

background_procs = [p for p in mc_procs if p not in signal_procs]

# Add processes
for proc in signal_procs:
    if proc in h_mc_base:
        writer.add_process(h_mc_base[proc], proc, channel_name, signal=False)

for proc in background_procs:
    if proc in h_mc_base:
        writer.add_process(h_mc_base[proc], proc, channel_name, signal=False)


def _project_var(h_corr, var_name, proc_name):
    names = list(h_mc_base[proc_name].axes.name)
    h = h_corr[{"vars": var_name}].project(*names)
    if args.factorizeSystAxes:
        keep = [n for n in names if n not in args.factorizeSystAxes]
        if keep != names[: len(keep)]:
            raise RuntimeError(
                "--factorizeSystAxes must be the trailing axes of the fit observable"
            )
        h_cen = h_corr[{"vars": 0}].project(*names)
        num = h.project(*keep).values()
        den = h_cen.project(*keep).values()
        ratio = np.divide(num, den, out=np.ones_like(num), where=den > 0)
        h_fact = h_cen.copy()
        view = h_fact.view(flow=False)
        scaled = h_cen.values() * ratio.reshape(
            ratio.shape + (1,) * (len(names) - len(keep))
        )
        if view.dtype.fields:
            view["value"] = scaled
        else:
            view[...] = scaled
        h = h_fact
    return _rebin(h) * mc_scale


# PDF variations from pdfvars correction histogram
# (constrained, correlated across Z processes via common systematic names)
# CT18Z hessian sets are published at 90% CL; scale to 68% (1/1.645)
pdf_cl_scale = theory_utils.pdfMap["ct18z"]["scale"]
for proc_name, h_corr in h_theory_corr_pdfvars_dict.items():
    if "vars" not in h_corr.axes.name:
        print_flush(f"Warning: pdfvars hist for {proc_name} has no vars axis")
        continue
    vars_axis = h_corr.axes["vars"]
    # Exclude central and alpha_s variations (which have "_as_" in the name)
    pdf_variations = [
        str(v) for v in vars_axis if str(v) != "central" and "_as_" not in str(v)
    ]
    num_pairs = 0
    for var_name_up, var_name_down in zip(pdf_variations[1::2], pdf_variations[2::2]):
        writer.add_systematic(
            [
                _project_var(h_corr, var_name_up, proc_name),
                _project_var(h_corr, var_name_down, proc_name),
            ],
            f"{var_name_up}_{var_name_down}",
            proc_name,
            channel_name,
            kfactor=pdf_cl_scale,
            constrained=True,
            symmetrize="quadratic",
            groups=["pdfCT18Z"],
        )
        num_pairs += 1
    print_flush(f"Added {num_pairs} PDF systematic pairs for process {proc_name}")

# alpha_s variations from pdfas correction histogram (unconstrained, noi=True)
for proc_name, h_corr in h_theory_corr_pdfas_dict.items():
    if "vars" not in h_corr.axes.name:
        print_flush(f"Warning: pdfas hist for {proc_name} has no vars axis")
        continue
    vars_axis = h_corr.axes["vars"]
    var_name_0120 = "pdfCT18ZNNLO_as_0120"
    var_name_0116 = "pdfCT18ZNNLO_as_0116"
    if var_name_0120 in vars_axis and var_name_0116 in vars_axis:
        writer.add_systematic(
            [
                _project_var(h_corr, var_name_0120, proc_name),
                _project_var(h_corr, var_name_0116, proc_name),
            ],
            "pdfAlphaS",
            proc_name,
            channel_name,
            constrained=False,
            noi=True,  # Only alpha_s variations have noi=True
        )
        print_flush(f"Added pdfAlphaS systematic for process {proc_name} (noi=True)")
    else:
        print_flush(f"Warning: Could not find both alpha_s variations for {proc_name}")

# SCETlib+DYTurbo theory uncertainties from the central correction histogram,
# following the canonical grouping of the main analysis
# (wremnants/postprocessing/theory_variation_labels.py, as applied in
# theory_fit_writer.py): correlated non-perturbative parameters, lattice
# gamma NP eigenvariations, theory nuisance parameters (TNPs), and
# transition/fixed-order scale variations. The remaining entries on the
# vars axis (nested scale envelopes, muf/kappa components, alternative NP
# ranges, single-parameter gamma variations) are deliberately not used:
# the TNPs replace the resummation scale envelopes, the pt20 FO envelope
# covers the muf/kappa components, and the gamma eigenvariations replace
# the single-parameter lattice variations.

# The 5 TeV SCETlib file names the delta_lambda2 variations by absolute
# value (central 0.125 +- 0.02) instead of by the delta itself
SCETLIB_VAR_ALTERNATES = {
    "delta_lambda20.02": "delta_lambda20.145",
    "delta_lambda2-0.02": "delta_lambda20.105",
}


def _canonical_scetlib_uncertainties():
    uncs = []  # (var_up, var_down, name, symmetrize, groups)
    # With the SCETlib-NP param model (--scetlibNPParamModel) the fit computes
    # the lambda variations on the fly, so the fixed-variation resumNonpert
    # nuisances must be dropped to avoid double counting; the TNPs and
    # transition/FO scales are perturbative and stay either way.
    if not args.scetlibNPParamModel:
        for up, down, name in LATTICE_CORRELATED_NP_UNCERTAINTIES:
            uncs.append(
                (
                    up,
                    down,
                    name.replace("chargeVgenNP0", ""),
                    "average",
                    ["resumNonpert", "theory"],
                )
            )
        for up, down, name in LATTICE_GAMMA_NP_UNCERTAINTIES:
            uncs.append((up, down, name, "average", ["resumNonpert", "theory"]))
    for up, down in TNP_UNCERTAINTIES:
        uncs.append(
            (
                up,
                down,
                f"resumTNP_{down.split('-')[0]}",
                "average",
                ["resumTNP", "theory"],
            )
        )
    for up, down, name in TRANSITION_FO_UNCERTAINTIES:
        uncs.append((up, down, name, "quadratic", ["resumTransitionFOScale", "theory"]))
    return uncs


for proc_name, h_corr in h_theory_corr_central_dict.items():
    if "vars" not in h_corr.axes.name:
        print_flush(f"Warning: central corr hist for {proc_name} has no vars axis")
        continue
    var_names = [str(v) for v in h_corr.axes["vars"]]
    used_vars = {"pdf0", "central"}
    n_added = 0
    for (
        var_up,
        var_down,
        syst_name,
        symmetrize,
        groups,
    ) in _canonical_scetlib_uncertainties():
        var_up = var_up if var_up in var_names else SCETLIB_VAR_ALTERNATES.get(var_up)
        var_down = (
            var_down if var_down in var_names else SCETLIB_VAR_ALTERNATES.get(var_down)
        )
        if var_up not in var_names or var_down not in var_names:
            print_flush(
                f"Warning: skipping {syst_name} for {proc_name}, "
                f"variations {var_up}/{var_down} not found"
            )
            continue
        writer.add_systematic(
            [
                _project_var(h_corr, var_up, proc_name),
                _project_var(h_corr, var_down, proc_name),
            ],
            syst_name,
            proc_name,
            channel_name,
            constrained=True,
            symmetrize=symmetrize,
            groups=groups,
        )
        used_vars.update((var_up, var_down))
        n_added += 1
    unused = [v for v in var_names if v not in used_vars]
    print_flush(
        f"Added {n_added} SCETlib theory systematics for process {proc_name} "
        f"({len(unused)} axis entries deliberately unused: {unused})"
    )

# Helicity-decomposed QCD scale uncertainties (angular coefficients): one
# nuisance per helicity cross section, spanning the muR/muF envelope of that
# helicity, reweighted through the angular decomposition (MiNNLO), following
# the 13 TeV conventions (rabbit_theory_helper.add_minnlo_scale_uncertainty):
# - sigma_UL (helicity -1) skipped: covered by the SCETlib TNP/transition/FO
#   groups (helicities_to_exclude=[-1] at 13 TeV whenever SCETlib is used)
# - per helicity, one set of nuisances decorrelated in coarse ptVgen bins
#   plus one inclusive nuisance scaled by sqrt((n-1)/n) against double counting
# - quadratic symmetrization
# - pythia_shower_kt as a separate mirrored nuisance (pre-FSR vs LHE moments)
for proc_name, h_hel in h_qcd_helicity_dict.items():
    if "vars" not in h_hel.axes.name:
        print_flush(
            f"Warning: qcdScaleByHelicity hist for {proc_name} has no vars axis"
        )
        continue
    var_names_hel = [str(v) for v in h_hel.axes["vars"]]
    has_ptv = "ptVgen" in h_hel.axes.name
    if has_ptv:
        ptv_edges = h_hel.axes["ptVgen"].edges
        n_ptv = len(ptv_edges) - 1
        kfactor_incl = np.sqrt((n_ptv - 1) / n_ptv)
        h_nom_slices = {
            k: h_hel[{"vars": "nominal", "ptVgen": slice(k, k + 1, hist.sum)}]
            for k in range(n_ptv)
        }
        h_nom_total = h_hel[{"vars": "nominal"}].project(
            *h_mc_base[proc_name].axes.name
        )
    else:
        kfactor_incl = 1.0
    n_added = 0
    for ihel in range(0, 8):
        var_up = f"helicity_{ihel}_Up"
        var_down = f"helicity_{ihel}_Down"
        if var_up not in var_names_hel or var_down not in var_names_hel:
            continue
        # inclusive-in-ptV nuisance (correlated part)
        writer.add_systematic(
            [
                _project_var(h_hel, var_up, proc_name),
                _project_var(h_hel, var_down, proc_name),
            ],
            f"qcdScaleHelicity{ihel}Inclusive",
            proc_name,
            channel_name,
            kfactor=kfactor_incl,
            constrained=True,
            symmetrize="quadratic",
            groups=["angularCoeffs", f"angularCoeffs_A{ihel}", "theory"],
        )
        n_added += 1
        if not has_ptv:
            continue
        # nuisances decorrelated in coarse ptVgen bins: vary only the
        # contribution of one ptV slice, keep the rest at nominal
        for k in range(n_ptv):
            fit_axes = h_mc_base[proc_name].axes.name
            h_var_k = {
                var: (
                    h_nom_total
                    - h_nom_slices[k].project(*fit_axes)
                    + h_hel[{"vars": var, "ptVgen": slice(k, k + 1, hist.sum)}].project(
                        *fit_axes
                    )
                )
                for var in (var_up, var_down)
            }
            writer.add_systematic(
                [
                    _rebin(h_var_k[var_up]) * mc_scale,
                    _rebin(h_var_k[var_down]) * mc_scale,
                ],
                f"qcdScaleHelicity{ihel}PtV{int(ptv_edges[k])}to{int(ptv_edges[k + 1])}",
                proc_name,
                channel_name,
                constrained=True,
                symmetrize="quadratic",
                groups=["angularCoeffs", f"angularCoeffs_A{ihel}", "theory"],
            )
            n_added += 1
    # parton-shower/recoil uncertainty on the coefficients (one-sided, mirrored)
    if "pythia_shower_kt" in var_names_hel:
        h_shower = _project_var(h_hel, "pythia_shower_kt", proc_name)
        h_shower_nom = _project_var(h_hel, "nominal", proc_name)
        if np.allclose(h_shower.values(), h_shower_nom.values()):
            print_flush(
                f"pythia_shower_kt is identical to nominal for {proc_name}, skipping"
            )
        else:
            writer.add_systematic(
                h_shower,
                "helicity_shower_kt",
                proc_name,
                channel_name,
                mirror=True,
                constrained=True,
                groups=["angularCoeffs", "theory"],
            )
            n_added += 1
    print_flush(
        f"Added {n_added} helicity-decomposed QCD scale systematics for {proc_name}"
    )

# Muon momentum scale/resolution uncertainties (scarekit): stat from the
# bootstrap spread, syst from the fit-window-variation spreads (syst3 for
# scale, syst4 for resolution). Kinematic up/down variations of the
# templates, correlated across MC processes via the shared systematic names
for proc_name, hvars in h_muon_var_dict.items():
    if proc_name not in h_mc_base:
        continue
    # Finer scale/resolution groups are nested alongside the muonCalibration
    # umbrella, as at 13 TeV (setupRabbit.py groups=["scaleCrctn"/"resolutionCrctn",
    # "muonCalibration", ...]). For the scale we use "muonScale" rather than the
    # 13 TeV-internal "scaleCrctn" because only muonScale/nonClosure carry the
    # "Muon scale" impact label in styles.py; resolutionCrctn is labeled there.
    for syst_name, (up_name, dn_name, fine_group) in {
        "muonScaleStat": ("ptll_muonScaleUp", "ptll_muonScaleDown", "muonScale"),
        "muonResStat": ("ptll_muonResUp", "ptll_muonResDown", "resolutionCrctn"),
        "muonScaleSyst": (
            "ptll_muonScaleSystUp",
            "ptll_muonScaleSystDown",
            "muonScale",
        ),
        "muonResSyst": (
            "ptll_muonResSystUp",
            "ptll_muonResSystDown",
            "resolutionCrctn",
        ),
    }.items():
        if up_name not in hvars or dn_name not in hvars:
            continue
        fit_axes = h_mc_base[proc_name].axes.name
        h_up = _rebin(hvars[up_name].project(*fit_axes)) * mc_scale
        h_down = _rebin(hvars[dn_name].project(*fit_axes)) * mc_scale
        if np.allclose(h_up.values(), h_down.values()):
            print_flush(
                f"{syst_name} up and down are identical for {proc_name} "
                "(uncertainty inputs missing or zero?), skipping"
            )
            continue
        writer.add_systematic(
            [h_up, h_down],
            syst_name,
            proc_name,
            channel_name,
            constrained=True,
            symmetrize="average",
            groups=[fine_group, "muonCalibration", "experiment"],
        )
        print_flush(f"Added {syst_name} systematic for process {proc_name}")

# Z boson mass uncertainty: +-2.1 MeV (PDG) Breit-Wigner reweighting entries
# of the massWeight tensor, correlated across Z MC processes
for proc_name, h_mw in h_massweight_dict.items():
    if proc_name not in h_mc_base:
        continue
    shift_labels = list(h_mw.axes["massShift"])
    up_lab, dn_lab = "massShiftZ2p1MeVUp", "massShiftZ2p1MeVDown"
    if up_lab not in shift_labels or dn_lab not in shift_labels:
        print_flush(
            f"{proc_name}: {up_lab}/{dn_lab} not found in {MASSWEIGHT_HIST} "
            f"(axis has {shift_labels}), skipping"
        )
        continue
    fit_axes = h_mc_base[proc_name].axes.name
    h_up = _rebin(h_mw[{"massShift": up_lab}].project(*fit_axes)) * mc_scale
    h_down = _rebin(h_mw[{"massShift": dn_lab}].project(*fit_axes)) * mc_scale
    if np.allclose(h_up.values(), h_down.values()):
        print_flush(
            f"massShiftZ2p1MeV up and down are identical for {proc_name} "
            "(weights missing in the sample?), skipping"
        )
        continue
    writer.add_systematic(
        [h_up, h_down],
        "massShiftZ2p1MeV",
        proc_name,
        channel_name,
        constrained=True,
        symmetrize="average",
        groups=["massShift", "theory"],
    )
    print_flush(f"Added massShiftZ2p1MeV systematic for process {proc_name}")


def _add_label_pair_syst(
    h_var, axis_name, up_lab, dn_lab, syst_name, groups, symmetrize="average"
):
    """Add one constrained symmetrized nuisance from two labeled entries
    of a variation histogram (shared pattern for width/sin2theta/bc masses)."""
    for proc_name, h in h_var.items():
        if proc_name not in h_mc_base:
            continue
        labels = list(h.axes[axis_name])
        if up_lab not in labels or dn_lab not in labels:
            print_flush(
                f"{proc_name}: {up_lab}/{dn_lab} not found on {axis_name} axis "
                f"(has {labels}), skipping {syst_name}"
            )
            continue
        fit_axes = h_mc_base[proc_name].axes.name
        h_up = _rebin(h[{axis_name: up_lab}].project(*fit_axes)) * mc_scale
        h_down = _rebin(h[{axis_name: dn_lab}].project(*fit_axes)) * mc_scale
        if np.allclose(h_up.values(), h_down.values()):
            print_flush(
                f"{syst_name} up and down are identical for {proc_name}, skipping"
            )
            continue
        writer.add_systematic(
            [h_up, h_down],
            syst_name,
            proc_name,
            channel_name,
            constrained=True,
            symmetrize=symmetrize,
            groups=groups,
        )
        print_flush(f"Added {syst_name} systematic for process {proc_name}")


# GammaZ width: EW-fit uncertainty +-0.8 MeV (entries 0/1 of the width weights,
# widthZ2p49493GeV/widthZ2p49333GeV = 2.49413 +- 0.0008 GeV), matching the
# 13 TeV alpha_s fit (setupRabbit.py: WidthZ0p8MeV, "Variation from EW fit").
# The PDG +-2.3 MeV pair (entries 2/4) is deliberately not used, for consistency
# with 13 TeV; switch to widthZ2p4975GeV/widthZ2p4929GeV to recover it.
_add_label_pair_syst(
    h_widthweight_dict,
    "width",
    "widthZ2p49493GeV",
    "widthZ2p49333GeV",
    "widthZ0p8MeV",
    ["widthZ", "theory"],
)

# effective weak mixing angle: EW-fit uncertainty +-0.00003 (entries 0/2;
# entry 1 is the central 0.23154)
_add_label_pair_syst(
    h_sin2thetaweight_dict,
    "sin2theta",
    "sin2thetaZ0p23157",
    "sin2thetaZ0p23151",
    "sin2thetaZ0p00003",
    ["sin2thetaZ", "theory"],
)

# b,c quark masses: the extreme MSHT20 mbrange/mcrange members (pdf1 =
# lowest mass, pdf6/pdf8 = highest; member 0 is the set's own central) as
# one Down/Up nuisance each, following the 13 TeV add_quark_mass_vars
# convention (BC_QUARK_MASS_VARIATIONS: mbrange pdf1/pdf6, mcrange
# pdf1/pdf8) with the quadratic symmetrization used for PDF uncertainties.
# No CL conversion: the MSHT20 range sets are not 90% CL eigenvectors.
_add_label_pair_syst(
    h_bcmass_dicts["pdfMSHT20mbrange"],
    "pdfVar",
    "pdf6",
    "pdf1",
    "pdfMSHT20mbrange",
    ["bcQuarkMass", "theory"],
    symmetrize="quadratic",
)
_add_label_pair_syst(
    h_bcmass_dicts["pdfMSHT20mcrange"],
    "pdfVar",
    "pdf8",
    "pdf1",
    "pdfMSHT20mcrange",
    ["bcQuarkMass", "theory"],
    symmetrize="quadratic",
)

# EW uncertainties from the borrowed 13 TeV ratio files, following
# rabbit_helpers.add_electroweak_uncertainty at 13 TeV:
# - powhegFOEW weak_ps/weak_aem -> virtual EW scheme variations,
#   weak_default -> virtual EW correction on/off (all mirrored)
# - horace FSR / photos MEC-off FSR: systIdx 1, mirrored
# - pythia ISR: systIdx 1, mirrored, kfactor 2
for proc_name, ew_hists in h_ew_corr_dict.items():
    n_added = 0
    h_fo = ew_hists.get("powhegFOEW")
    if h_fo is not None:
        for entry, subgroup in (
            ("weak_ps", "theory_ew_virtZ_scheme"),
            ("weak_aem", "theory_ew_virtZ_scheme"),
            ("weak_default", "theory_ew_virtZ_corr"),
        ):
            writer.add_systematic(
                _rebin(h_fo[{"weak": entry}]) * mc_scale,
                f"powhegFOEW_Corr{entry}",
                proc_name,
                channel_name,
                mirror=True,
                constrained=True,
                groups=[subgroup, "theory_ew", "theory"],
            )
            n_added += 1
    for tag, kfactor in (
        ("horaceqedew_FSR", 1),
        ("horacelophotosmecoffew_FSR", 1),
        ("pythiaew_ISR", 2),
    ):
        h_ew = ew_hists.get(tag)
        if h_ew is None:
            continue
        writer.add_systematic(
            _rebin(h_ew[{"systIdx": 1}]) * mc_scale,
            f"{tag}Corr1",
            proc_name,
            channel_name,
            mirror=True,
            constrained=True,
            kfactor=kfactor,
            groups=[f"theory_ew_{tag}", "theory_ew", "theory"],
        )
        n_added += 1
    print_flush(f"Added {n_added} EW/FSR systematics for {proc_name}")

# Normalization uncertainties on the backgrounds (lnN)
for proc in mc_procs:
    if proc not in h_mc_base:
        continue
    if "Ztautau" in proc:
        writer.add_norm_systematic(
            f"norm_{proc}", proc, channel_name, 1.05, groups=["norm"]
        )
        print_flush(f"Added 5% normalization uncertainty for {proc}")
    elif proc in background_procs:
        writer.add_norm_systematic(
            f"norm_{proc}", proc, channel_name, 1.10, groups=["norm"]
        )
        print_flush(f"Added 10% normalization uncertainty for {proc}")

# Luminosity: the MC yields are absolute (lumi x xsec), so the lumi
# nuisance is meaningful. Value is the canonical per-era number from
# Datagroups.lumi_uncertainties (2017G = 1.019 = 1.9%), fully correlated
# across all MC processes via the shared name "luminosity".
lumi_unc = Datagroups.lumi_uncertainties["2017G"]
lumi_procs = [p for p in mc_procs if p in h_mc_base]
writer.add_norm_systematic(
    "luminosity",
    lumi_procs,
    channel_name,
    lumi_unc,
    groups=["luminosity", "experiment"],
)
print_flush(
    f"Added {(lumi_unc - 1) * 100:.1f}% luminosity uncertainty for {lumi_procs}"
)

# SCETlib-NP param model (PR #701): embed the response matrix R + gen-total
# N_gen as the 'scetlib_np' auxiliary (mirroring setupRabbit), so the
# ParamModel reads R only from the datacard.
if args.scetlibNPParamModel:
    zmumu_procs = [p for p in mc_procs if "Zmumu" in p]
    if len(zmumu_procs) != 1:
        raise RuntimeError(
            f"--scetlibNPParamModel expects exactly one Zmumu process, got {zmumu_procs}"
        )
    R_info = scetlib_np_response.load_R(
        infile_path, sample_key=zmumu_procs[0], reco_axes=("ptll", "yll")
    )
    if R_info["N_gen"] is None:
        raise RuntimeError(
            "histmaker output lacks the 'prefsr' gen-total hist; rerun the "
            "histmaker with the response-matrix booking (PR #701 branch)"
        )
    writer.add_auxiliary(
        "scetlib_np",
        {
            "R": R_info["R"],
            "N_gen": R_info["N_gen"],
            "reco_axes": [n for n, _ in R_info["reco_axes"]],
            "gen_axes": [n for n, _ in R_info["gen_axes"]],
            # one edges dataset per reco/gen axis (variable length)
            **{f"edges__{n}": e for n, e in R_info["reco_axes"] + R_info["gen_axes"]},
        },
    )
    print_flush(
        f"Embedded scetlib_np auxiliary: R {R_info['R'].shape} from {zmumu_procs[0]}"
    )

# Propagate meta info (mirroring setupRabbit): the histmaker's meta_info goes
# in as meta_info_input so the ParamModel can auto-detect lambda_central
# (scetlib_np_lambda_central key) from the fit tensor's metadata.
meta = {
    "meta_info": output_tools.make_meta_info_dict(args=args, wd=common.base_dir),
    "meta_info_input": results.get("meta_info", {}),
}

# Write output
writer.write(outfolder=args.output, outfilename=args.outname, meta_data_dict=meta)
print_flush(f"Tensor written to: {args.output}/{args.outname}.hdf5")
