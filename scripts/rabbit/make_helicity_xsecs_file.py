"""Aggregate the gen-level helicity cross sections (with muR/muF scale
variations) from an mz_5TeV histmaker output into a w_z_helicity_xsecs-format
file, usable by theory_corrections.make_qcd_uncertainty_helper_by_helicity.

The "Z_lhe" key is stored as a duplicate of "Z": the pythia_shower_kt
variation derived from their difference is then exactly unity and is not
used in the fit.
"""

import argparse
import os
import sys

import h5py

wremnants_base = os.environ.get("WREM_BASE", None)
if wremnants_base is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wremnants_base = os.path.abspath(os.path.join(script_dir, "../.."))
    if os.path.exists(os.path.join(wremnants_base, "wums")):
        sys.path.insert(0, wremnants_base)

from wremnants.utilities.io_tools import base_io
from wums import ioutils

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="mz_5TeV histmaker output HDF5 file")
parser.add_argument("-o", "--output", default="./", help="output directory")
parser.add_argument(
    "--outname", default="w_z_helicity_xsecs_5TeV", help="output file name"
)
parser.add_argument(
    "--histName",
    default="nominal_gen_helicity_xsecs_scale",
    help="gen helicity xsecs histogram name",
)
parser.add_argument(
    "--procFilters",
    nargs="*",
    default=["Zmumu"],
    help="processes whose gen helicity xsecs are summed into the Z entry",
)
args = parser.parse_args()


def _sum_hists(results, hist_name):
    h_sum = None
    for proc in results:
        if proc == "meta_info" or not any(f in proc for f in args.procFilters):
            continue
        output = results[proc].get("output", {})
        if hist_name not in output:
            continue
        h_proxy = output[hist_name]
        h = h_proxy.get() if hasattr(h_proxy, "get") else h_proxy
        print(f"Adding {hist_name} from process {proc}")
        h_sum = h.copy() if h_sum is None else h_sum + h
    return h_sum


with h5py.File(os.path.abspath(args.infile), "r") as h5file:
    results = base_io.load_results_h5py(h5file)
    h_sum = _sum_hists(results, args.histName)
    h_sum_lhe = _sum_hists(results, f"{args.histName}_lhe")

if h_sum is None:
    raise RuntimeError(
        f"No histogram {args.histName} found for processes {args.procFilters}"
    )
if h_sum_lhe is None:
    print(
        "Warning: no LHE-level hist found; storing Z_lhe as a copy of Z "
        "(pythia_shower_kt variation will be unity)"
    )
    h_sum_lhe = h_sum

os.makedirs(args.output, exist_ok=True)
outpath = os.path.join(args.output, f"{args.outname}.hdf5")
with h5py.File(outpath, "w") as f:
    ioutils.pickle_dump_h5py("Z", h_sum, f)
    ioutils.pickle_dump_h5py("Z_lhe", h_sum_lhe, f)

print(f"Helicity xsecs written to: {outpath}")
