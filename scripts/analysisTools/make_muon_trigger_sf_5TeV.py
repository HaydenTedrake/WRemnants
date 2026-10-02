#!/usr/bin/env python3
"""Per-muon trigger efficiency maps for the 5.02 TeV '1HLT' step.

    mz_5TeV.py --muonEfficiencyHists      fills muonEffTrigPair_trigger
    THIS SCRIPT                           fits eps per muon, data and MC
    muon_efficiencies_5TeV.make_muon_trigger_helpers_5TeV(<output>)

Full lowPU parity. lowPU (lowpu_efficiencies.hpp::lepSF_HLT_q) applies

    corr = (1 - prod_i (1 - eps_data_i)) / (1 - prod_i (1 - eps_MC_i))

from a TAGGED tag-and-probe map, where each muon is individually known to have
fired. No muon can be tagged in this production -- TrigObj holds no muon objects
-- but that same formula read backwards EXTRACTS eps per muon from the only thing
observable, the event decision as a function of BOTH muons' bins. The fit lives
in wremnants/production/trigger_efficiency_fit_5TeV.py.

TWO THINGS THAT MAKE THIS DIFFER FROM lowPU, both forced by the above:

 * eps_data and eps_MC are written SEPARATELY, not as a ratio. The combination is
   non-linear in eps, so a per-bin SF cannot reproduce it.

 * the statistical nuisances are EIGENVECTORS of the fit covariance, not one per
   bin. Only the event decision is observed, so a single event constrains the
   product (1-eps_a)(1-eps_b): the overall (1-eps) scale is near-degenerate with
   the bin-to-bin structure (|rho| up to 0.7; pulls from the marginal diagonal
   errors come out 0.46-0.84 wide instead of 1). Per-bin independence is correct
   for lowPU's tagged map and wrong for a fitted one. Eigenvectors are the
   representation the 13 TeV SMOOTH scale factors already use.

The information budget is the DOUBLE-failure count, since only events where BOTH
muons fail are sensitive to eps: ~925 in the full 5 TeV data sample and ~98 in
1.4M MC events. That is why the binning must be checked rather than assumed --
--scanBinning reports where the fit stops being identifiable.

usage:
    make_muon_trigger_sf_5TeV.py -i <effmeas.hdf5> -o trig_5TeV.hdf5
        [--rebinEta N] [--rebinPt N] [--mergeCharge] [--scanBinning]
"""

import argparse

import h5py
import hist
import numpy as np
from wums import ioutils, logging

from wremnants.production.muon_efficiencies_5TeV import (
    KIND_DATA,
    KIND_MC,
    TRIG_EIG_HIST,
    TRIG_EPS_HIST,
    TRIG_N_ETA,
    TRIG_N_PT,
    TRIG_N_CHARGE,
    TRIG_PT_RANGE,
)
from wremnants.production.trigger_efficiency_fit_5TeV import (
    eigen_variations,
    fit_per_muon_efficiency,
)

logger = logging.setup_logger(__file__, 3, False)
PAIR_HIST = "muonEffTrigPair_trigger"


def pair_counts_from_nanoaod(n_files_data=None, n_files_mc=None, nthreads=8,
                             denom="HLT_HIL3DoubleMu0", numer="HLT_HIMu17"):
    """Count (bin1, bin2, pass) straight off the NanoAOD.

    WHY THIS EXISTS. The same counts can be booked inside the histmaker
    (muon_efficiencies_5TeV.book_trigger_pair_hist), but that booking currently
    aborts on DATA with "Incompatible underflow/overflow for histogram
    conversion" from narf's atomic fill path -- it survives MC-only runs and
    every axis set passes in isolation under threads, so it is unresolved. This
    path computes exactly the same quantity with two plain Histo2D calls and is
    what produced the validated numbers (86705 data events, 925 double failures).

    The selection is the analysis one with NO trigger requirement other than the
    orthogonal denominator, which is the whole point: the numerator is the
    trigger being measured.
    """
    import glob

    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.EnableImplicitMT(nthreads)

    base = "/scratch/submit/cms/wmass/NanoAOD"
    paths = {
        "data": sorted(glob.glob(f"{base}/LowPU/2017G/SingleMuon/"
                                 "Run2017G-UL2017_MiniAODv2_NanoAODv9_GT36-v2/*.root")),
        "mc": sorted(glob.glob(f"{base}/DYJetsToMuMu_H2ErratumFix_PDFExt_TuneCP5_"
                               "5020GeV-powhegMiNNLO-pythia8-photos/"
                               "NanoV9MC2017_TrackFitV722_NanoProdv3/*/*/*.root")),
    }
    NB = TRIG_N_ETA * TRIG_N_PT * TRIG_N_CHARGE
    out = {}
    for tag, files in paths.items():
        lim = n_files_data if tag == "data" else n_files_mc
        if lim:
            files = files[:lim]
        v = ROOT.std.vector("string")()
        for f in files:
            v.push_back(f)
        logger.info(f"{tag}: counting pairs over {len(files)} files")
        df = ROOT.RDataFrame("Events", v)
        df = df.Define("goodMu", "Muon_pt > 18 && abs(Muon_eta) < 2.4"
                       " && Muon_mediumId && Muon_isGlobal")
        df = df.Define("gidx", "ROOT::VecOps::Nonzero(goodMu)")
        df = df.Filter("gidx.size() == 2 && nElectron == 0")
        df = df.Define("i0", "int(gidx[0])").Define("i1", "int(gidx[1])")
        df = df.Filter("Muon_charge[i0]*Muon_charge[i1] < 0")
        df = df.Define(
            "dimu",
            "ROOT::Math::PtEtaPhiMVector(Muon_pt[i0],Muon_eta[i0],Muon_phi[i0],0.105658)"
            "+ROOT::Math::PtEtaPhiMVector(Muon_pt[i1],Muon_eta[i1],Muon_phi[i1],0.105658)",
        )
        df = df.Filter("dimu.M() > 76 && dimu.M() < 106").Filter(denom)
        df = df.Define("fired", f"({numer}) ? 1 : 0")
        for k, im in ((1, "i0"), (2, "i1")):
            # same flattening as muon_efficiencies_5TeV.trigger_bin_index_expr
            df = df.Define(
                f"pb{k}",
                f"((Muon_charge[{im}] > 0 ? 1 : 0) * {TRIG_N_ETA}"
                f" + std::clamp(int((Muon_eta[{im}] + 2.4f) * {TRIG_N_ETA} / 4.8f),"
                f" 0, {TRIG_N_ETA - 1})) * {TRIG_N_PT}"
                f" + std::clamp(int((Muon_pt[{im}] - {TRIG_PT_RANGE[0]}f)"
                f" * {TRIG_N_PT} / {TRIG_PT_RANGE[1] - TRIG_PT_RANGE[0]}f),"
                f" 0, {TRIG_N_PT - 1})",
            )
        hp = df.Filter("fired == 1").Histo2D(
            ROOT.RDF.TH2DModel("p", "", NB, 0, NB, NB, 0, NB), "pb1", "pb2")
        hf = df.Filter("fired == 0").Histo2D(
            ROOT.RDF.TH2DModel("f", "", NB, 0, NB, NB, 0, NB), "pb1", "pb2")
        g = lambda h: np.array(
            [[h.GetValue().GetBinContent(i + 1, j + 1) for j in range(NB)]
             for i in range(NB)])
        P, F = g(hp), g(hf)
        logger.info(f"{tag}: {P.sum() + F.sum():.0f} events, {F.sum():.0f} double failures")
        out[tag] = (P, F)
    return out


def is_data(proc):
    return proc.startswith("SingleMuon")


def read_pairs(filename, signal="Zmumu"):
    """(pass, fail) pair matrices for data and for signal MC."""
    out = {}
    with h5py.File(filename, "r") as f:
        for proc in f.keys():
            if proc == "meta_info":
                continue
            res = ioutils.pickle_load_h5py(f[proc])
            h = res.get("output", {}).get(PAIR_HIST)
            if h is None:
                continue
            h = h.get() if hasattr(h, "get") else h
            v = h.values()
            tag = "data" if is_data(proc) else ("mc" if signal in proc else None)
            if tag is None:
                continue
            P, F = v[..., 1], v[..., 0]
            if tag in out:
                out[tag] = (out[tag][0] + P, out[tag][1] + F)
            else:
                out[tag] = (P, F)
    return out


def rebin_pairs(M, ke, kp, merge_charge,
                n_eta=TRIG_N_ETA, n_pt=TRIG_N_PT, n_q=TRIG_N_CHARGE):
    """Coarsen the flattened (charge, eta, pt) index on BOTH axes of the matrix.

    The index convention is set by muon_efficiencies_5TeV.trigger_bin_index_expr:
    ((q * n_eta) + ieta) * n_pt + ipt.
    """
    if n_eta % ke or n_pt % kp:
        raise ValueError(f"rebin {ke}x{kp} does not divide {n_eta}x{n_pt} evenly")
    A = M.reshape(n_q, n_eta, n_pt, n_q, n_eta, n_pt)
    A = A.reshape(
        n_q, n_eta // ke, ke, n_pt // kp, kp, n_q, n_eta // ke, ke, n_pt // kp, kp
    ).sum((2, 4, 7, 9))
    if merge_charge:
        A = A.sum(axis=(0, 3))
        n = A.shape[0] * A.shape[1]
    else:
        n = A.shape[0] * A.shape[1] * A.shape[2]
    return A.reshape(n, n)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-i", "--input", default=None,
        help="histmaker output holding muonEffTrigPair_trigger. Omit to count "
        "straight off the NanoAOD instead (--fromNanoAOD), which is the route "
        "that currently works -- see pair_counts_from_nanoaod.",
    )
    p.add_argument(
        "--fromNanoAOD", action="store_true",
        help="compute the pair counts directly from the NanoAOD rather than "
        "reading them from a histmaker file",
    )
    p.add_argument("--nFilesData", type=int, default=None)
    p.add_argument("--nFilesMC", type=int, default=None)
    p.add_argument("-j", "--nThreads", type=int, default=8)
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--signal", default="Zmumu")
    p.add_argument("--rebinEta", type=int, default=1)
    p.add_argument("--rebinPt", type=int, default=1)
    p.add_argument(
        "--mergeCharge",
        action="store_true",
        help="one charge-inclusive map. Recommended: the charge-split fit leaves "
        "bins pinned at a bound on the 5 TeV data volume.",
    )
    p.add_argument(
        "--scanBinning",
        action="store_true",
        help="fit several binnings and report identifiability instead of writing",
    )
    args = p.parse_args()
    if not args.scanBinning and not args.output:
        p.error("-o/--output is required unless --scanBinning")
    if not args.input and not args.fromNanoAOD:
        args.fromNanoAOD = True

    if args.fromNanoAOD or not args.input:
        pairs = pair_counts_from_nanoaod(
            args.nFilesData, args.nFilesMC, args.nThreads)
    else:
        pairs = read_pairs(args.input, args.signal)
    for need in ("data", "mc"):
        if need not in pairs:
            raise SystemExit(
                f"no {need} entry for the trigger pair counts."
                " With -i, the histmaker must have run with"
                " --muonEfficiencyHists; otherwise use --fromNanoAOD."
            )

    if args.scanBinning:
        print(f"{'binning':22s} {'nbins':>6} {'dblFail':>9} {'bound':>6} "
              f"{'ident':>6} {'medSig':>8}")
        for ke, kp, mq in ((1, 1, False), (1, 1, True), (2, 2, False), (2, 2, True),
                           (3, 5, True), (4, 5, True)):
            P = rebin_pairs(pairs["data"][0], ke, kp, mq)
            F = rebin_pairs(pairs["data"][1], ke, kp, mq)
            eps, sig, info = fit_per_muon_efficiency(P, F, verbose=False)
            free = ~(info["empty"] | info["at_bound"])
            lbl = (f"{TRIG_N_ETA // ke}eta x {TRIG_N_PT // kp}pt x"
                   f" {1 if mq else TRIG_N_CHARGE}q")
            print(f"  {lbl:22s} {P.shape[0]:6d} {info['n_double_fail']:9.0f}"
                  f" {int(info['at_bound'].sum()):6d} {str(info['identifiable']):>6s}"
                  f" {np.median(sig[free]) if free.sum() else float('nan'):8.4f}")
        return

    ke, kp, mq = args.rebinEta, args.rebinPt, args.mergeCharge
    n_eta, n_pt = TRIG_N_ETA // ke, TRIG_N_PT // kp
    n_q = 1 if mq else TRIG_N_CHARGE

    fits = {}
    for tag in ("data", "mc"):
        P = rebin_pairs(pairs[tag][0], ke, kp, mq)
        F = rebin_pairs(pairs[tag][1], ke, kp, mq)
        logger.info(
            f"{tag}: {P.sum() + F.sum():.0f} events, {F.sum():.0f} double failures,"
            f" {P.shape[0]} per-muon bins"
        )
        eps, sig, info = fit_per_muon_efficiency(P, F)
        if int(info["at_bound"].sum()):
            logger.warning(
                f"{tag}: {int(info['at_bound'].sum())} bins pinned at a bound (no"
                " double failures there, so eps is not determined). They are held"
                " at the fitted value with zero uncertainty; coarsen the binning"
                " if there are many."
            )
        shifts, evals = eigen_variations(info["cov"])
        tot = evals.sum() if len(evals) else 1.0
        logger.info(
            f"{tag}: {len(evals)} eigen modes, leading one carries"
            f" {100 * evals[0] / tot:.1f}%" if len(evals) else f"{tag}: no modes"
        )
        fits[tag] = (eps, sig, shifts)

    n_modes = max(len(fits["data"][2]), len(fits["mc"][2]))
    axes_eps = [
        hist.axis.Regular(n_eta, -2.4, 2.4, name="eta"),
        hist.axis.Regular(n_pt, *TRIG_PT_RANGE, name="pt"),
        hist.axis.Regular(n_q, -2.0, 2.0, underflow=False, overflow=False,
                          name="charge"),
        hist.axis.Integer(0, 2, underflow=False, overflow=False, name="kind"),
    ]
    h_eps = hist.Hist(*axes_eps, storage=hist.storage.Weight())
    h_eig = hist.Hist(
        *axes_eps,
        hist.axis.Integer(0, n_modes, underflow=False, overflow=False, name="mode"),
        storage=hist.storage.Weight(),
    )

    def unflatten(a):
        """flattened (q, eta, pt) -> (eta, pt, charge), the map's axis order"""
        return a.reshape(n_q, n_eta, n_pt).transpose(1, 2, 0)

    ve, vg = h_eps.view(flow=True), h_eig.view(flow=True)
    for tag, kind in (("data", KIND_DATA), ("mc", KIND_MC)):
        eps, sig, shifts = fits[tag]
        ve.value[1:-1, 1:-1, :, kind] = unflatten(eps)
        ve.variance[1:-1, 1:-1, :, kind] = unflatten(sig**2)
        for k in range(n_modes):
            row = shifts[k] if k < len(shifts) else np.zeros_like(eps)
            vg.value[1:-1, 1:-1, :, kind, k] = unflatten(row)
    # eta and pt flow bins take the adjacent value, as in the counted maps
    for v in (ve, vg):
        v[0, ...] = v[1, ...]
        v[-1, ...] = v[-2, ...]
        v[:, 0, ...] = v[:, 1, ...]
        v[:, -1, ...] = v[:, -2, ...]

    with h5py.File(args.output, "w") as f:
        ioutils.pickle_dump_h5py(TRIG_EPS_HIST, h_eps, f)
        ioutils.pickle_dump_h5py(TRIG_EIG_HIST, h_eig, f)
        ioutils.pickle_dump_h5py(
            "meta_info",
            dict(input=args.input, rebinEta=ke, rebinPt=kp, mergeCharge=mq,
                 nModes=n_modes, perMuon=True,
                 eps_data_mean=float(fits["data"][0].mean()),
                 eps_mc_mean=float(fits["mc"][0].mean())),
            f,
        )
    logger.info(
        f"wrote {args.output}: eps_data mean {fits['data'][0].mean():.4f},"
        f" eps_MC mean {fits['mc'][0].mean():.4f}, {n_modes} eigen modes"
    )


if __name__ == "__main__":
    main()
