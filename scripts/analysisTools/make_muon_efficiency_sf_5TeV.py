#!/usr/bin/env python3
"""Turn the 5.02 TeV pass/fail histograms into a muon SF map.

    mz_5TeV.py --muonEfficiencyHists      fills muonEff_<step>
    THIS SCRIPT                           SF = eps_data / eps_MC
    muon_efficiencies_5TeV.make_muon_efficiency_helpers_5TeV(<output>)

Same division of labour as David's CVH measurement (mz_dilepton.py
--cvhEfficiencyHists -> make_cvh_efficiency_sf.py): the histmaker only counts,
every efficiency and every uncertainty is formed here, where it can be inspected
and varied without re-running over the NanoAOD.

WHAT THE EFFICIENCY IS. Each muonEff_<step> histogram is filled once per probe
muon over (pt, eta, [phi], charge, pass). eps is the pass fraction of that bin.
Both muons of the Z candidate are probes, no tag/probe split, because the
measured steps are uncorrelated with the trigger -- see the module docstring of
wremnants/production/muon_efficiencies_5TeV.py.

UNCERTAINTIES. The counts are weighted, so the binomial variance is the weighted
form
    Var(eps) = [ (1-eps)^2 * sumw2_pass + eps^2 * sumw2_fail ] / N^2,
which reduces to eps(1-eps)/N for unit weights, and the ratio adds in quadrature
    Var(SF)/SF^2 = Var(eps_data)/eps_data^2 + Var(eps_MC)/eps_MC^2.
That variance is what the effStatTnP nuisances are built from, one per (eta, pt,
charge) bin, in muon_efficiencies_5TeV.hpp.

BACKGROUNDS. eps_data is measured on data with the MC prediction for every
non-signal process subtracted bin by bin, pass and fail separately (--noBkgSub
turns it off). At 0.3/fb in a 76-106 GeV opposite-sign dimuon window the
subtraction is small, but it is not zero and it is not the same in the pass and
the fail bin, which is exactly the shape a cut-and-count efficiency is sensitive
to.

THE ALTERNATIVE MAP (slot 1 of the nom-alt axis) is what the effSystTnP
nuisances are built from: one fully correlated variation per step, as opposed to
the per-bin statistical ones. Written only when asked for, because a
systematic that equals the nominal is a degenerate fit direction, not a small
uncertainty, and the helper factory refuses to build one. Two ways to fill it:

  --altBkgVar X   recompute eps_data with the subtracted background scaled by
                  1+X. The background normalisation systematic; set X from the
                  normalisation uncertainty of the backgrounds, not by taste.
  --altFrom FILE  take the alternative from a second, independent measurement:
                  a different probe definition, a different mass window, or the
                  same measurement on a different data period. This is the
                  closer analogue of the 13 TeV 'originalDataAltSig', which is
                  the alternative-signal-model fit of the tag-and-probe pass and
                  fail distributions -- there is no such fit here, the
                  efficiency is counted rather than fitted, so the alternative
                  has to come from varying the counting instead.

usage:
    make_muon_efficiency_sf_5TeV.py -i <histmaker.hdf5> -o <sf.hdf5>
        [--steps id,cvh] [--noBkgSub] [--altBkgVar 0.5] [--altFrom other.hdf5]
        [--signal Zmumu] [--chargeInclusive/--chargeSplit]
"""

import argparse

import h5py
import hist
import numpy as np
from wums import ioutils, logging

from wremnants.production.muon_efficiencies_5TeV import (
    IDX_ALT,
    IDX_NOM,
    SF_AXES,
    SF_EVENT_AXES,
    SF_HIST_NAME,
)

logger = logging.setup_logger(__file__, 3, False)

# A step that is the PRODUCT of others. The nominal helper multiplies every step
# in the map, so a merged step must never share a map with its own components.
MERGED_STEPS = {"id": ("reco", "idip")}
# preferred default when the measurement offers both the merged and the split form
FACTORISED_DEFAULT = ("reco", "idip")


def read_measurement(filename, steps=None, n_pt=1, n_eta=1):
    """{step: {process: hist}} from a histmaker output.

    Each hist is the raw muonEff_<step> with axes (pt, eta, [phi], charge,
    pass). Read inside the open file: these are lazy H5PickleProxy objects and
    resolving one after the file closes raises "underlying file has been
    closed".
    """
    out = {}
    with h5py.File(filename, "r") as f:
        for proc in f.keys():
            if proc == "meta_info":
                continue
            res = ioutils.pickle_load_h5py(f[proc])
            for name, h in res.get("output", {}).items():
                if not name.startswith("muonEff_"):
                    continue
                step = name[len("muonEff_") :]
                if steps and step not in steps:
                    continue
                h = h.get() if hasattr(h, "get") else h
                out.setdefault(step, {})[proc] = rebin(h, n_pt, n_eta)
    return out


def read_event_measurement(filename, steps=None):
    """{step: {process: hist}} for the EVENT-level muonEffEvent_* histograms.

    Not rebinned: an event-level map is already binned in the fit variables and
    deliberately coarse there (see build_event_axes).
    """
    out = {}
    with h5py.File(filename, "r") as f:
        for proc in f.keys():
            if proc == "meta_info":
                continue
            res = ioutils.pickle_load_h5py(f[proc])
            for name, h in res.get("output", {}).items():
                if not name.startswith("muonEffEvent_"):
                    continue
                step = name[len("muonEffEvent_") :]
                if steps and step not in steps:
                    continue
                out.setdefault(step, {})[proc] = h.get() if hasattr(h, "get") else h
    return out


def collapse_event(h):
    """(yll, ptll, pass) -> values and variances with a dummy charge axis.

    The event map reuses the five-slot per-muon layout so the C++ base class and
    the SF file format stay identical; charge is a single bin that is never read.
    """
    names = [a.name for a in h.axes]
    order = [names.index(n) for n in ("yll", "ptll", "pass")]
    v = h.values(flow=False).transpose(order)
    w2 = h.variances(flow=False).transpose(order)
    # insert the dummy charge axis before 'pass'
    return v[:, :, None, :], w2[:, :, None, :]


def build_event_axes_sf(ref_hist, steps):
    """SF map axes for an event-level measurement, in the contractual order."""
    names = [a.name for a in ref_hist.axes]
    axes = [
        hist.axis.Variable(ref_hist.axes[names.index("yll")].edges, name="yll"),
        hist.axis.Variable(ref_hist.axes[names.index("ptll")].edges, name="ptll"),
        hist.axis.Regular(1, -2.0, 2.0, underflow=False, overflow=False, name="charge"),
        hist.axis.StrCategory(steps, name="step"),
        hist.axis.Integer(0, 2, underflow=False, overflow=False, name="nom-alt"),
    ]
    assert tuple(a.name for a in axes) == SF_EVENT_AXES, [a.name for a in axes]
    return axes


def is_data(proc):
    """Data process names in the 5 TeV production: SingleMuon_2017G (and the
    2017H low-PU equivalent, which must never be mixed in here but is recognised
    so it can be reported)."""
    return proc.startswith("SingleMuon")



# Per-muon ID counting straight off the NanoAOD, bypassing the histmaker.
#
# WHY: effSystTnP needs an ALTERNATIVE measurement of the same efficiency, and
# the defensible one here is the probe-selection envelope -- reselect the probes
# with a trigger whose offline-quality correlation is weaker and see how far the
# SF moves. Doing that through the histmaker means a second full measurement run;
# counting here needs none, and -- more importantly -- it puts the nominal and the
# alternative through ONE IDENTICAL code path, so the probe trigger is the only
# difference between them. Two separate histmaker runs would differ in whatever
# else changed in between.
#
# Measured envelope on the full sample (inclusive SF): HIMu17 0.9940,
# HIL3Mu12 0.9941, HIL2Mu7 0.9934, HIL1Mu12 0.9934, HIL1DoubleMu0 0.9945 -- at
# most 0.056%, correlated across bins, and real.

ID_STEPS_NANOAOD = {
    "reco": ("Muon_looseId", "Muon_isGlobal"),
    "idip": ("Muon_looseId && Muon_isGlobal", "Muon_mediumId"),
}


def id_counts_from_nanoaod(probe_trigger, n_eta, n_pt, n_q, nthreads=8,
                           n_files_data=None, n_files_mc=None):
    """{step: {'data'|'mc': (npass, nfail)}} on a flattened (q, eta, pt) grid.

    probe_trigger is the EVENT filter defining the probe sample. Everything else
    is the analysis Z selection with the muon ID relaxed to loose, as
    mz_5TeV.py --muonEfficiencyHists does.
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
    NB = n_eta * n_pt * n_q
    out = {st: {} for st in ID_STEPS_NANOAOD}
    for tag, files in paths.items():
        lim = n_files_data if tag == "data" else n_files_mc
        if lim:
            files = files[:lim]
        v = ROOT.std.vector("string")()
        for f in files:
            v.push_back(f)
        base_df = ROOT.RDataFrame("Events", v).Filter(probe_trigger)
        for st, (probe_sel, pass_req) in ID_STEPS_NANOAOD.items():
            df = base_df.Define(
                "probe", f"Muon_pt > 18 && abs(Muon_eta) < 2.4 && {probe_sel}")
            df = df.Define("pidx", "ROOT::VecOps::Nonzero(probe)")
            df = df.Filter("pidx.size() == 2 && nElectron == 0")
            df = df.Define("j0", "int(pidx[0])").Define("j1", "int(pidx[1])")
            df = df.Filter("Muon_charge[j0]*Muon_charge[j1] < 0")
            df = df.Define(
                "dimu",
                "ROOT::Math::PtEtaPhiMVector(Muon_pt[j0],Muon_eta[j0],Muon_phi[j0],0.105658)"
                "+ROOT::Math::PtEtaPhiMVector(Muon_pt[j1],Muon_eta[j1],Muon_phi[j1],0.105658)")
            df = df.Filter("dimu.M() > 76 && dimu.M() < 106")
            for k, im in ((1, "j0"), (2, "j1")):
                qexpr = f"(Muon_charge[{im}] > 0 ? 1 : 0)" if n_q == 2 else "0"
                df = df.Define(
                    f"ib{k}",
                    f"(({qexpr}) * {n_eta}"
                    f" + std::clamp(int((Muon_eta[{im}] + 2.4f) * {n_eta} / 4.8f),"
                    f" 0, {n_eta - 1})) * {n_pt}"
                    f" + std::clamp(int((Muon_pt[{im}] - 18.f) * {n_pt} / 47.f),"
                    f" 0, {n_pt - 1})")
                df = df.Define(f"ip{k}", f"({pass_req}[{im}]) ? 1 : 0")
            P = np.zeros(NB)
            F = np.zeros(NB)
            for k in (1, 2):
                hp = df.Filter(f"ip{k} == 1").Histo1D(
                    ROOT.RDF.TH1DModel(f"h{k}", "", NB, 0, NB), f"ib{k}")
                hf = df.Filter(f"ip{k} == 0").Histo1D(
                    ROOT.RDF.TH1DModel(f"g{k}", "", NB, 0, NB), f"ib{k}")
                P += np.array([hp.GetValue().GetBinContent(i+1) for i in range(NB)])
                F += np.array([hf.GetValue().GetBinContent(i+1) for i in range(NB)])
            out[st][tag] = (P, F)
            tot = P.sum() + F.sum()
            logger.info(f"{st} {tag} [{probe_trigger}]: {tot:.0f} probes,"
                        f" eff {P.sum()/max(tot,1):.4f}")
    return out


def rebin(h, n_pt, n_eta):
    """Merge adjacent pt and/or eta bins of a raw muonEff_<step> histogram.

    Summing counts is exactly equivalent to having filled the coarse binning in
    the first place, so this is a free way to change the SF map granularity. A
    factor that does not divide the axis evenly is refused rather than silently
    dropping the remainder, which would quietly bias the last bin.
    """
    for name, n in (("pt", n_pt), ("eta", n_eta)):
        if n <= 1:
            continue
        size = h.axes[name].size
        if size % n:
            raise ValueError(
                f"--rebin{name.capitalize()} {n} does not divide the {size}"
                f" {name} bins of the measurement evenly"
            )
        h = h[{name: slice(None, None, hist.rebin(n))}]
    return h


def collapse(h, charge_inclusive):
    """(pt, eta, [phi], charge, pass) -> (eta, pt, charge, pass) arrays.

    phi is integrated over: it is filled only for the CVH step, to localise
    module-level structure, and the SF map is 2D in (eta, pt) by construction.
    charge is summed when the map is charge-inclusive, which is the expected
    configuration at 0.3/fb -- splitting the charges halves the probes per bin
    and the ID efficiency has no charge dependence to resolve.

    Returns (values, variances) with the axis order the SF map needs.
    """
    names = [a.name for a in h.axes]
    v = h.values(flow=False)
    w2 = h.variances(flow=False)
    if "phi" in names:
        i = names.index("phi")
        v, w2 = v.sum(axis=i), w2.sum(axis=i)
        names.pop(i)
    # (pt, eta, charge, pass) -> (eta, pt, charge, pass)
    order = [names.index(n) for n in ("eta", "pt", "charge", "pass")]
    v, w2 = v.transpose(order), w2.transpose(order)
    if charge_inclusive:
        v = v.sum(axis=2, keepdims=True)
        w2 = w2.sum(axis=2, keepdims=True)
    return v, w2


def efficiency(v, w2):
    """Weighted pass fraction and its variance, per (eta, pt, charge) bin.

    v, w2 have a trailing length-2 'pass' axis: index 0 fail, 1 pass.
    """
    npass, nfail = v[..., 1], v[..., 0]
    vpass, vfail = w2[..., 1], w2[..., 0]
    tot = npass + nfail
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = np.where(tot > 0, npass / tot, 0.0)
        var = np.where(
            tot > 0,
            ((1.0 - eps) ** 2 * vpass + eps**2 * vfail) / np.where(tot > 0, tot**2, 1.0),
            0.0,
        )
    return eps, var


def data_counts(per_proc, signal, charge_inclusive, bkg_scale):
    """(data - bkg_scale * background MC) counts and variances.

    The subtraction is done on the pass and fail COUNTS, not on the efficiency:
    the background pass fraction differs from the signal one, which is the whole
    reason it matters.
    """
    vd = w2d = None
    vb = w2b = None
    for proc, h in per_proc.items():
        v, w2 = collapse(h, charge_inclusive)
        if is_data(proc):
            vd = v if vd is None else vd + v
            w2d = w2 if w2d is None else w2d + w2
        elif signal not in proc:
            vb = v if vb is None else vb + v
            w2b = w2 if w2b is None else w2b + w2
    if vd is None:
        raise RuntimeError("no data process found (expected SingleMuon*)")
    if vb is None or bkg_scale == 0.0:
        return vd, w2d
    # the background prediction's own MC statistical error enters the data
    # efficiency error, so it is added rather than dropped
    return vd - bkg_scale * vb, w2d + (bkg_scale**2) * w2b


def mc_counts(per_proc, signal, charge_inclusive):
    v = w2 = None
    for proc, h in per_proc.items():
        if is_data(proc) or signal not in proc:
            continue
        a, b = collapse(h, charge_inclusive)
        v = a if v is None else v + a
        w2 = b if w2 is None else w2 + b
    if v is None:
        raise RuntimeError(f"no signal MC process matching '{signal}'")
    return v, w2


def sf_and_variance(eps_d, var_d, eps_m, var_m):
    """SF = eps_data/eps_MC with the two relative variances added.

    A bin with no probes in data or in MC gets SF = 1 with zero variance: it
    carries no measurement, and 1 is what the nominal lookup should return for a
    muon that lands there. It also gets no stat nuisance, because the helper
    skips cells with a non-positive value or variance.
    """
    ok = (eps_d > 0) & (eps_m > 0)
    sf = np.where(ok, np.divide(eps_d, eps_m, out=np.ones_like(eps_d), where=ok), 1.0)
    rel = np.where(
        ok,
        np.divide(var_d, np.where(ok, eps_d**2, 1.0), out=np.zeros_like(var_d), where=ok)
        + np.divide(
            var_m, np.where(ok, eps_m**2, 1.0), out=np.zeros_like(var_m), where=ok
        ),
        0.0,
    )
    return sf, sf**2 * rel


def build_axes(ref_hist, steps, charge_inclusive):
    """The SF map axes, in the order muon_efficiencies_5TeV.hpp expects."""
    names = [a.name for a in ref_hist.axes]
    axis_eta = ref_hist.axes[names.index("eta")]
    axis_pt = ref_hist.axes[names.index("pt")]
    axis_charge = (
        hist.axis.Regular(
            1, -2.0, 2.0, underflow=False, overflow=False, name="charge"
        )
        if charge_inclusive
        else ref_hist.axes[names.index("charge")]
    )
    axes = [
        hist.axis.Variable(axis_eta.edges, name="eta"),
        hist.axis.Variable(axis_pt.edges, name="pt"),
        axis_charge,
        hist.axis.StrCategory(steps, name="step"),
        hist.axis.Integer(0, 2, underflow=False, overflow=False, name="nom-alt"),
    ]
    assert tuple(a.name for a in axes) == SF_AXES, [a.name for a in axes]
    return axes


def extend_to_flow(view):
    """Copy the edge bins into the eta and pt flow bins.

    A muon just outside the measured range must get the SF of the nearest
    measured bin, not the 0 an empty flow bin would hand it -- the nominal
    lookup passes the raw axis index straight to boost::histogram::at(), which
    honours the flow bins. Same treatment as muon_efficiencies_binned.py.
    """
    view[0, ...] = view[1, ...]
    view[-1, ...] = view[-2, ...]
    view[:, 0, ...] = view[:, 1, ...]
    view[:, -1, ...] = view[:, -2, ...]


def build_from_nanoaod(args):
    """SF maps counted directly off the NanoAOD, with an optional alternative.

    Writes the same file format the histmaker route produces, so
    make_muon_efficiency_helpers_5TeV reads it unchanged. When
    --altProbeTrigger is given, slot 1 of the nom-alt axis holds the SF measured
    with that probe selection and the factory will build effSystTnP from it.
    """
    n_eta, n_pt = args.nEta, args.nPt
    n_q = 2 if args.chargeSplit else 1

    def maps(trigger):
        counts = id_counts_from_nanoaod(
            trigger, n_eta, n_pt, n_q, args.nThreads,
            args.nFilesData, args.nFilesMC)
        res = {}
        for st, per_tag in counts.items():
            Pd, Fd = per_tag["data"]
            Pm, Fm = per_tag["mc"]
            # Poisson errors on the counts; no MC-stat weights here because the
            # NanoAOD path counts raw entries
            ed, vard = Pd / np.maximum(Pd + Fd, 1), None
            tot_d, tot_m = Pd + Fd, Pm + Fm
            ed = np.where(tot_d > 0, Pd / np.maximum(tot_d, 1), 0.0)
            em = np.where(tot_m > 0, Pm / np.maximum(tot_m, 1), 0.0)
            vard = np.where(tot_d > 0, ed * (1 - ed) / np.maximum(tot_d, 1), 0.0)
            varm = np.where(tot_m > 0, em * (1 - em) / np.maximum(tot_m, 1), 0.0)
            res[st] = sf_and_variance(ed, vard, em, varm)
        return res

    nom = maps(args.probeTrigger)
    alt = maps(args.altProbeTrigger) if args.altProbeTrigger else nom
    if args.altProbeTrigger:
        for st in nom:
            d = 100 * np.abs(alt[st][0] - nom[st][0])
            ok = nom[st][0] > 0
            logger.info(
                f"{st}: probe-selection shift (= effSystTnP) median"
                f" {np.median(d[ok]):.3f}%, max {d[ok].max():.3f}%")

    steps = sorted(nom)
    axes = [
        hist.axis.Regular(n_eta, -2.4, 2.4, name="eta"),
        hist.axis.Regular(n_pt, 18.0, 65.0, name="pt"),
        hist.axis.Regular(n_q, -2.0, 2.0, underflow=False, overflow=False,
                          name="charge"),
        hist.axis.StrCategory(steps, name="step"),
        hist.axis.Integer(0, 2, underflow=False, overflow=False, name="nom-alt"),
    ]
    h_sf = hist.Hist(*axes, storage=hist.storage.Weight())
    view = h_sf.view(flow=True)

    def unflatten(a):
        return a.reshape(n_q, n_eta, n_pt).transpose(1, 2, 0)

    for i, st in enumerate(steps):
        view.value[1:-1, 1:-1, :, i, IDX_NOM] = unflatten(nom[st][0])
        view.variance[1:-1, 1:-1, :, i, IDX_NOM] = unflatten(nom[st][1])
        view.value[1:-1, 1:-1, :, i, IDX_ALT] = unflatten(alt[st][0])
        view.variance[1:-1, 1:-1, :, i, IDX_ALT] = unflatten(alt[st][1])
    extend_to_flow(view)

    with h5py.File(args.output, "w") as f:
        ioutils.pickle_dump_h5py(SF_HIST_NAME, h_sf, f)
        ioutils.pickle_dump_h5py(
            "meta_info",
            dict(fromNanoAOD=True, probeTrigger=args.probeTrigger,
                 altProbeTrigger=args.altProbeTrigger, steps=steps,
                 nEta=n_eta, nPt=n_pt, nCharge=n_q,
                 hasAlt=args.altProbeTrigger is not None),
            f,
        )
    logger.info(f"wrote {args.output}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-i", "--input", default=None,
        help="mz_5TeV.py --muonEfficiencyHists output. Not needed with "
        "--fromNanoAOD, which counts the probes itself.")
    p.add_argument("-o", "--output", required=True)
    p.add_argument(
        "--steps",
        default=None,
        help="comma separated subset. Default is the FACTORISED set (reco,idip) "
        "when the measurement provides it, because that is 13 TeV parity; the "
        "merged 'id' step is then left out, since applying it alongside its own "
        "components would correct the ID twice.",
    )
    p.add_argument("--signal", default="Zmumu", help="substring identifying the signal MC")
    p.add_argument("--noBkgSub", action="store_true", help="do not subtract background MC from data")
    p.add_argument(
        "--rebinPt",
        type=int,
        default=1,
        help="merge this many adjacent pt bins of the measurement before forming "
        "the SF. The measurement is filled fine (24 eta x 10 pt) and coarsened "
        "HERE, so changing the SF map binning never needs a new histmaker pass. "
        "Efficiency is flat in pt (0.9806-0.9847 over the 10 bins) but has real "
        "eta structure, so --rebinPt 5 is the sensible choice and --rebinEta is "
        "usually not.",
    )
    p.add_argument(
        "--rebinEta",
        type=int,
        default=1,
        help="merge this many adjacent eta bins. Use with care: eta carries real "
        "detector structure (dips to 0.9425 at |eta| 0.2-0.4, 0.973 at 0.8-1.0, "
        "0.963 at 2.0-2.2) that coarsening washes out.",
    )
    p.add_argument(
        "--chargeSplit",
        action="store_true",
        help="keep the two charges independent (default: charge inclusive)",
    )
    p.add_argument(
        "--fromNanoAOD",
        action="store_true",
        help="count the ID pass/fail straight off the NanoAOD instead of reading "
        "a histmaker file. Needed for --altProbeTrigger, and the route that puts "
        "nominal and alternative through one identical code path.",
    )
    p.add_argument(
        "--altProbeTrigger",
        default=None,
        help="with --fromNanoAOD: a second probe-selection trigger (e.g. "
        "'HLT_HIL1DoubleMu0'). Its SF becomes the alternative map, i.e. the "
        "effSystTnP input -- the probe-selection systematic.",
    )
    p.add_argument(
        "--probeTrigger",
        default="HLT_HIMu17 || HLT_HIL3DoubleMu0",
        help="nominal probe-selection trigger for --fromNanoAOD",
    )
    p.add_argument("--nEta", type=int, default=48)
    p.add_argument("--nPt", type=int, default=2)
    p.add_argument("-j", "--nThreads", type=int, default=8)
    p.add_argument("--nFilesData", type=int, default=None)
    p.add_argument("--nFilesMC", type=int, default=None)
    p.add_argument(
        "--eventOutput",
        default=None,
        help="also write an EVENT-level SF map (the '1HLT' trigger step) from the "
        "muonEffEvent_* histograms, to this path. Kept a separate file because it "
        "is one factor per event rather than one per muon and is binned in the fit "
        "variables, so it needs its own helper (see muon_efficiencies_5TeV.hpp).",
    )
    p.add_argument(
        "--altBkgVar",
        type=float,
        default=None,
        help="alternative map: background scaled by 1+X in the data efficiency",
    )
    p.add_argument(
        "--altFrom",
        default=None,
        help="alternative map: a second measurement file, used as-is",
    )
    args = p.parse_args()

    if args.altBkgVar is not None and args.altFrom is not None:
        p.error("--altBkgVar and --altFrom are two ways to fill the same slot")
    if args.altBkgVar is not None and args.noBkgSub:
        p.error("--altBkgVar varies the background subtraction that --noBkgSub removes")

    if args.fromNanoAOD:
        build_from_nanoaod(args)
        return
    if not args.input:
        p.error("-i/--input is required unless --fromNanoAOD")

    wanted = args.steps.split(",") if args.steps else None
    meas = read_measurement(args.input, wanted, args.rebinPt, args.rebinEta)
    if not meas:
        raise SystemExit(f"no muonEff_* histograms in {args.input}")

    # A step measured in MC only is not measurable at all: an SF is a ratio of
    # two efficiencies. That is the actual state of the CVH step at 5 TeV -- the
    # data NanoAOD carries no refit branches, so muon_efficiencies_5TeV.
    # available_steps drops it on data and it arrives here with MC alone.
    for step in sorted(meas):
        if not any(is_data(pr) for pr in meas[step]):
            logger.warning(
                f"Dropping step '{step}': no data process among {sorted(meas[step])},"
                " so eps_data/eps_MC cannot be formed"
            )
            del meas[step]
        elif not any(
            not is_data(pr) and args.signal in pr for pr in meas[step]
        ):
            logger.warning(
                f"Dropping step '{step}': no signal MC matching '{args.signal}'"
                f" among {sorted(meas[step])}"
            )
            del meas[step]
    if not meas:
        raise SystemExit("no step has both data and signal MC")

    if wanted is None:
        # prefer the factorised form over the merged one
        for merged, parts in MERGED_STEPS.items():
            if merged in meas and all(pt in meas for pt in parts):
                logger.info(
                    f"Dropping merged step '{merged}' in favour of its factorised"
                    f" components {list(parts)} (13 TeV parity); pass"
                    f" --steps {merged} to use the merged form instead"
                )
                del meas[merged]
    else:
        # explicit request: refuse a combination that would double count
        for merged, parts in MERGED_STEPS.items():
            clash = [pt for pt in parts if pt in meas]
            if merged in meas and clash:
                raise SystemExit(
                    f"--steps asks for '{merged}' together with {clash}, but"
                    f" '{merged}' IS the product of {list(parts)}. Every step in"
                    " the map is multiplied into the event weight, so this would"
                    " apply the correction twice. Pick one form."
                )
    steps = sorted(meas)
    charge_inclusive = not args.chargeSplit
    bkg_scale = 0.0 if args.noBkgSub else 1.0

    alt_meas = (
        read_measurement(args.altFrom, wanted, args.rebinPt, args.rebinEta)
        if args.altFrom
        else None
    )

    ref = next(iter(meas[steps[0]].values()))
    h_sf = hist.Hist(
        *build_axes(ref, steps, charge_inclusive), storage=hist.storage.Weight()
    )
    view = h_sf.view(flow=True)
    istep_of = {s: i for i, s in enumerate(steps)}

    for step in steps:
        per_proc = meas[step]
        procs = sorted(per_proc)
        logger.info(f"step '{step}': {procs}")

        def measure(source, scale):
            vd, w2d = data_counts(source, args.signal, charge_inclusive, scale)
            vm, w2m = mc_counts(source, args.signal, charge_inclusive)
            eps_d, var_d = efficiency(vd, w2d)
            eps_m, var_m = efficiency(vm, w2m)
            return (*sf_and_variance(eps_d, var_d, eps_m, var_m), eps_d, eps_m)

        sf, var, eps_d, eps_m = measure(per_proc, bkg_scale)
        ok = (eps_d > 0) & (eps_m > 0)
        logger.info(
            f"  eps_data {eps_d[ok].mean():.4f}  eps_MC {eps_m[ok].mean():.4f}"
            f"  SF {sf[ok].mean():.4f} +- {np.sqrt(var[ok]).mean():.4f}"
            f"  ({ok.sum()}/{ok.size} bins measured)"
        )

        if alt_meas is not None:
            sf_alt, var_alt = measure(alt_meas[step], bkg_scale)[:2]
        elif args.altBkgVar is not None:
            sf_alt, var_alt = measure(per_proc, bkg_scale * (1.0 + args.altBkgVar))[:2]
        else:
            sf_alt, var_alt = sf, var  # factory then refuses to build effSystTnP

        i = istep_of[step]
        # [1:-1, 1:-1] is the non-flow part of the eta and pt axes; charge, step
        # and nom-alt were built without flow bins
        view.value[1:-1, 1:-1, :, i, IDX_NOM] = sf
        view.variance[1:-1, 1:-1, :, i, IDX_NOM] = var
        view.value[1:-1, 1:-1, :, i, IDX_ALT] = sf_alt
        view.variance[1:-1, 1:-1, :, i, IDX_ALT] = var_alt

    extend_to_flow(view)

    with h5py.File(args.output, "w") as f:
        ioutils.pickle_dump_h5py(SF_HIST_NAME, h_sf, f)
        ioutils.pickle_dump_h5py(
            "meta_info",
            dict(
                input=args.input,
                steps=steps,
                signal=args.signal,
                bkgSubtracted=not args.noBkgSub,
                chargeInclusive=charge_inclusive,
                rebinPt=args.rebinPt,
                rebinEta=args.rebinEta,
                altBkgVar=args.altBkgVar,
                altFrom=args.altFrom,
                hasAlt=args.altBkgVar is not None or args.altFrom is not None,
            ),
            f,
        )
    logger.info(f"wrote {args.output}")

    if args.eventOutput:
        write_event_map(args)


def write_event_map(args):
    """SF = eps_data/eps_MC for the event-level steps, same arithmetic as the
    per-muon map but one entry per event and no charge or eta/pt structure."""
    meas = read_event_measurement(args.input, args.steps.split(",") if args.steps else None)
    if not meas:
        logger.warning(
            f"no muonEffEvent_* histograms in {args.input}; nothing written to"
            f" {args.eventOutput}. Did the histmaker run with --muonEfficiencyHists?"
        )
        return
    for step in sorted(meas):
        if not any(is_data(pr) for pr in meas[step]):
            logger.warning(f"Dropping event step '{step}': no data process")
            del meas[step]
    if not meas:
        return
    steps = sorted(meas)
    ref = next(iter(meas[steps[0]].values()))
    h_sf = hist.Hist(*build_event_axes_sf(ref, steps), storage=hist.storage.Weight())
    view = h_sf.view(flow=True)
    bkg_scale = 0.0 if args.noBkgSub else 1.0

    for i, step in enumerate(steps):
        per_proc = meas[step]

        def measure(scale):
            vd = w2d = vb = w2b = None
            vm = w2m = None
            for proc, h in per_proc.items():
                v, w2 = collapse_event(h)
                if is_data(proc):
                    vd = v if vd is None else vd + v
                    w2d = w2 if w2d is None else w2d + w2
                elif args.signal in proc:
                    vm = v if vm is None else vm + v
                    w2m = w2 if w2m is None else w2m + w2
                else:
                    vb = v if vb is None else vb + v
                    w2b = w2 if w2b is None else w2b + w2
            if vb is not None and scale:
                vd, w2d = vd - scale * vb, w2d + scale**2 * w2b
            ed, vard = efficiency(vd, w2d)
            em, varm = efficiency(vm, w2m)
            return sf_and_variance(ed, vard, em, varm), ed, em

        (sf, var), ed, em = measure(bkg_scale)
        ok = (ed > 0) & (em > 0)
        logger.info(
            f"event step '{step}': eps_data {ed[ok].mean():.4f}"
            f"  eps_MC {em[ok].mean():.4f}  SF {sf[ok].mean():.4f}"
            f" +- {np.sqrt(var[ok]).mean():.4f}  ({ok.sum()}/{ok.size} bins)"
        )
        sf_alt, var_alt = (
            measure(bkg_scale * (1.0 + args.altBkgVar))[0]
            if args.altBkgVar is not None
            else (sf, var)
        )
        # yll and ptll carry flow bins; charge, step and nom-alt do not
        view.value[1:-1, 1:-1, :, i, IDX_NOM] = sf
        view.variance[1:-1, 1:-1, :, i, IDX_NOM] = var
        view.value[1:-1, 1:-1, :, i, IDX_ALT] = sf_alt
        view.variance[1:-1, 1:-1, :, i, IDX_ALT] = var_alt
    extend_to_flow(view)

    with h5py.File(args.eventOutput, "w") as f:
        ioutils.pickle_dump_h5py(SF_HIST_NAME, h_sf, f)
        ioutils.pickle_dump_h5py(
            "meta_info",
            dict(input=args.input, steps=steps, eventLevel=True,
                 bkgSubtracted=not args.noBkgSub, altBkgVar=args.altBkgVar,
                 hasAlt=args.altBkgVar is not None),
            f,
        )
    logger.info(f"wrote {args.eventOutput}")


if __name__ == "__main__":
    main()
