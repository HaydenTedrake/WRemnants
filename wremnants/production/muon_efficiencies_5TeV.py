"""Muon efficiency measurement for the 5.02 TeV (2017G) low-PU Z -> mumu.

Follows David's CVH-efficiency measurement (mz_dilepton.py --cvhEfficiencyHists
-> scripts/analysisTools/w_mass_13TeV/make_cvh_efficiency_sf.py): fill a
pass/fail histogram against the muon kinematics separately for data and MC,
then form SF = eps_data / eps_MC outside the histmaker. Nothing is imported
from a pre-measured SF file.

WHY NOT muon_efficiencies_smooth.py: that is the 13 TeV machinery. Its era map
has no "2017G" and it reads a 13 TeV SF file. The one low-PU SF file in the
repo (data/lowPU/efficiencies/2021-10-22_allSFs_nodz_dxybs_lowPU.root) belongs
to era 2017H = the 13 TeV low-PU run (datasetDict_2017H.py: TuneCP5_13TeV), not
to 2017G = 5.02 TeV (datasetDict_2017G.py: TuneCP5_5020GeV).

TRIGGER OBJECTS ARE NOT AVAILABLE IN THIS PRODUCTION. The 5 TeV NanoAOD
(NanoV9MC2017_TrackFitV722_NanoProdv3) has the TrigObj_* branches but they are
empty for muons: 0 objects with TrigObj_id == 13 across 3 MC and 2 data files,
while a 2017H file gives 1852 muon objects for 1852 triggered events. So no
muon can be matched to the object that fired HLT_HIMu17.

That costs nothing for the per-muon steps below, because -- exactly as in
David's measurement -- the quantity being measured is uncorrelated with the
trigger, so BOTH muons of the Z candidate can be used and no tag/probe split is
needed.

It blocks only the PER-MUON trigger efficiency. The trigger step itself IS
implemented, at event level (trigger_step / part 3), because that is the quantity
this analysis actually applies: mz_5TeV.py requires HLT_HIMu17 of the EVENT, not
of a muon, so P(event fires) is what the correction has to be, and that is a
ratio of event counts needing no objects. Measured SF 0.9895 +- 0.0004, flat in
both fit axes. Muon_triggerIdLoose is an offline quality flag mirroring the HLT
muon ID, NOT a record that the muon fired.

NO uT DEPENDENCE anywhere: the 13 TeV smooth SFs can bin trigger/iso in the
hadronic recoil (`smooth3D`); a 2D (eta, pt) map is what this analysis needs.
No function here takes a uT argument.
"""

import hist

from wums import logging

logger = logging.child_logger(__name__)

# The analysis ID, defined once so the probe selection and the pass flag cannot
# drift apart (mz_5TeV.py:478 uses the same pair).
ANALYSIS_ID = "Muon_mediumId && Muon_isGlobal"
LOOSE_ID = "Muon_looseId"


class EfficiencyStep:
    """One measurable efficiency.

    probe_sel  selection defining which muons ENTER the measurement. It must
               NOT contain the requirement being measured, or the efficiency is
               1 by construction.
    pass_expr  per-muon expression, evaluated on the probes, that is the
               measured requirement.
    needs_tag  whether an unbiased probe is required (i.e. the measured
               quantity is correlated with the trigger). False lets both muons
               of the Z enter, which is what David's CVH measurement does.
    """

    def __init__(
        self, name, probe_sel, pass_expr, needs_tag=False, doc="", needs_branches=()
    ):
        self.name = name
        self.probe_sel = probe_sel
        self.pass_expr = pass_expr
        self.needs_tag = needs_tag
        self.doc = doc
        # branches that must exist for this step to be measurable at all
        self.needs_branches = tuple(needs_branches)


def cvh_step(is_data, branch_mc="cvhideal"):
    """David's step, ported: did the CVH refit succeed?

    'cvh' in data, 'cvhideal' in MC -- MC is reconstructed with ideal geometry,
    so the ideal refit is the one actually applied (same convention as
    mz_dilepton.py --cvhEfficiencyBranchMC, whose default is cvhideal).

    Refit success is an offline track-fitting outcome, uncorrelated with the
    trigger, so both muons are used.
    """
    branch = "cvh" if is_data else branch_mc
    return EfficiencyStep(
        name="cvh",
        probe_sel=ANALYSIS_ID,
        pass_expr=f"Muon_{branch}Pt > 0.f",
        needs_tag=False,
        doc=f"CVH refit success via Muon_{branch}Pt > 0",
        needs_branches=(f"Muon_{branch}Pt",),
    )


# The analysis ID factorised the way 13 TeV factorises it. Measuring
# mediumId && isGlobal as ONE step hides that it is two effects with different
# eta structure and different causes (diagnosed on the full 2017G data):
#
#   eta bin        isGlobal SF   mediumId|isGlobal SF
#   [ 0.2, 0.4)      0.9880            0.9971          <- isGlobal drives it
#   [-0.4,-0.2)      0.9893            0.9969          <- isGlobal
#   [ 0.8, 1.0)      0.9949            0.9984          <- isGlobal
#   [ 2.0, 2.2)      0.9990            0.9793          <- mediumId drives it
#   [-2.2,-2.0)      0.9953            0.9814          <- mediumId
#
# PER MUON the split is an identity -- P(global and mediumId) =
# P(global) * P(mediumId | global) is just the chain rule -- so it changes no
# physics. What it buys is an independent nuisance block per effect, which is what
# 13 TeV does, and a direct numerical comparison against their idip SF.
#
# The INCLUSIVE numbers do not close exactly, by ~1e-4 (measured 8.3e-5 in data,
# 6.8e-6 in MC), and that is expected rather than a bug: define_probes requires
# EXACTLY TWO probes per event, and the two steps have different probe
# definitions, so they run on slightly different event samples. An event with two
# loose muons one of which is not global contributes a failing probe to 'reco'
# but is dropped from 'idip', which sees only one probe. 13 TeV has the same
# feature -- its reco, tracking and idip probes also come from different
# selections. At 1e-4 this is two orders of magnitude below the per-cell
# statistical error (~0.3%), so it is a consistency check, not an identity.


def reco_step():
    """Global-muon reconstruction: probe is a loose muon, requirement isGlobal.

    Named after the 13 TeV 'reco' step so the nuisance groups line up
    (muon_eff_stat_reco already exists in styles.py), but the REQUIREMENT IS NOT
    IDENTICAL and should not be quoted as such: 13 TeV's reco is
    isGlobal && highPurity plus standalone-track requirements
    (muon_selections.py:84) with a separate 'tracking' step layered on top
    (:90). Neither highPurity nor the standalone branches enter the 5 TeV
    selection, which asks only for mediumId && isGlobal, so this step is
    isGlobal alone and absorbs what 13 TeV splits between reco and tracking.

    This is the half that carries the CENTRAL eta dips.
    """
    return EfficiencyStep(
        name="reco",
        probe_sel=LOOSE_ID,
        pass_expr="Muon_isGlobal",
        needs_tag=False,
        doc="Muon_isGlobal, probed on loose muons (13 TeV 'reco'+'tracking')",
        needs_branches=("Muon_isGlobal", "Muon_looseId"),
    )


def idip_step():
    """Analysis ID given reconstruction: probe is a loose GLOBAL muon,
    requirement mediumId.

    Directly comparable to the 13 TeV 'idip' SF
    (wremnants-data/data/muonSF/allSmooth_GtoHout_vtxAgnIso.root,
    SF_original_GtoH_idip_{plus,minus}): their endcap dip reads 0.9784-0.9794 and
    this step measures 0.9793-0.9814 -- the same detector effect recovered
    independently at two beam energies, which is the strongest validation the
    measurement has.

    The 'ip' half of the inherited name is NOT measured here: the 5 TeV selection
    applies no impact-parameter cut (mz_5TeV.py asks only for pt, |eta|,
    mediumId, isGlobal), whereas 13 TeV's idip includes abs(Muon_dxybs) < cut.
    The name is kept for group alignment; the content is the ID alone.

    This is the half that carries the ENDCAP eta dip.
    """
    return EfficiencyStep(
        name="idip",
        # probe must already be reconstructed, or this measures reco x id again
        probe_sel=f"{LOOSE_ID} && Muon_isGlobal",
        pass_expr="Muon_mediumId",
        needs_tag=False,
        doc="Muon_mediumId given isGlobal (13 TeV 'idip', no ip cut at 5 TeV)",
        needs_branches=("Muon_mediumId", "Muon_isGlobal", "Muon_looseId"),
    )


def id_step():
    """The MERGED analysis ID: probe is a loose muon, requirement is
    mediumId && isGlobal.

    Superseded by reco_step() x idip_step() for the SF map, but still booked,
    because it is a free consistency check: eps(reco) * eps(idip) should track
    eps(id) to ~1e-4, the residual coming from the differing event samples the
    'exactly two probes' filter produces (see the comment above reco_step). A
    disagreement well beyond that means the probe selections have drifted apart.

    DO NOT put this step in an SF map alongside reco/idip -- the nominal helper
    multiplies every step in the map, so that would apply the ID correction
    twice. make_muon_efficiency_sf_5TeV.py refuses the combination.
    """
    return EfficiencyStep(
        name="id",
        probe_sel=LOOSE_ID,
        pass_expr=ANALYSIS_ID,
        needs_tag=False,
        doc="mediumId && isGlobal, probed on loose muons (MERGED, closure only)",
        needs_branches=("Muon_mediumId", "Muon_isGlobal", "Muon_looseId"),
    )


# Event-level steps. The measurement is one entry per EVENT, binned in the fit
# variables, rather than one per probe muon binned in (eta, pt).
EVENT_AXES = ("yll", "ptll")


class EventEfficiencyStep:
    """One efficiency that belongs to the EVENT, not to a muon.

    denom_sel  an orthogonal or looser trigger defining the denominator. The
               numerator is denom_sel && pass_expr.
    """

    def __init__(self, name, denom_sel, pass_expr, doc="", needs_branches=()):
        self.name = name
        self.denom_sel = denom_sel
        self.pass_expr = pass_expr
        self.doc = doc
        self.needs_branches = tuple(needs_branches)


def trigger_step(denom="HLT_HIL3DoubleMu0", numerator="HLT_HIMu17"):
    """The single-muon trigger efficiency, measured at EVENT level.

    This is the '1HLT' step. mz_5TeV.py requires HLT_HIMu17 of the EVENT
    (df.Filter("HLT_HIMu17")), never of an individual muon, so the quantity the
    analysis needs is P(event fires | Z event), which is a ratio of event counts
    and needs no trigger objects.

    A PER-MUON trigger efficiency is not available in this production: assigning
    "this muon fired" requires the HLT object that caused the decision, matched in
    (eta, phi), and TrigObj holds no muon objects at all (no TrigObj_id == 13 in
    5.16M events; only id 11). Muon_triggerIdLoose is an offline quality flag
    mirroring the HLT muon ID, NOT a record that the muon fired, and the L1_*
    branches are event-level seed decisions. If a per-muon efficiency is ever
    needed without new NanoAOD, it can be FITTED from these event-level counts by
    exploiting the two-muon structure -- P(fires) = 1 - (1-eps1)(1-eps2) -- which
    recovers eps(eta, pt) without any matching.

    THE DENOMINATOR is an orthogonal trigger, not "no requirement": every event in
    the SingleMuon primary dataset already fired something in that dataset, so an
    unrequired denominator is not trigger-unbiased. HLT_HIL3DoubleMu0 is the
    least-correlated path with usable rate (measured SF 0.9895 with it, 0.9937
    with no requirement). It is only fully unbiased if that path sits in a
    different primary dataset -- the one open question on the absolute value,
    which does not touch the shape.
    """
    return EventEfficiencyStep(
        name="trigger",
        denom_sel=denom,
        pass_expr=numerator,
        doc=f"{numerator} given {denom} (event level, the '1HLT' step)",
        needs_branches=(denom, numerator),
    )


def available_steps(df, is_data, steps):
    """Drop steps whose input branches are absent from this dataset.

    Needed because the 5 TeV productions are asymmetric: the MC carries the CVH
    refit branches (Muon_cvhPt, Muon_cvhidealPt, ...) but the DATA NanoAOD has
    NONE of them -- this analysis calibrates with scarekit, not CVH, so the data
    production never stored a refit. An efficiency SF is eps_data/eps_MC, so a
    step measurable only in MC is not measurable at all; it is dropped here with
    a warning rather than failing at JIT time with an undeclared identifier.
    """
    cols = {str(c) for c in df.GetColumnNames()}
    keep = []
    for st in steps:
        missing = sorted(b for b in st.needs_branches if b not in cols)
        if missing:
            logger.warning(
                f"Skipping efficiency step '{st.name}' on "
                f"{'data' if is_data else 'MC'}: missing {missing}"
            )
            continue
        keep.append(st)
    return keep


def define_probes(
    df,
    step,
    pt_col="Muon_pt",
    pt_min=18.0,
    eta_max=2.4,
    prefix="effProbes",
    n_probes=2,
):
    """Select the probe muons for `step` and flag which of them pass.

    Uses UNCORRECTED kinematics by default (pt_col="Muon_pt"), following David:
    the corrected quantities are undefined when a refit fails, so measuring
    refit efficiency against them would silently drop the failing bin.

    Requires exactly `n_probes` probes (2 = both muons of the Z candidate), so
    the Z selection stays unambiguous; each probe is filled separately.
    """
    kin = f"{pt_col} > {pt_min} && abs(Muon_eta) < {eta_max}"
    df = df.Define(f"{prefix}", f"{kin} && {step.probe_sel}")
    df = df.Filter(
        f"ROOT::VecOps::Sum({prefix}) == {n_probes}",
        f"exactly {n_probes} probes ({step.name})",
    )
    df = df.Define(f"{prefix}_pt", f"{pt_col}[{prefix}]")
    df = df.Define(f"{prefix}_eta", f"Muon_eta[{prefix}]")
    # nanoAOD phi is [-pi,pi]; tracker modules are naturally [0,2pi)
    # ROOT::VecOps::Map with a lambda does not compile in this ROOT build
    # (nor does RVec<int>(RVec<bool>)); Where() is the portable form.
    df = df.Define(
        f"{prefix}_phi",
        f"ROOT::VecOps::Where(Muon_phi[{prefix}] < 0.f,"
        f" Muon_phi[{prefix}] + 2.f*float(M_PI), Muon_phi[{prefix}])",
    )
    df = df.Define(f"{prefix}_charge", f"Muon_charge[{prefix}]")
    df = df.Define(
        f"{prefix}_pass",
        f"ROOT::VecOps::Where(({step.pass_expr})[{prefix}], 1, 0)",
    )
    return df


def build_axes(pt_min=18.0, pt_max=65.0, n_pt=10, n_eta=48, with_phi=False):
    """Axes shared by every step, so all steps carry one variation scheme.

    2D in (eta, pt) -- no uT axis. pt starts at the analysis cut of 18 GeV; the
    2017H low-PU SF file starts at 25 and would leave 18-25 uncovered.

    MEASURE FINE, COARSEN LATER: make_muon_efficiency_sf_5TeV.py --rebinPt /
    --rebinEta can only merge bins, so anything the measurement does not resolve
    is lost for good. n_eta defaults to 48 (0.1 wide, matching the 13 TeV SF
    files) because the data/MC efficiency difference is NOT smooth in eta: at 0.2
    the SF bottoms out at 0.9769 with a 2.25% spread, at 0.1 it reaches 0.9538
    with a 4.71% spread. The 0.2 binning was averaging real structure.

    with_phi adds the azimuth, needed only to localise module-level effects such
    as David's glued-module hotspots.
    """
    axes = [
        hist.axis.Regular(n_pt, pt_min, pt_max, name="pt", overflow=True),
        hist.axis.Regular(n_eta, -2.4, 2.4, name="eta"),
    ]
    if with_phi:
        import math

        axes.append(
            hist.axis.Regular(
                72, 0.0, 2.0 * math.pi, name="phi", underflow=False, overflow=False
            )
        )
    axes += [
        hist.axis.Regular(2, -2.0, 2.0, underflow=False, overflow=False, name="charge"),
        hist.axis.Integer(0, 2, name="pass", underflow=False, overflow=False),
    ]
    return axes


def book_efficiency_hist(
    df, results, step, weight="nominal_weight", prefix="effProbes", with_phi=False, **kw
):
    """Append the pass/fail histogram for one step. Call on data and on MC."""
    name = f"muonEff_{step.name}"
    cols = [f"{prefix}_{v}" for v in ("pt", "eta")]
    if with_phi:
        cols.append(f"{prefix}_phi")
    cols += [f"{prefix}_charge", f"{prefix}_pass", weight]
    logger.info(f"Booking {name} ({step.doc}) with columns {cols}")
    results.append(df.HistoBoost(name, build_axes(with_phi=with_phi, **kw), cols))
    return results


# --- per-muon trigger extraction (full lowPU / '1HLT' parity) ---------------
#
# lowPU (mz_lowPU.py:343, lowpu_efficiencies.hpp:lepSF_HLT_q) keeps PER-MUON
# trigger efficiencies for data and MC separately and combines them as
#     corr = (1 - prod_i (1 - eps_data_i)) / (1 - prod_i (1 - eps_MC_i))
#          = P(at least one muon fired)_data / P(...)_MC
# which is non-linear in eps, so the two maps cannot be collapsed into one ratio.
# Its map comes from an external TnP file where each muon was individually tagged.
#
# No muon can be tagged here (no trigger objects), but the SAME formula read
# backwards recovers the per-muon efficiencies from what IS observable: whether
# the EVENT fired, as a function of BOTH muons' bins. The sufficient statistic is
# therefore the (bin_1, bin_2, pass) triplet, which is what this histogram holds;
# the fit lives in make_muon_efficiency_sf_5TeV.py.
#
# Binning defaults to lowPU's 12 eta x 10 pt x 2 charges for parity. Note the
# information content is much lower than a tagged TnP: only the event decision is
# seen, so eps_a is constrained through pairs containing bin a, diluted by
# (1 - eps_b). Expect per-bin errors of a few percent on eps and check them
# before trusting the finest binning.

TRIG_N_ETA, TRIG_N_PT, TRIG_N_CHARGE = 12, 10, 2
TRIG_PT_RANGE = (18.0, 65.0)


def trigger_bin_index_expr(imu, n_eta=TRIG_N_ETA, n_pt=TRIG_N_PT,
                           pt_lo=TRIG_PT_RANGE[0], pt_hi=TRIG_PT_RANGE[1]):
    """RDF expression for a muon's flattened (eta, pt, charge) bin index.

    Flattened as ((charge_bin * n_eta) + eta_bin) * n_pt + pt_bin, clamped into
    range so overflow muons share the edge bin (the same convention the SF
    lookups use).
    """
    eta = f"Muon_eta[{imu}]"
    pt = f"Muon_pt[{imu}]"
    q = f"Muon_charge[{imu}]"
    ieta = f"std::clamp(int(({eta} + 2.4f) * {n_eta} / 4.8f), 0, {n_eta - 1})"
    ipt = (f"std::clamp(int(({pt} - {pt_lo}f) * {n_pt} / {pt_hi - pt_lo}f),"
           f" 0, {n_pt - 1})")
    iq = f"({q} > 0 ? 1 : 0)"
    return f"(({iq} * {n_eta} + {ieta}) * {n_pt} + {ipt})"


def build_trigger_pair_axes(n_bins=None):
    """Axes for the per-muon trigger extraction: (bin1, bin2, pass)."""
    if n_bins is None:
        n_bins = TRIG_N_ETA * TRIG_N_PT * TRIG_N_CHARGE
    return [
        hist.axis.Integer(0, n_bins, name="bin1", underflow=False, overflow=False),
        hist.axis.Integer(0, n_bins, name="bin2", underflow=False, overflow=False),
        hist.axis.Integer(0, 2, name="pass", underflow=False, overflow=False),
    ]


def book_trigger_pair_hist(
    df, results, step, weight="nominal_weight", mu0="i0", mu1="i1", **kw
):
    """Append the (bin1, bin2, pass) histogram the per-muon trigger fit needs.

    Filters to the orthogonal denominator first, exactly as
    book_event_efficiency_hist does, so the histogram holds the denominator
    sample with the event decision as the pass flag.
    """
    name = f"muonEffTrigPair_{step.name}"
    df = df.Filter(step.denom_sel, f"{step.name} pair denominator")
    df = df.Define("trigPair_bin1", trigger_bin_index_expr(mu0))
    df = df.Define("trigPair_bin2", trigger_bin_index_expr(mu1))
    df = df.Define(f"trigPair_{step.name}_pass", f"({step.pass_expr}) ? 1 : 0")
    cols = ["trigPair_bin1", "trigPair_bin2", f"trigPair_{step.name}_pass", weight]
    logger.info(f"Booking {name} for the per-muon trigger fit, columns {cols}")
    results.append(df.HistoBoost(name, build_trigger_pair_axes(**kw), cols))
    return results


def build_event_axes(yll_edges=None, ptll_edges=None):
    """Axes for an event-level measurement: the FIT variables.

    Defaults are the fit binnings themselves (binning.yll_10quantiles_binning and
    a coarse ptll), so the correction is measured in exactly the variables whose
    templates it can distort. Coarser than the fit's ptll axis on purpose: the
    measured trigger SF is flat in ptll (chi2/ndf 0.60 against a constant), so
    spending bins there only adds statistical noise.
    """
    if yll_edges is None:
        from wremnants.utilities import binning

        yll_edges = binning.yll_10quantiles_binning
    if ptll_edges is None:
        ptll_edges = [0.0, 8.0, 16.0, 30.0, 100.0]
    return [
        hist.axis.Variable(list(yll_edges), name="yll"),
        hist.axis.Variable(list(ptll_edges), name="ptll"),
        hist.axis.Integer(0, 2, name="pass", underflow=False, overflow=False),
    ]


def book_event_efficiency_hist(
    df,
    results,
    step,
    weight="nominal_weight",
    yll_col="yll",
    ptll_col="ptll",
    **kw,
):
    """Append the event-level pass/fail histogram for one step.

    Filters to the denominator first, so the histogram holds exactly the
    denominator sample with a pass flag -- the same shape the per-muon booking
    produces, one entry per event instead of one per probe.
    """
    name = f"muonEffEvent_{step.name}"
    df = df.Filter(step.denom_sel, f"{step.name} denominator")
    df = df.Define(f"{step.name}_pass", f"({step.pass_expr}) ? 1 : 0")
    cols = [yll_col, ptll_col, f"{step.name}_pass", weight]
    logger.info(f"Booking {name} ({step.doc}) with columns {cols}")
    results.append(df.HistoBoost(name, build_event_axes(**kw), cols))
    return results


# ---------------------------------------------------------------------------
# Part 2: the scale factors and their template variations.
#
# Part 1 above MEASURES the efficiencies (booked by mz_5TeV.py
# --muonEfficiencyHists). scripts/analysisTools/make_muon_efficiency_sf_5TeV.py
# turns those pass/fail histograms into an SF map; this part loads that map and
# builds the three helpers the histmaker applies:
#
#   helper       nominal per-event SF, multiplied into nominal_weight
#   helper_stat  {step: helper}, one variation per (eta, pt, charge) bin of that
#                step -- the effStatTnP nuisances
#   helper_syst  one fully correlated variation per step -- the effSystTnP
#                nuisances, built only when the map carries an alternative
#                measurement
#
# Same three-object contract as make_muon_efficiency_helpers_binned/_smooth, so
# systematics.add_muon_efficiency_unc_hists's booking pattern carries over; the
# call signatures differ because there is no tag/probe split and no uT (see
# muon_efficiencies_5TeV.hpp).
# ---------------------------------------------------------------------------

SF_HIST_NAME = "SF"
# axis order is contractual: muon_efficiencies_5TeV.hpp indexes it positionally
SF_AXES = ("eta", "pt", "charge", "step", "nom-alt")
# event-level maps reuse the same five slots with the first two holding the fit
# variables and charge carrying a single dummy bin (see the hpp)
SF_EVENT_AXES = ("yll", "ptll", "charge", "step", "nom-alt")
IDX_NOM, IDX_ALT = 0, 1


def _declare():
    """Declare the C++ helpers on first use.

    Deferred rather than done at import time so that merely importing this
    module for the measurement (part 1) does not pull in ROOT/Eigen.
    """
    import narf.clingutils

    if not getattr(_declare, "done", False):
        narf.clingutils.Declare('#include "muon_efficiencies_5TeV.hpp"')
        _declare.done = True


def load_sf_map(filename):
    """Read the SF map written by make_muon_efficiency_sf_5TeV.py."""
    import h5py
    from wums import ioutils

    with h5py.File(filename, "r") as f:
        if SF_HIST_NAME not in f:
            raise KeyError(
                f"{filename} has no '{SF_HIST_NAME}' entry"
                f" (found {sorted(f.keys())}); was it written by"
                " make_muon_efficiency_sf_5TeV.py?"
            )
        h = ioutils.pickle_load_h5py(f[SF_HIST_NAME])
        h = h.get() if hasattr(h, "get") else h
        # resolve inside the context manager: an H5PickleProxy read after the
        # file closes raises "underlying file has been closed"
        names = tuple(a.name for a in h.axes)
    if names not in (SF_AXES, SF_EVENT_AXES):
        raise ValueError(
            f"SF map axes are {names}, expected {SF_AXES} (per muon)"
            f" or {SF_EVENT_AXES} (per event)"
        )
    return h


def is_event_map(h):
    """True for an event-level SF map (first two axes are the fit variables)."""
    return tuple(a.name for a in h.axes) == SF_EVENT_AXES


def _tensor_axis(axis, name=None):
    """A copy of `axis` with no flow bins, to index a fixed-size Eigen tensor."""
    import boost_histogram as bh

    name = name or axis.name
    if isinstance(axis, bh.axis.Regular):
        return hist.axis.Regular(
            axis.size,
            axis.edges[0],
            axis.edges[-1],
            name=name,
            underflow=False,
            overflow=False,
        )
    return hist.axis.Variable(
        axis.edges, name=name, underflow=False, overflow=False
    )


def make_muon_efficiency_helpers_5TeV(filename, max_pt=None):
    """Build (helper, helper_syst, {step: helper_stat}) from an SF map file.

    max_pt drops the pt bins above the analysis cut from the STAT tensor only,
    so no nuisance is created for a bin no selected muon can occupy. The nominal
    SF lookup still sees the full map, as in the 13 TeV factories.

    helper_syst is None when the map carries no alternative measurement, i.e.
    when its nom-alt slots are identical. A syst helper built on an identical
    alternative would hand the fit one nuisance per step whose up and down
    templates both equal the nominal -- a degenerate direction in the
    likelihood, not a zero uncertainty -- so it is refused rather than returned.
    """
    import narf
    import numpy as np
    import ROOT

    _declare()

    h = load_sf_map(filename)
    steps = [str(s) for s in h.axes["step"]]
    n_eta = h.axes["eta"].size
    n_charge = h.axes["charge"].size
    n_pt_all = h.axes["pt"].size
    n_pt = (
        int(np.count_nonzero(h.axes["pt"].edges[:-1] < max_pt))
        if max_pt is not None
        else n_pt_all
    )
    logger.info(
        f"5 TeV muon SF map: steps {steps}, {n_eta} eta x {n_pt_all} pt"
        f" x {n_charge} charge; {n_pt} pt bins carry stat nuisances"
    )

    # A cell with no measurement has SF 1 and variance 0, so its stat variation
    # is identically 1: the fit would get a nuisance whose up and down templates
    # both equal the nominal, which is a flat direction in the likelihood rather
    # than a small uncertainty. Report them so they can be pruned downstream.
    n_empty = int((h.variances()[..., IDX_NOM] <= 0).sum())
    if n_empty:
        logger.warning(
            f"{n_empty} of {h.variances()[..., IDX_NOM].size} map cells carry no"
            " measurement (zero variance); their effStatTnP variations are"
            " identically 1 and must be pruned when the datacard is built."
        )

    has_alt = not np.allclose(
        h.values()[..., IDX_NOM], h.values()[..., IDX_ALT], equal_nan=True
    )

    # one pyroot copy per helper: hist_to_pyroot_boost hands over ownership and
    # ROOT.std.move empties the python-side object
    def pyroot_copy():
        return narf.hist_to_pyroot_boost(h)

    sf_nom = pyroot_copy()
    helper = ROOT.wrem.muon_efficiency_5TeV_helper[type(sf_nom)](
        ROOT.std.move(sf_nom)
    )

    helper_syst = None
    if has_alt:
        sf_syst = pyroot_copy()
        helper_syst = ROOT.wrem.muon_efficiency_5TeV_helper_syst[
            len(steps), type(sf_syst)
        ](ROOT.std.move(sf_syst))
        # Integer rather than StrCategory: the python StrCategory bindings always
        # carry an overflow bin, and a tensor axis has to match the fixed Eigen
        # size exactly. Same choice as muon_efficiencies_binned.py.
        helper_syst.tensor_axes = [
            hist.axis.Integer(
                0, len(steps), underflow=False, overflow=False, name="-".join(steps)
            )
        ]
    else:
        logger.warning(
            "SF map has no alternative measurement (nom-alt slots identical):"
            " building the statistical variations only, no effSystTnP."
        )

    axis_eta_t = _tensor_axis(h.axes["eta"])
    axis_charge_t = _tensor_axis(h.axes["charge"])

    helpers_stat = {}
    for istep, step in enumerate(steps):
        sf_stat = pyroot_copy()
        hs = ROOT.wrem.muon_efficiency_5TeV_helper_stat[
            n_eta, n_pt, n_charge, type(sf_stat)
        ](ROOT.std.move(sf_stat), istep)
        pt_edges = h.axes["pt"].edges[: n_pt + 1]
        hs.tensor_axes = [
            axis_eta_t,
            hist.axis.Variable(
                pt_edges, name="pt", underflow=False, overflow=False
            ),
            axis_charge_t,
        ]
        helpers_stat[step] = hs

    return helper, helper_syst, helpers_stat


SF_MUON_COLS = (
    "sfMu_pt0",
    "sfMu_eta0",
    "sfMu_charge0",
    "sfMu_pt1",
    "sfMu_eta1",
    "sfMu_charge1",
)


def define_muon_sf_columns(
    df,
    muon_exprs=(
        ("Muon_pt_corr[i0]", "Muon_eta[i0]", "Muon_charge[i0]"),
        ("Muon_pt_corr[i1]", "Muon_eta[i1]", "Muon_charge[i1]"),
    ),
):
    """Materialise the six scalar columns the helpers take.

    RDataFrame's Define(name, callable, columns) resolves `columns` as column
    NAMES, not as expressions, so the per-muon quantities have to exist as
    columns of their own before any helper can be attached to them. Doing it once
    here also pins the types the helpers are declared with: float pt and eta, int
    charge (Muon_charge is not an RVec<int> in every nanoAOD version).

    muon_exprs defaults to the two good muons of mz_5TeV.py, whose indices are
    the columns i0 and i1.
    """
    for imu, (pt, eta, charge) in enumerate(muon_exprs):
        df = df.Define(f"sfMu_pt{imu}", f"float({pt})")
        df = df.Define(f"sfMu_eta{imu}", f"float({eta})")
        df = df.Define(f"sfMu_charge{imu}", f"int({charge})")
    return df


def define_nominal_sf_weight(df, helper, out_col="weight_muonSF",
                             skip_columns=False):
    """Define the per-event nominal SF column. Call on MC only, then multiply it
    into nominal_weight.

    skip_columns=True when define_muon_sf_columns has already run (the per-muon
    trigger step calls it first); RDF refuses to Define an existing column.
    """
    if not skip_columns:
        df = define_muon_sf_columns(df)
    df = df.Define(out_col, helper, list(SF_MUON_COLS))
    return df, out_col


def add_muon_efficiency_unc_hists_5TeV(
    results,
    df,
    helper_stat,
    helper_syst,
    axes,
    cols,
    base_name="ptll",
    **kwargs,
):
    """Book the efficiency template variations onto the fit axes.

    Deliberately named and shaped after
    systematics.add_muon_efficiency_unc_hists so the two are comparable side by
    side, but it takes no analysis-type argument: that function assembles
    tnpPt0/tnpEta0/tnpUT0/tnpCharge0 lists per analysis type and skips the
    non-triggering muon for the trigger step, and none of that applies here --
    this selection has two symmetric probes and no uT.

    Assumes define_muon_sf_columns has already run, which
    define_nominal_sf_weight does; it is idempotent-unsafe (RDF refuses to
    Define an existing column), so it is not repeated here.

    Nuisance names come out as <base_name>_effStatTnP_<step> and
    <base_name>_effSystTnP, matching 13 TeV.
    """
    from wremnants.production import systematics
    from wremnants.utilities import common

    mucols = list(SF_MUON_COLS)

    for step, helper in helper_stat.items():
        tensor = f"effStatTnP_{step}_tensor"
        df = df.Define(tensor, helper, [*mucols, "nominal_weight"])
        systematics.add_syst_hist(
            results,
            df,
            common.hist_name(base_name, syst=f"effStatTnP_{step}"),
            axes,
            cols,
            tensor,
            helper.tensor_axes,
            **kwargs,
        )

    if helper_syst is not None:
        df = df.Define("effSystTnP_weight", helper_syst, [*mucols, "nominal_weight"])
        systematics.add_syst_hist(
            results,
            df,
            common.hist_name(base_name, syst="effSystTnP"),
            axes,
            cols,
            "effSystTnP_weight",
            helper_syst.tensor_axes,
            **kwargs,
        )

    return df


# ---------------------------------------------------------------------------
# Part 3: the event-level ('1HLT') trigger scale factor.
#
# Same three helpers and the same SF file format as part 2, but one factor per
# event instead of one per muon, and the map is binned in the fit variables
# (yll, ptll) rather than in muon kinematics -- see the hpp for why a per-muon
# trigger efficiency is not available in this production.
# ---------------------------------------------------------------------------

SF_EVENT_COLS = ("sfEvt_ptll", "sfEvt_yll")


def define_event_sf_columns(df, ptll_col="ptll", yll_col="yll"):
    """Materialise FLOAT copies of the fit variables for the event-level helper.

    ptll and yll are doubles in the histmaker (ROOT::Math::PtEtaPhiMVector
    returns double) and the helper signature is (float, float); cppyy will not
    narrow a double for a templated callable -- it fails with
    "TypeError: could not convert argument 1".
    """
    if not df.HasColumn(SF_EVENT_COLS[0]):
        df = df.Define(SF_EVENT_COLS[0], f"float({ptll_col})")
        df = df.Define(SF_EVENT_COLS[1], f"float({yll_col})")
    return df


def make_muon_efficiency_event_helpers_5TeV(filename):
    """Build (helper, helper_syst, {step: helper_stat}) from an EVENT-level map.

    Mirrors make_muon_efficiency_helpers_5TeV, including the refusal to build a
    syst helper when the map carries no alternative measurement.
    """
    import narf
    import numpy as np
    import ROOT

    _declare()

    h = load_sf_map(filename)
    if not is_event_map(h):
        raise ValueError(
            f"{filename} is a PER-MUON SF map (axes {tuple(a.name for a in h.axes)});"
            " pass it to make_muon_efficiency_helpers_5TeV instead"
        )
    steps = [str(x) for x in h.axes["step"]]
    n0, n1 = h.axes["yll"].size, h.axes["ptll"].size
    logger.info(
        f"5 TeV event-level SF map: steps {steps}, {n0} yll x {n1} ptll bins"
    )
    n_empty = int((h.variances()[..., IDX_NOM] <= 0).sum())
    if n_empty:
        logger.warning(
            f"{n_empty} of {h.variances()[..., IDX_NOM].size} event-map cells carry"
            " no measurement; their effStatTnP variations are identically 1 and"
            " must be pruned when the datacard is built."
        )
    has_alt = not np.allclose(
        h.values()[..., IDX_NOM], h.values()[..., IDX_ALT], equal_nan=True
    )

    def pyroot_copy():
        return narf.hist_to_pyroot_boost(h)

    sf = pyroot_copy()
    helper = ROOT.wrem.muon_efficiency_5TeV_event_helper[type(sf)](ROOT.std.move(sf))

    helper_syst = None
    if has_alt:
        sfs = pyroot_copy()
        helper_syst = ROOT.wrem.muon_efficiency_5TeV_event_helper_syst[
            len(steps), type(sfs)
        ](ROOT.std.move(sfs))
        helper_syst.tensor_axes = [
            hist.axis.Integer(
                0, len(steps), underflow=False, overflow=False, name="-".join(steps)
            )
        ]
    else:
        logger.warning(
            "Event-level SF map has no alternative measurement: statistical"
            " variations only, no effSystTnP."
        )

    # RENAMED: the tensor axes are appended to the FIT axes, which are also
    # called ptll and yll, and hist refuses duplicated axis names
    # ("Hist instance cannot contain axes with duplicated names").
    ax0 = _tensor_axis(h.axes["yll"], name="effBinYll")
    ax1 = _tensor_axis(h.axes["ptll"], name="effBinPtll")
    helpers_stat = {}
    for istep, step in enumerate(steps):
        sfst = pyroot_copy()
        hs = ROOT.wrem.muon_efficiency_5TeV_event_helper_stat[n0, n1, type(sfst)](
            ROOT.std.move(sfst), istep
        )
        hs.tensor_axes = [ax0, ax1]
        helpers_stat[step] = hs
    return helper, helper_syst, helpers_stat


def define_nominal_event_sf_weight(df, helper, out_col="weight_muonEventSF"):
    """Define the per-event trigger SF column. MC only; multiply into
    nominal_weight.

    Argument order follows the per-muon helper's (pt, eta) convention: the first
    argument indexes axis<1> (ptll) and the second axis<0> (yll).
    """
    df = define_event_sf_columns(df)
    df = df.Define(out_col, helper, list(SF_EVENT_COLS))
    return df, out_col


def add_muon_efficiency_event_unc_hists_5TeV(
    results, df, helper_stat, helper_syst, axes, cols, base_name="ptll", **kwargs
):
    """Book the event-level efficiency variations onto the fit axes.

    Nuisance names match the per-muon ones -- <base_name>_effStatTnP_<step> and
    <base_name>_effSystTnP_event -- so setupRabbit picks them up with the same
    logic; the syst hist gets its own name because the two maps' step axes are
    independent.
    """
    from wremnants.production import systematics
    from wremnants.utilities import common

    df = define_event_sf_columns(df)
    mucols = list(SF_EVENT_COLS)
    for step, helper in helper_stat.items():
        tensor = f"effStatTnP_{step}_tensor"
        df = df.Define(tensor, helper, [*mucols, "nominal_weight"])
        systematics.add_syst_hist(
            results,
            df,
            common.hist_name(base_name, syst=f"effStatTnP_{step}"),
            axes,
            cols,
            tensor,
            helper.tensor_axes,
            **kwargs,
        )
    if helper_syst is not None:
        df = df.Define(
            "effSystTnP_event_weight", helper_syst, [*mucols, "nominal_weight"]
        )
        systematics.add_syst_hist(
            results,
            df,
            common.hist_name(base_name, syst="effSystTnP_event"),
            axes,
            cols,
            "effSystTnP_event_weight",
            helper_syst.tensor_axes,
            **kwargs,
        )
    return df


# ---------------------------------------------------------------------------
# Part 4: the per-muon trigger scale factor (full lowPU / '1HLT' parity).
#
# Carries eps_data and eps_MC separately and combines them with lowPU's own
# formula; the statistical nuisances are EIGENVECTORS of the fit covariance, not
# one per bin (see trigger_efficiency_fit_5TeV.py for why).
# ---------------------------------------------------------------------------

TRIG_EPS_HIST, TRIG_EIG_HIST = "eps", "eig"
TRIG_EPS_AXES = ("eta", "pt", "charge", "kind")
TRIG_EIG_AXES = ("eta", "pt", "charge", "kind", "mode")
KIND_DATA, KIND_MC = 0, 1


def load_trigger_maps(filename):
    """Read the (eps, eig) pair written by make_muon_trigger_sf_5TeV.py."""
    import h5py
    from wums import ioutils

    with h5py.File(filename, "r") as f:
        for key in (TRIG_EPS_HIST, TRIG_EIG_HIST):
            if key not in f:
                raise KeyError(
                    f"{filename} has no '{key}' entry (found {sorted(f.keys())})"
                )
        eps = ioutils.pickle_load_h5py(f[TRIG_EPS_HIST])
        eig = ioutils.pickle_load_h5py(f[TRIG_EIG_HIST])
        eps = eps.get() if hasattr(eps, "get") else eps
        eig = eig.get() if hasattr(eig, "get") else eig
        na = tuple(a.name for a in eps.axes)
        nb = tuple(a.name for a in eig.axes)
    if na != TRIG_EPS_AXES:
        raise ValueError(f"eps axes are {na}, expected {TRIG_EPS_AXES}")
    if nb != TRIG_EIG_AXES:
        raise ValueError(f"eig axes are {nb}, expected {TRIG_EIG_AXES}")
    return eps, eig


def make_muon_trigger_helpers_5TeV(filename, max_modes=None):
    """Build (helper, {kind: helper_stat}) for the per-muon trigger step.

    Returns one stat helper per kind, so the data and MC statistical
    uncertainties enter as independent nuisance blocks exactly as they do in
    lowPU (muSF_HLT_DATA_stat vs muSF_HLT_MC_stat).

    max_modes truncates the eigen expansion. The dropped variance is reported, so
    truncation is never silent; leave it None to keep every mode.
    """
    import narf
    import numpy as np
    import ROOT

    _declare()
    eps, eig = load_trigger_maps(filename)
    n_modes_all = eig.axes["mode"].size
    n_modes = n_modes_all if max_modes is None else min(max_modes, n_modes_all)
    if n_modes < n_modes_all:
        v = eig.values()
        kept = (v[..., :n_modes] ** 2).sum()
        tot = (v**2).sum()
        logger.warning(
            f"Keeping {n_modes}/{n_modes_all} eigen modes:"
            f" {100 * kept / tot:.2f}% of the variance retained"
        )
        eig = eig[{"mode": slice(0, n_modes)}]
    logger.info(
        f"5 TeV per-muon trigger maps: {eps.axes['eta'].size} eta x"
        f" {eps.axes['pt'].size} pt x {eps.axes['charge'].size} charge,"
        f" {n_modes} eigen modes"
    )

    def copies():
        return narf.hist_to_pyroot_boost(eps), narf.hist_to_pyroot_boost(eig)

    a, b = copies()
    helper = ROOT.wrem.muon_efficiency_5TeV_trigger_helper[type(a), type(b)](
        ROOT.std.move(a), ROOT.std.move(b)
    )

    axis_mode = hist.axis.Integer(
        0, n_modes, underflow=False, overflow=False, name="eigenMode"
    )
    helpers_stat = {}
    for name, kind in (("data", KIND_DATA), ("mc", KIND_MC)):
        a, b = copies()
        hs = ROOT.wrem.muon_efficiency_5TeV_trigger_helper_stat[
            n_modes, type(a), type(b)
        ](ROOT.std.move(a), ROOT.std.move(b), kind)
        hs.tensor_axes = [axis_mode]
        helpers_stat[name] = hs
    return helper, helpers_stat


def define_nominal_trigger_sf_weight(df, helper, out_col="weight_muonTrigSF"):
    """Per-event per-muon trigger SF. MC only; multiply into nominal_weight."""
    df = define_muon_sf_columns(df)
    df = df.Define(out_col, helper, list(SF_MUON_COLS))
    return df, out_col


def add_muon_trigger_unc_hists_5TeV(
    results, df, helpers_stat, axes, cols, base_name="ptll", **kwargs
):
    """Book the per-muon trigger eigen variations.

    Names follow lowPU's: <base_name>_effStatTnP_trigger_{DATA,MC}, so the data
    and MC statistical uncertainties stay separate nuisance blocks.
    """
    from wremnants.production import systematics
    from wremnants.utilities import common

    for name, helper in helpers_stat.items():
        tag = f"trigger_{name.upper()}"
        tensor = f"effStatTnP_{tag}_tensor"
        df = df.Define(tensor, helper, [*SF_MUON_COLS, "nominal_weight"])
        systematics.add_syst_hist(
            results,
            df,
            common.hist_name(base_name, syst=f"effStatTnP_{tag}"),
            axes,
            cols,
            tensor,
            helper.tensor_axes,
            **kwargs,
        )
    return df
