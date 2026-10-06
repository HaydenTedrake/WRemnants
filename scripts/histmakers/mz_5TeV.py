import math
import os

from wremnants.production import muon_efficiencies_5TeV
from wremnants.utilities import binning, common, parsing, samples, theory_utils
from wums import logging

analysis_label = common.analysis_label(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)
parser.add_argument(
    "--muonEfficiencyHists",
    action="store_true",
    help="Measure muon efficiencies (tag-free, David's CVH method: both muons of "
    "the Z fill a pass/fail histogram; SF = eps_data/eps_MC is formed afterwards). "
    "RELAXES the analysis muon ID to loose so the ID step is measurable at all -- "
    "the physics histograms from such a run are NOT the analysis ones. Uses "
    "uncorrected kinematics, since corrected pt is undefined when a refit fails.",
)
parser.add_argument(
    "--nanoVersion",
    choices=["v9", "v15"],
    default="v9",
    help="5.02 TeV NanoAOD production for data AND MC (datasetDict_2017G "
    "NANO_PROD_TAGS / NANO_DATA_TAGS). v15 (Oct 2026) adds the muon trigger "
    "objects and the standalone-muon branches the in-situ efficiencies need.",
)
# ---- In-situ muon efficiencies (WRemnants PR #709) --------------------------
# ID, trigger and isolation efficiencies floated in the fit as unconstrained
# Chebyshev coefficients, measured from four categories of the dimuon sample
# (nominal = 2HLT_passIso, failIso, failHLT = 1HLT_passID, failID). The mode is
# implied by its inputs, exactly as in mz_dilepton.py:
#   pass 0   --makeInsituEffMC                -> effMCprobe_* spectra, from which
#            scripts/corrections/make_insitu_effMC.py builds the effMC file
#   pass 1   --insituEffMCFile <effMC>        -> four fit channels + variations
#   pass n   ... --insituSFFile <theta_central> (from make_insitu_effSF.py)
# REQUIRES muon trigger objects (TrigObj id 13) and standalone-muon branches in
# the NanoAOD; neither exists in the current 5.02 TeV production. See
# hayden_notes.txt, "IN-SITU MUON EFFICIENCIES".
parser.add_argument(
    "--insituEffMCFile",
    type=str,
    default=None,
    help="MC efficiency pkl from make_insitu_effMC.py. Enables the in-situ "
    "efficiency mode: four fit channels and the muonInsituEff variations.",
)
parser.add_argument(
    "--insituSFFile",
    type=str,
    default=None,
    help="Accumulated theta_central pkl from make_insitu_effSF.py (iteration "
    ">= 1): reweights the MC by the current central in-situ scale factor.",
)
parser.add_argument(
    "--makeInsituEffMC",
    action="store_true",
    help="In-situ pass 0: emit the effMCprobe_* probe spectra that "
    "make_insitu_effMC.py turns into the --insituEffMCFile input.",
)
# The common parser only gives these to high-PU analyses; the low-PU label of
# this script does not get them, and the in-situ mode needs an isolation.
if "--isolationDefinition" not in parser._option_string_actions:
    parser.add_argument(
        "--isolationDefinition",
        choices=["iso04vtxAgn", "iso04", "iso04chg", "iso04chgvtxAgn"],
        # iso04 is the only definition whose branch (Muon_pfRelIso04_all)
        # exists in the 5.02 TeV DATA NanoAOD; the 13 TeV default, iso04vtxAgn,
        # needs Muon_vtxAgnPfRelIso04_all, which only the MC carries.
        default="iso04",
        help="Isolation used by the in-situ efficiency categories",
    )
if "--isolationThreshold" not in parser._option_string_actions:
    parser.add_argument(
        "--isolationThreshold",
        type=float,
        default=0.15,
        help="Isolation threshold for the in-situ efficiency categories",
    )
parser.add_argument(
    "--muonEventScaleFactors",
    default=None,
    help="EVENT-level SF map (the '1HLT' single-muon trigger step) from "
    "make_muon_efficiency_sf_5TeV.py --eventOutput. Separate from "
    "--muonScaleFactors because it is one factor per event, not one per muon.",
)
parser.add_argument(
    "--muonTriggerScaleFactors",
    default=None,
    help="PER-MUON trigger maps (eps_data, eps_MC + eigen shifts) from "
    "make_muon_trigger_sf_5TeV.py -- full lowPU '1HLT' parity. Mutually "
    "exclusive with --muonEventScaleFactors, which is the event-level "
    "alternative to the same correction.",
)
parser.add_argument(
    "--muonEffProbeTrigger",
    default="HLT_HIMu17 || HLT_HIL3DoubleMu0",
    help="event filter used by the MEASUREMENT run instead of HLT_HIMu17 (which "
    "would make the '1HLT' step 1 by construction). The default is the OR of the "
    "trigger numerator and denominator -- everything either measurement needs, "
    "~3%% of the input. Re-running with a different, weaker path (e.g. "
    "'HLT_HIL1DoubleMu0') gives the ALTERNATIVE measurement that "
    "make_muon_efficiency_sf_5TeV.py --altFrom turns into effSystTnP: the "
    "probe-selection systematic, which is the counting analogue of 13 TeV's "
    "originalDataAltSig.",
)
parser.add_argument(
    "--muonEffEtaBins",
    type=int,
    default=48,
    help="eta bins for the efficiency MEASUREMENT (default 48 = 0.1 wide, the "
    "13 TeV SF granularity). Measure fine: the SF builder can only coarsen, so "
    "structure this run does not resolve is lost for good.",
)
parser.add_argument(
    "--muonEffPtBins",
    type=int,
    default=10,
    help="pt bins for the efficiency measurement (default 10).",
)
parser.add_argument(
    "--muonScaleFactors",
    default=None,
    help="SF map from scripts/analysisTools/make_muon_efficiency_sf_5TeV.py. Applies "
    "the nominal muon efficiency SF to MC and books the effStatTnP (one nuisance per "
    "eta-pt-charge bin per step) and effSystTnP (one fully correlated nuisance per "
    "step, only if the map carries an alternative measurement) template variations.",
)
parser.add_argument(
    "--muonCorr",
    default="none",
    choices=["none", "rochester", "scarekit"],
    help="Muon momentum correction to apply",
)
parser.add_argument(
    "--helicityXsecsFile",
    type=str,
    default=None,
    help="5 TeV w_z_helicity_xsecs file (from scripts/rabbit/make_helicity_xsecs_file.py, "
    "based on the nominal_gen_helicity_xsecs_scale hist that this histmaker always "
    "fills for Z MC); enables the helicity-decomposed QCD scale systematic hist",
)
parser.add_argument(
    "--oneMCfileEveryN",
    type=int,
    default=None,
    help="Use 1 MC file every N, where N is given by this option. Mainly for tests",
)
parser.add_argument(
    "--sameSign",
    action="store_true",
    help="Select same-sign dimuon pairs instead of opposite-sign: signal-free "
    "control region for the one-off fakes validation (fakes are ~charge-"
    "symmetric, so N_fakes(OS) ~ N(SS)). Combine with --postfix.",
)
# This is the 5 TeV low-PU analysis: default to the 5 TeV era (2017G)
parser.set_defaults(era="2017G")
args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

import hist
import ROOT

import narf
from wremnants.production import (
    generator_level_definitions,
    muon_efficiencies_insitu,
    muon_selections,
    nanoaod_compat,
    systematics,
    theory_corrections,
)
from wremnants.production.datasets.dataset_tools import getDatasets
from wremnants.production.histmaker_tools import write_analysis_output

if args.muonCorr == "rochester":
    narf.clingutils.Load("libPhysics")
    narf.clingutils.Load("libROOTVecOps")
    narf.clingutils.Load("libROOTDataFrame")
    narf.clingutils.Declare('#include "lowpu_rochester.hpp"')
elif args.muonCorr == "scarekit":
    narf.clingutils.Load("libROOTDataFrame")
    narf.clingutils.Declare('#include "lowpu_muonscarekit.hpp"')
    scarekit_mc_helper = ROOT.wrem.MuonScarekitMCHelper(args.randomSeedForToys)
    # resolution up/down: smearing factor k shifted by +-1 bootstrap std;
    # identical per-event RNG seeding keeps the same smearing random number
    scarekit_mc_helper_resolup = ROOT.wrem.MuonScarekitMCHelper(
        args.randomSeedForToys, 1.0
    )
    scarekit_mc_helper_resoldn = ROOT.wrem.MuonScarekitMCHelper(
        args.randomSeedForToys, -1.0
    )
    # resolution systematic (scarekit --syst 4 window-variation spread),
    # same construction as the stat variation but reading the syst4 file
    scarekit_mc_helper_resolsystup = ROOT.wrem.MuonScarekitMCHelper(
        args.randomSeedForToys, 1.0, True
    )
    scarekit_mc_helper_resolsystdn = ROOT.wrem.MuonScarekitMCHelper(
        args.randomSeedForToys, -1.0, True
    )

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    base_path=args.dataPath,
    era=args.era,
    oneMCfileEveryN=args.oneMCfileEveryN,
    nanoVersion=args.nanoVersion,
)

import pickle

import lz4.frame

theory_corrs = args.theoryCorr
theory_corr_base = f"{common.data_dir}/TheoryCorrections/5020GeV"


def load_corr_hist_5020(filename, proc, histname):
    """Handle both standard-format 5020 GeV pickles (proc key 'Z') and the
    legacy ones from David (ZMUMU5020GEV keys and legacy hist names)."""
    with lz4.frame.open(filename) as f:
        corr = pickle.load(f)
    if proc in corr and histname in corr[proc]:
        return corr[proc][histname]
    key = histname.replace("scetlib_dyturbo_LatticeNP", "scetlib_dyturboLatticeNP")
    key = key.replace("_minnlo_ratio", "__minnlo_ratio")
    return corr["ZMUMU5020GEV"][key]


theory_corrections.load_corr_hist = load_corr_hist_5020

# The scetlib_np lambda_central metadata hook (histmaker_tools) resolves the
# corr pkl in the top-level TheoryCorrections/ dir, where a 13 TeV file with
# the SAME tag exists but a different central runcard (delta_lambda2 = 0.0 vs
# our 0.125) - point it at the 5020GeV files instead.
from wremnants.postprocessing.scetlib_np import (
    lambda_central as scetlib_np_lambda_central,
)


def _correction_pkl_path_5020(tag, proc, data_dir=None):
    return f"{theory_corr_base}/{tag}_Corr{proc}.pkl.lz4"


scetlib_np_lambda_central._correction_pkl_path = _correction_pkl_path_5020

corr_helpers = theory_corrections.load_corr_helpers(
    [d.name for d in datasets if d.name in samples.zprocs],
    theory_corrs,
    base_dir=theory_corr_base,
)

# EW/FSR corrections: the 13 TeV ratio files are borrowed as-is (agreed
# Jul 15) - they are functions of the gen dilepton kinematics only, which
# are sqrt(s)-independent to good approximation. Variations only, the
# central prediction is not modified (same convention as 13 TeV).
ew_theory_corrs = [
    "powhegFOEW",
    "pythiaew_ISR",
    "horaceqedew_FSR",
    "horacelophotosmecoffew_FSR",
]
ew_corr_helpers = theory_corrections.load_corr_helpers(
    [d.name for d in datasets if d.name in samples.zprocs],
    ew_theory_corrs,
)

# define histogram axes, see: https://hist.readthedocs.io/en/latest/index.html
axis_nLepton = hist.axis.Integer(0, 5, name="nLepton", underflow=False)
axis_mll = hist.axis.Regular(60, 76, 106, name="mll")
dilepton_ptV_binning = [
    0,
    1,
    1.5,
    2,
    2.5,
    3,
    3.5,
    4,
    4.5,
    5,
    5.5,
    6,
    6.5,
    7,
    7.5,
    8,
    8.5,
    9,
    9.5,
    10,
    10.5,
    11,
    11.5,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    24,
    26,
    28,
    30,
    33,
    37,
    44,
    100,
]
axis_ptll = hist.axis.Variable(
    dilepton_ptV_binning, name="ptll", underflow=False, overflow=True
)
yll_10quantiles_binning = [-2.5, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.5]
axis_yll = hist.axis.Variable(
    yll_10quantiles_binning, name="yll", underflow=True, overflow=True
)

# Gen axes for the SCETlib-NP param-model response matrix (PR #701).
# ptVGen ends at 44 (the last reco edge below the wide 44-100 bin) with
# overflow=True: load_R folds the true qT > 44 flow into a trailing (44, 100]
# gen bin so the model can feed it from the btgrid.
axis_ptVGen = hist.axis.Variable(
    dilepton_ptV_binning[:-1], name="ptVGen", underflow=False, overflow=True
)
axis_absYVGen = hist.axis.Regular(
    10, 0, 2.5, name="absYVGen", underflow=False, overflow=True
)
axis_acceptance = hist.axis.Boolean(name="acceptance")
# only the UL(-1) entry: for a 2D ptll-yll fit load_R sums the helicity
# partition right back out (R) or takes the UL bin (N_gen), so a single
# angular-integrated bin filled with the plain weight is exact (Integer, not
# IntCategory: narf's category-axis conversion rejects int categories)
axis_helicitySig_ul = hist.axis.Integer(
    -1, 0, name="helicitySig", underflow=False, overflow=False
)
axis_mu_pt = hist.axis.Regular(60, 25, 150, name="mu_pt")
axis_mu_eta = hist.axis.Regular(48, -2.4, 2.4, name="mu_eta")
axis_mu_phi = hist.axis.Regular(32, -math.pi, math.pi, circular=True, name="mu_phi")
axis_mu_oneOverPt = hist.axis.Regular(50, 0.005, 0.04, name="mu_oneOverPt")
axis_mu_charge = hist.axis.Variable([-1.5, -0.5, 0.5, 1.5], name="mu_charge")
axis_mu_nl = hist.axis.Variable(
    [6.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 17.5], name="mu_nl"
)
axis_mu_masspt = hist.axis.Regular(100, 0, 1e4, name="mu_masspt")
axis_cosThetaStarll = hist.axis.Regular(
    200, -1.0, 1.0, name="cosThetaStarll", underflow=False, overflow=False
)
axis_phiStarll = hist.axis.Regular(
    20, -math.pi, math.pi, circular=True, name="phiStarll"
)
axis_phill = hist.axis.Regular(50, -math.pi, math.pi, circular=True, name="phill")

# entries: 0/1 muon stat up/down, 2/3 muon syst up/down, 4/5 ECAL up/down

# Gen-level axes for the helicity cross sections with muR/muF variations
# (input to the helicity-decomposed QCD scale uncertainty). Filled before
# any reco selection so the angular coefficients are acceptance-unbiased.
# flow bins required: the correction helper looks up events by gen kinematics
# and out-of-range values (gen mass tails) must land in flow bins, which the
# helper initializes to a safe weight of 1
axis_massVgen = hist.axis.Regular(
    1, 60.0, 120.0, name="massVgen", underflow=True, overflow=True
)
axis_absYVgen = hist.axis.Variable(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
    name="absYVgen",
    underflow=False,
    overflow=True,
)
axis_ptVgen = hist.axis.Variable(
    dilepton_ptV_binning, name="ptVgen", underflow=False, overflow=True
)
axis_chargeVgen = hist.axis.Regular(
    1, -1.0, 1.0, name="chargeVgen", underflow=False, overflow=False
)
# coarse gen-ptV axis on the reco qcdScaleByHelicity hist, for nuisances
# decorrelated in ~10% ptV quantiles as in the 13 TeV setup
axis_ptVgen_decorr = hist.axis.Variable(
    dilepton_ptV_binning[::4], name="ptVgen", underflow=False, overflow=True
)

qcd_helicity_helper = None
if args.helicityXsecsFile:
    qcd_helicity_helper = theory_corrections.make_qcd_uncertainty_helper_by_helicity(
        is_z=True,
        filename=args.helicityXsecsFile,
        rebin_ptVgen=False,
        rebin_absYVgen=False,
        rebin_massVgen=False,
    )
    logger.info(f"Loaded qcdScaleByHelicity helper from {args.helicityXsecsFile}")


muon_efficiency_helper = None
muon_efficiency_helper_syst = None
muon_efficiency_helpers_stat = {}
if args.muonScaleFactors:
    if args.muonEfficiencyHists:
        raise ValueError(
            "--muonEfficiencyHists relaxes the muon ID to loose to MEASURE the"
            " efficiencies; applying SFs in the same run would weight a selection"
            " that is not the analysis one. Run the two separately."
        )
    (
        muon_efficiency_helper,
        muon_efficiency_helper_syst,
        muon_efficiency_helpers_stat,
    ) = muon_efficiencies_5TeV.make_muon_efficiency_helpers_5TeV(
        # no max_pt: unlike 13 TeV (--pt upper edge) this selection puts no upper
        # cut on the muon pt, so every bin of the map is in use. Muons above the
        # map go to its overflow, which carries the last bin's SF and shares its
        # nuisance (see extend_to_flow / the clamp in the stat helper).
        args.muonScaleFactors
    )
    logger.info(f"Muon SF map: {args.muonScaleFactors}")

muon_event_helper = None
muon_event_helper_syst = None
muon_event_helpers_stat = {}
if args.muonEventScaleFactors:
    if args.muonEfficiencyHists:
        raise ValueError(
            "--muonEfficiencyHists MEASURES the efficiencies on a relaxed"
            " selection; applying SFs in the same run would weight a selection"
            " that is not the analysis one. Run the two separately."
        )
    (
        muon_event_helper,
        muon_event_helper_syst,
        muon_event_helpers_stat,
    ) = muon_efficiencies_5TeV.make_muon_efficiency_event_helpers_5TeV(
        args.muonEventScaleFactors
    )
    logger.info(f"Muon event-level SF map: {args.muonEventScaleFactors}")

muon_trig_helper = None
muon_trig_helpers_stat = {}
if args.muonTriggerScaleFactors:
    if args.muonEventScaleFactors:
        raise ValueError(
            "--muonTriggerScaleFactors (per muon) and --muonEventScaleFactors"
            " (per event) are two forms of the SAME trigger correction; applying"
            " both would correct the trigger twice. Pick one."
        )
    if args.muonEfficiencyHists:
        raise ValueError(
            "--muonEfficiencyHists MEASURES the efficiencies; applying SFs in the"
            " same run would weight a selection that is not the analysis one."
        )
    muon_trig_helper, muon_trig_helpers_stat = (
        muon_efficiencies_5TeV.make_muon_trigger_helpers_5TeV(
            args.muonTriggerScaleFactors
        )
    )
    logger.info(f"Muon per-muon trigger maps: {args.muonTriggerScaleFactors}")


# ---- In-situ efficiency setup (PR #709) --------------------------------------
insituMode = args.insituEffMCFile is not None or args.makeInsituEffMC
isoBranch = None
muon_insitu_efficiency_helper = None
muon_insitu_central_helper = None
if insituMode:
    for _opt in (
        "muonEfficiencyHists",
        "muonScaleFactors",
        "muonTriggerScaleFactors",
        "muonEventScaleFactors",
    ):
        if getattr(args, _opt):
            raise ValueError(
                f"--{_opt} belongs to the external-SF efficiency treatment; the "
                "in-situ mode floats ID/HLT/Iso in the fit instead. Use one."
            )
    isoBranch = muon_selections.getIsoBranch(args.isolationDefinition)
    (
        muon_insitu_efficiency_helper,
        muon_insitu_central_helper,
        _insitu_labels,
    ) = muon_efficiencies_insitu.setup_muon_insitu_helpers(
        args.insituEffMCFile, args.insituSFFile, args.makeInsituEffMC
    )
    # probe axes of the three control channels, exactly as in mz_dilepton.py
    # (the Chebyshev window is fixed centrally in muon_efficiencies_insitu)
    axis_insitu_eta = hist.axis.Regular(
        int(args.eta[0]),
        args.eta[1],
        args.eta[2],
        name="eta",
        underflow=False,
        overflow=False,
    )
    axis_pt_failID = hist.axis.Variable(
        muon_efficiencies_insitu.insitu_pt_edges(args.pt[2]),
        name="pt",
        underflow=False,
        overflow=False,
    )
    axis_recoUT = hist.axis.Variable(
        muon_efficiencies_insitu.insitu_ut_edges,
        name="recoUT",
        underflow=False,
        overflow=False,
    )
    axis_insitu_charge = binning.axis_charge
    axes_insitu_effMC, cols_insitu_effMC = (
        muon_efficiencies_insitu.make_insitu_effMC_axes(
            args.eta[0], args.eta[1], args.eta[2]
        )
    )
    logger.info(
        f"In-situ efficiency mode: iso {isoBranch} < {args.isolationThreshold},"
        f" probe pt > {muon_efficiencies_insitu.insitu_pt_range[0]} GeV"
    )

# NanoAOD branches the in-situ selection reads. Checked per dataset so the job
# stops with the list instead of failing inside the JIT.
INSITU_REQUIRED_BRANCHES = [
    "Muon_standalonePt",
    "Muon_standaloneEta",
    "Muon_standalonePhi",
    "Muon_standaloneNumberOfValidHits",
    "TrigObj_id",
    "TrigObj_filterBits",
    "TrigObj_eta",
    "TrigObj_phi",
    "Muon_highPurity",
    "Muon_dxybs",
]


def define_insitu_muon_selection(df, dataset):
    """The PR #709 tag-and-probe muon selection, on the 5.02 TeV kinematics.

    Reuses the shared muon_selections helpers unchanged. They read the
    Muon_corrected* columns, which at 13 TeV come from the CVH calibration; here
    they are the scarekit-calibrated kinematics. Returns a dataframe with
    firstMuons / secondMuons (negative / positive, as in the dilepton
    convention), their per-leg passID0 / passTrigger0 / passIso0 flags, and the
    i0 / i1 indices the rest of this histmaker builds its dimuon kinematics from.
    """
    cols = {str(c) for c in df.GetColumnNames()}
    missing = [b for b in INSITU_REQUIRED_BRANCHES if b not in cols]
    if isoBranch not in cols:
        missing.append(isoBranch)
    if missing:
        raise RuntimeError(
            f"{dataset.name}: the in-situ efficiency mode needs NanoAOD branches"
            f" this production does not have: {missing}. The standalone-muon and"
            " vertex-agnostic isolation branches exist in the 5.02 TeV MC but not"
            " in the data; they have to be added in the NanoAOD reproduction."
        )

    df = df.Alias("Muon_correctedPt", "Muon_pt_corr")
    df = df.Alias("Muon_correctedEta", "Muon_eta")
    df = df.Alias("Muon_correctedPhi", "Muon_phi")
    df = df.Alias("Muon_correctedCharge", "Muon_charge")

    # defines Muon_isGoodGlobal (from the standalone-muon branches) and vetoMuons
    df = muon_selections.select_veto_muons(df, nMuons=-1)
    # T&P probe denominator: good global muon, no ID requirement. The pt cut is
    # the lower edge of the shared Chebyshev window, so changing that window
    # moves the probe cut with it.
    df = df.Define(
        "insituProbeMuons",
        "Muon_isGoodGlobal && Muon_correctedCharge != -99"
        f" && Muon_correctedPt > {muon_efficiencies_insitu.insitu_pt_range[0]}"
        " && abs(Muon_correctedEta) < 2.4",
    )
    df = df.Filter("Sum(insituProbeMuons) == 2", "in-situ: two probe muons")
    # at least one good (= passes ID) muon: the tag
    df = muon_selections.select_good_muons(
        df,
        args.pt[1],
        args.pt[2],
        nMuons=1,
        condition=">=",
        use_isolation=False,
        isoBranch=isoBranch,
        isoThreshold=args.isolationThreshold,
    )
    df = muon_selections.define_two_muons(df, dilepton=True, muons="insituProbeMuons")
    # MUST come before anything evaluates firstMuons_pt0: that column indexes
    # [firstMuons][0] and runs off the end for a same-charge probe pair
    df = df.Filter(
        "Sum(firstMuons) == 1 && Sum(secondMuons) == 1", "in-situ: opposite charge"
    )
    df = muon_selections.select_standalone_muons(df, dataset, False, "firstMuons")
    df = muon_selections.select_standalone_muons(df, dataset, False, "secondMuons")
    # per-leg trigger match, OR of the two legs. The HLT_HIMu17 muon objects are
    # selected with the low-PU helper, as in mz_lowPU.py (TrigObj_id == 13 &&
    # filterBits bit 0; needs the nano v15 trigger objects -- v9 has no muon ones)
    df = muon_selections.apply_triggermatching_muon(
        df,
        dataset,
        "firstMuons",
        "secondMuons",
        goodTrigObjs="wrem::goodMuonTriggerCandidateLowPU(TrigObj_id, TrigObj_pt,"
        " TrigObj_l1pt, TrigObj_l2pt, TrigObj_filterBits)",
    )
    df = df.Define("firstMuons_passID0", "Sum(goodMuons && firstMuons) == 1")
    df = df.Define("secondMuons_passID0", "Sum(goodMuons && secondMuons) == 1")
    df = df.Define(
        "firstMuons_passIso0", f"{isoBranch}[firstMuons][0] < {args.isolationThreshold}"
    )
    df = df.Define(
        "secondMuons_passIso0",
        f"{isoBranch}[secondMuons][0] < {args.isolationThreshold}",
    )
    df = df.Define("i0", "int(ROOT::VecOps::Nonzero(firstMuons)[0])")
    df = df.Define("i1", "int(ROOT::VecOps::Nonzero(secondMuons)[0])")
    return df


def define_insitu_mc_uT(df):
    """Gen-level uT per leg (MC only): the muon projected on the gen boson.

    The in-situ helper parameterises trigger and isolation in this variable, as
    SMP-23-002 does; only MC is reweighted, so data never needs it.
    """
    for leg in ("firstMuons", "secondMuons"):
        df = df.Define(
            f"{leg}_tnpUT0",
            f"wrem::zqtproj0_boson({leg}_pt0, {leg}_phi0, ptVgen, phiVgen)",
        )
    return df


def define_insitu_central_weight(df):
    """W(theta_central) for iteration >= 1, folded into exp_weight. Mirrors
    mz_dilepton.py: events with no tag (dropped from every channel) get W = 1."""
    ci = systematics.insitu_category_index
    df = df.Define("insituW_firstTag", "firstMuons_passID0 && firstMuons_passTrigger0")
    df = df.Define(
        "insituW_secondTag", "secondMuons_passID0 && secondMuons_passTrigger0"
    )
    df = df.Define(
        "insituW_probeIsFirst",
        "(insituW_firstTag && insituW_secondTag) ? isEvenEvent"
        " : static_cast<bool>(!insituW_firstTag)",
    )
    for v in ("pt0", "eta0", "charge0", "tnpUT0"):
        df = df.Define(
            f"insituW_probe_{v}",
            f"insituW_probeIsFirst ? firstMuons_{v} : secondMuons_{v}",
        )
        df = df.Define(
            f"insituW_tag_{v}",
            f"insituW_probeIsFirst ? secondMuons_{v} : firstMuons_{v}",
        )
    df = df.Define(
        "insituW_cat",
        "static_cast<int>((insituW_firstTag && insituW_secondTag) ? "
        "((insituW_probeIsFirst ? firstMuons_passIso0 : secondMuons_passIso0) ? "
        f"{ci['nominal']} : {ci['failIso']}) : "
        "((insituW_probeIsFirst ? firstMuons_passID0 : secondMuons_passID0) ? "
        f"{ci['failHLT']} : {ci['failID']}))",
    )
    df = df.Define(
        "insituCentralW_raw",
        muon_insitu_central_helper,
        [
            *(f"insituW_probe_{v}" for v in ("pt0", "eta0", "charge0", "tnpUT0")),
            *(f"insituW_tag_{v}" for v in ("pt0", "eta0", "charge0", "tnpUT0")),
            "insituW_cat",
        ],
    )
    return df.Define(
        "insituCentralW",
        "(insituW_firstTag + insituW_secondTag) >= 1 ? insituCentralW_raw : 1.0",
    )


def _assign_probe_tag(d, probe_is_first, is_mc):
    """probeMuons_* / tagMuons_* columns given a C++ bool expression."""
    for v in ("pt0", "eta0", "charge0", "phi0"):
        d = d.Define(
            f"probeMuons_{v}", f"({probe_is_first}) ? firstMuons_{v} : secondMuons_{v}"
        )
    for v in ("pt0", "eta0", "charge0"):
        d = d.Define(
            f"tagMuons_{v}", f"({probe_is_first}) ? secondMuons_{v} : firstMuons_{v}"
        )
    # reco uT: probe projected on the reconstructed Z -- no MET, data and MC
    d = d.Define(
        "probeMuons_recoUT0",
        "static_cast<double>(wrem::zqtproj0_boson(probeMuons_pt0, probeMuons_phi0,"
        " ptll, phill))",
    )
    if is_mc:
        d = d.Define(
            "probeMuons_tnpUT0",
            f"({probe_is_first}) ? firstMuons_tnpUT0 : secondMuons_tnpUT0",
        )
        d = d.Define(
            "tagMuons_tnpUT0",
            f"({probe_is_first}) ? secondMuons_tnpUT0 : firstMuons_tnpUT0",
        )
    return d


def define_insitu_channels(df, dataset, results):
    """The four PR #709 fit channels, plus the pass-0 effMC spectra.

    1HLT = exactly one muon passes ID and HLT (the tag); the other is the probe,
           split into passID (-> failHLT channel) and failID.
    2HLT = both are tags; one is taken at random as the probe, split into
           passIso (-> the NOMINAL signal channel) and failIso.
    """
    is_mc = not dataset.is_data
    df = df.Define("firstMuons_tag0", "firstMuons_passID0 && firstMuons_passTrigger0")
    df = df.Define(
        "secondMuons_tag0", "secondMuons_passID0 && secondMuons_passTrigger0"
    )

    df_1HLT = df.Filter("firstMuons_tag0 != secondMuons_tag0", "1HLT")
    df_1HLT = _assign_probe_tag(df_1HLT, "!firstMuons_tag0", is_mc)
    df_1HLT = df_1HLT.Define(
        "probeMuons_passID0",
        "firstMuons_tag0 ? secondMuons_passID0 : firstMuons_passID0",
    )
    df_1HLT_passID = df_1HLT.Filter("probeMuons_passID0 == 1")
    df_1HLT_failID = df_1HLT.Filter("probeMuons_passID0 == 0")

    df_2HLT = df.Filter("firstMuons_tag0 && secondMuons_tag0", "2HLT")
    # the two legs are indistinguishable valid tags: pick the probe at random
    df_2HLT = _assign_probe_tag(df_2HLT, "isEvenEvent", is_mc)
    df_2HLT = df_2HLT.Define(
        "probeMuons_passIso0",
        "isEvenEvent ? firstMuons_passIso0 : secondMuons_passIso0",
    )
    df_2HLT_passIso = df_2HLT.Filter("probeMuons_passIso0 == 1")
    df_2HLT_failIso = df_2HLT.Filter("probeMuons_passIso0 == 0")

    if args.makeInsituEffMC and is_mc:
        book_insitu_effMC(df_2HLT, df_1HLT_passID, df_1HLT_failID, results)

    eta, pt, q, ut = axis_insitu_eta, axis_pt_failID, axis_insitu_charge, axis_recoUT
    return {
        # the signal channel keeps today's names, so setupRabbit_5TeV.py reads it
        "nominal": {
            "df": df_2HLT_passIso,
            "axes": [axis_ptll, axis_yll],
            "cols": ["ptll", "yll"],
            "base": "ptll",
            "nominal_name": "ptll_vs_yll",
        },
        "failIso": {
            "df": df_2HLT_failIso,
            "axes": [eta, pt, ut],
            "cols": ["probeMuons_eta0", "probeMuons_pt0", "probeMuons_recoUT0"],
            "base": "failIso",
            "nominal_name": "failIso",
        },
        "failHLT": {
            "df": df_1HLT_passID,
            "axes": [eta, pt, ut, q],
            "cols": [
                "probeMuons_eta0",
                "probeMuons_pt0",
                "probeMuons_recoUT0",
                "probeMuons_charge0",
            ],
            "base": "failHLT",
            "nominal_name": "failHLT",
        },
        "failID": {
            "df": df_1HLT_failID,
            "axes": [eta, pt, q],
            "cols": ["probeMuons_eta0", "probeMuons_pt0", "probeMuons_charge0"],
            "base": "failID",
            "nominal_name": "failID",
        },
    }


def book_insitu_effMC(df_2HLT, df_1HLT_passID, df_1HLT_failID, results):
    """Pass-0 effMCprobe_* spectra, as in mz_dilepton.py: both legs of each 2HLT
    event (routed by their own isolation), the single probe of each 1HLT event."""
    clamp = muon_efficiencies_insitu.insitu_ut_clamp_expr
    d = df_2HLT
    d = d.Define(
        "effProbe2HLT_isoMask",
        "ROOT::VecOps::RVec<bool>{firstMuons_passIso0, secondMuons_passIso0}",
    )
    d = d.Define(
        "effProbe2HLT_failIsoMask",
        "ROOT::VecOps::RVec<bool>{!firstMuons_passIso0, !secondMuons_passIso0}",
    )
    d = d.Define(
        "effProbe2HLT_eta",
        "ROOT::VecOps::RVec<float>{firstMuons_eta0, secondMuons_eta0}",
    )
    d = d.Define(
        "effProbe2HLT_pt", "ROOT::VecOps::RVec<float>{firstMuons_pt0, secondMuons_pt0}"
    )
    d = d.Define(
        "effProbe2HLT_charge",
        "ROOT::VecOps::RVec<int>{firstMuons_charge0, secondMuons_charge0}",
    )
    d = d.Define(
        "effProbe2HLT_uT",
        "ROOT::VecOps::RVec<float>{"
        + clamp("firstMuons_tnpUT0")
        + ", "
        + clamp("secondMuons_tnpUT0")
        + "}",
    )
    for cat, mask in (
        ("nominal", "effProbe2HLT_isoMask"),
        ("failIso", "effProbe2HLT_failIsoMask"),
    ):
        for v in ("eta", "pt", "charge", "uT"):
            d = d.Define(f"effProbe_{cat}_{v}", f"effProbe2HLT_{v}[{mask}]")
        results.append(
            d.HistoBoost(
                f"effMCprobe_{cat}",
                axes_insitu_effMC,
                [f"effProbe_{cat}_{v}" for v in ("eta", "pt", "charge", "uT")]
                + ["nominal_weight"],
            )
        )
    for cat, d1 in (("failHLT", df_1HLT_passID), ("failID", df_1HLT_failID)):
        if cat == "failID":
            # an ID-passing probe above the good-muon pt window would otherwise
            # be misclassified as failID (same guard as mz_dilepton.py)
            d1 = d1.Filter(f"probeMuons_pt0 < {args.pt[2]}")
        d1 = d1.Define("probeMuons_tnpUT0Clamped", clamp("probeMuons_tnpUT0"))
        results.append(
            d1.HistoBoost(
                f"effMCprobe_{cat}",
                axes_insitu_effMC,
                [*cols_insitu_effMC[:-1], "probeMuons_tnpUT0Clamped", "nominal_weight"],
            )
        )


def build_graph(df, dataset):
    # NanoAOD v15 stores several integer branches in narrower types than v9
    df = nanoaod_compat.harmonize_nano_types(df)
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    df = df.Define("isEvenEvent", f"event % 2 == 0")

    # Gen-level helicity cross sections with muR/muF scale variations, on a
    # branch of the graph before any reco selection (acceptance-unbiased
    # angular coefficients); aggregated into the w_z_helicity_xsecs input by
    # scripts/rabbit/make_helicity_xsecs_file.py
    is_z_mc = not dataset.is_data and (
        "Zmumu" in dataset.name or "Ztautau" in dataset.name
    )
    if is_z_mc:
        df_gen = generator_level_definitions.define_prefsr_vars(df)
        df_gen = df_gen.DefinePerSample("theory_weight_truncate", "10.0")
        # only the raw tensor; systematics.define_scale_tensor also defines a
        # *_wnom variant needing nominal_weight, which the gen branch lacks
        df_gen = df_gen.Define(
            "scaleWeights_tensor",
            "wrem::makeScaleTensor(LHEScaleWeight, theory_weight_truncate);",
        )
        # apply the central SCETlib+DYTurbo correction to the gen weight, as
        # in the 13 TeV w_z_gen_dists production
        gen_weight_col = "weight"
        main_corr = theory_corrs[0] if theory_corrs else None
        if main_corr is not None and main_corr in corr_helpers.get(dataset.name, {}):
            df_gen = theory_corrections.define_central_pdf_weight(
                df_gen, dataset.name, "ct18z"
            )
            df_gen = df_gen.Define("nominal_weight_uncorr", "weight*central_pdf_weight")
            df_gen = theory_corrections.define_theory_corr_weight_column(
                df_gen, main_corr
            )
            df_gen = df_gen.Define(
                f"gen_{main_corr}Weight_tensor",
                corr_helpers[dataset.name][main_corr],
                [
                    "massVgen",
                    "absYVgen",
                    "ptVgen",
                    "chargeVgen",
                    f"{main_corr}_corr_weight",
                ],
            )
            df_gen = df_gen.Define(
                "gen_nominal_weight", f"gen_{main_corr}Weight_tensor[0]"
            )
            gen_weight_col = "gen_nominal_weight"
        df_gen = df_gen.Define(
            "helicity_xsecs_scale_tensor",
            f"wrem::makeHelicityMomentScaleTensor(csSineCosThetaPhigen, scaleWeights_tensor, {gen_weight_col})",
        )
        hist_helicity_xsecs_scale = df_gen.HistoBoost(
            "nominal_gen_helicity_xsecs_scale",
            [axis_massVgen, axis_absYVgen, axis_ptVgen, axis_chargeVgen],
            [
                "massVgen",
                "absYVgen",
                "ptVgen",
                "chargeVgen",
                "helicity_xsecs_scale_tensor",
            ],
            tensor_axes=[binning.axis_helicity, *systematics.scale_tensor_axes],
            storage=hist.storage.Double(),
        )
        results.append(hist_helicity_xsecs_scale)

        # LHE-level (pre-shower) moments: the difference wrt the pre-FSR
        # moments gives the pythia_shower_kt variation of the coefficients
        df_gen = generator_level_definitions.define_lhe_vars(df_gen)
        df_gen = df_gen.Define(
            "helicity_xsecs_scale_lhe_tensor",
            f"wrem::makeHelicityMomentScaleTensor(csSineCosThetaPhilhe, scaleWeights_tensor, {gen_weight_col})",
        )
        hist_helicity_xsecs_scale_lhe = df_gen.HistoBoost(
            "nominal_gen_helicity_xsecs_scale_lhe",
            [axis_massVgen, axis_absYVgen, axis_ptVgen, axis_chargeVgen],
            [
                "massVlhe",
                "absYVlhe",
                "ptVlhe",
                "chargeVlhe",
                "helicity_xsecs_scale_lhe_tensor",
            ],
            tensor_axes=[binning.axis_helicity, *systematics.scale_tensor_axes],
            storage=hist.storage.Double(),
        )
        results.append(hist_helicity_xsecs_scale_lhe)

        # SCETlib-NP param-model gen-total N_gen (PR #701): normalizes the
        # response, P(b|g) = R/N_gen. N_gen must be INCLUSIVE in muon
        # acceptance: R is loaded acceptance=True (gen-fiducial), so with an
        # inclusive N_gen the ratio P = R/N_gen = acceptance x efficiency x
        # migration, and the model's INCLUSIVE sigma_gen(lambda_c) folds to the
        # fiducial reco spectrum (closes vs the card). Applying the muon
        # pt/|eta| fiducial cuts here divides acceptance OUT of P, leaving the
        # inclusive sigma_gen mismatched to a fiducial baseline -> the
        # check_agreement guard trips (forward-|y| shape blow-up). Only the mass
        # window stays (matching the model Q_lo=76 / Q_hi=106). Pre-FSR,
        # SCETlib-corrected gen weight, booked before any reco selection.
        if "Zmumu" in dataset.name:
            df_gen_ngen = df_gen.Filter("massVgen > 76 && massVgen < 106")
            df_gen_ngen = df_gen_ngen.Define("helicitySigUL", "int(-1)")
            results.append(
                df_gen_ngen.HistoBoost(
                    "prefsr",
                    [axis_ptVGen, axis_absYVGen, axis_helicitySig_ul],
                    ["ptVgen", "absYVgen", "helicitySigUL", gen_weight_col],
                )
            )

    # apply muon momentum corrections before selection
    if args.muonCorr == "rochester":
        if dataset.is_data:
            df = df.Define(
                "Muon_pt_corr",
                "wrem::applyRochesterData(Muon_pt, Muon_eta, Muon_phi, ROOT::VecOps::RVec<float>(Muon_charge.begin(), Muon_charge.end()))",
            )
        else:
            df = df.Define(
                "Muon_pt_corr",
                "wrem::applyRochesterMC(Muon_pt, Muon_eta, Muon_phi, ROOT::VecOps::RVec<float>(Muon_charge.begin(), Muon_charge.end()), Muon_genPartIdx, GenPart_pt, Muon_nTrackerLayers)",
            )
    elif args.muonCorr == "scarekit":
        if dataset.is_data:
            df = df.Define(
                "Muon_pt_corr",
                "wrem::applyMuonScarekitData(Muon_pt, Muon_eta, Muon_phi, Muon_charge)",
            )
        else:
            scarekit_mc_cols = [
                "run",
                "luminosityBlock",
                "event",
                "Muon_pt",
                "Muon_eta",
                "Muon_phi",
                "Muon_charge",
                "Muon_nTrackerLayers",
            ]
            df = df.Define("Muon_pt_corr", scarekit_mc_helper, scarekit_mc_cols)
            # statistical (bootstrap) variations of the corrections, MC only:
            # scale from the kappa/lambda stds (+ correlation), resolution
            # from the k std with the same per-event smearing random number
            df = df.Define(
                "Muon_pt_corr_scaleUp",
                "wrem::varyMuonScarekitScaleMC(Muon_pt_corr, Muon_eta, Muon_phi, Muon_charge, 1.0)",
            )
            df = df.Define(
                "Muon_pt_corr_scaleDown",
                "wrem::varyMuonScarekitScaleMC(Muon_pt_corr, Muon_eta, Muon_phi, Muon_charge, -1.0)",
            )
            df = df.Define(
                "Muon_pt_corr_resolUp", scarekit_mc_helper_resolup, scarekit_mc_cols
            )
            df = df.Define(
                "Muon_pt_corr_resolDown", scarekit_mc_helper_resoldn, scarekit_mc_cols
            )
            # systematic variations: scale from the syst3 (mass-window)
            # spread, resolution from the syst4 (fit-window) spread
            df = df.Define(
                "Muon_pt_corr_scaleSystUp",
                "wrem::varyMuonScarekitScaleMC(Muon_pt_corr, Muon_eta, Muon_phi, Muon_charge, 1.0, true)",
            )
            df = df.Define(
                "Muon_pt_corr_scaleSystDown",
                "wrem::varyMuonScarekitScaleMC(Muon_pt_corr, Muon_eta, Muon_phi, Muon_charge, -1.0, true)",
            )
            df = df.Define(
                "Muon_pt_corr_resolSystUp",
                scarekit_mc_helper_resolsystup,
                scarekit_mc_cols,
            )
            df = df.Define(
                "Muon_pt_corr_resolSystDown",
                scarekit_mc_helper_resolsystdn,
                scarekit_mc_cols,
            )
    else:  # "none"
        df = df.Alias("Muon_pt_corr", "Muon_pt")

    # filter events
    #
    # The efficiency MEASUREMENT run skips this. Two reasons, and the first is a
    # correctness requirement, not a preference:
    #  * the '1HLT' step measures P(HLT_HIMu17 | orthogonal trigger). Applying
    #    HLT_HIMu17 here would make that efficiency 1 by construction, exactly as
    #    measuring the ID after the ID cut would. The step applies its own
    #    denominator instead (HLT_HIL3DoubleMu0, inside
    #    book_event_efficiency_hist).
    #  * it also removes the L3-trigger bias on the per-muon probes: HLT_HIMu17
    #    needs a global-like track, so the muon that fired it is biased toward
    #    passing isGlobal. That bias cancels in the SF (measured: the SF moves by
    #    at most 0.056% when the probe sample is reselected with L3/L2/L1
    #    triggers, all within ~2 sigma of zero), so dropping the filter changes
    #    the per-muon numbers negligibly while making them cleaner.
    if args.muonEfficiencyHists:
        # The measurement cannot apply HLT_HIMu17 (the '1HLT' step would then be
        # 1 by construction), but it must not drop the trigger filter either:
        # without it the full 5.16M events per data file go through the whole
        # selection instead of the ~100k that fire, and the job goes from ~5 min
        # to over 12 h (job 6614264 timed out exactly this way). Filtering on the
        # OR of the numerator and denominator paths keeps every event either
        # measurement needs and still cuts ~97% of the input.
        df = df.Filter(args.muonEffProbeTrigger, "measurement probe trigger")
    else:
        df = df.Filter("HLT_HIMu17")

    # available columns, see: https://cms-xpog.docs.cern.ch/autoDoc/

    # define new columns
    df = df.Define("nLepton", "nElectron + nMuon")

    # ---- Good muons (for Z->mumu selection) ----
    if insituMode:
        if args.sameSign:
            raise ValueError("--sameSign is not defined for the in-situ categories")
        # PR #709 tag-and-probe selection; defines i0 / i1 (negative / positive)
        df = define_insitu_muon_selection(df, dataset)
    # When measuring efficiencies the probe must not already carry the
    # requirement being measured, so the analysis ID is relaxed to loose.
    _muId = (
        "Muon_looseId" if args.muonEfficiencyHists else "Muon_mediumId && Muon_isGlobal"
    )
    if not insituMode:
        df = df.Define(
            "goodMu",
            f"Muon_pt_corr > 18 && abs(Muon_eta) < 2.4 && {_muId}",
        )
        df = df.Define("goodMu_idx", "ROOT::VecOps::Nonzero(goodMu)")
        df = df.Filter("goodMu_idx.size() == 2", "Exactly two good muons")

    # ---- Filter out events with extra electrons ----
    df = df.Filter("nElectron == 0", "No electrons in the event")

    # Opposite sign (or same sign for the fakes control region)
    if insituMode:
        pass  # opposite charge is built into firstMuons / secondMuons
    elif args.sameSign:
        df = df.Define("i0", "int(goodMu_idx[0])").Define("i1", "int(goodMu_idx[1])")
        df = df.Filter("Muon_charge[i0] * Muon_charge[i1] > 0", "Same-sign muons")
    else:
        df = df.Define("i0", "int(goodMu_idx[0])").Define("i1", "int(goodMu_idx[1])")
        df = df.Filter("Muon_charge[i0] * Muon_charge[i1] < 0", "Opposite-sign muons")

    # ---- Build dimuon kinematics ----
    MU_MASS = 0.105658
    df = (
        df.Define(
            "mu0_p4",
            f"ROOT::Math::PtEtaPhiMVector(Muon_pt_corr[i0], Muon_eta[i0], Muon_phi[i0], {MU_MASS})",
        )
        .Define(
            "mu1_p4",
            f"ROOT::Math::PtEtaPhiMVector(Muon_pt_corr[i1], Muon_eta[i1], Muon_phi[i1], {MU_MASS})",
        )
        .Define("dimu_p4", "mu0_p4 + mu1_p4")
        .Define("mll", "dimu_p4.M()")
        .Define("ptll", "dimu_p4.Pt()")
        .Define("yll", "dimu_p4.Rapidity()")
        .Define("absYll", "std::fabs(yll)")
        .Define("phill", "dimu_p4.Phi()")
    )
    df = df.Filter("mll > 76 && mll < 106", "Z mass window")

    # ---- Rank muons: leading/trailing by pT; positive/negative by charge ----
    df = (
        df.Define("i_lead", "Muon_pt_corr[i0] >= Muon_pt_corr[i1] ? i0 : i1")
        .Define("i_trail", "Muon_pt_corr[i0] >= Muon_pt_corr[i1] ? i1 : i0")
        .Define("i_pos", "Muon_charge[i0] > 0 ? i0 : i1")
        .Define("i_neg", "Muon_charge[i0] > 0 ? i1 : i0")
        .Define("muleadpt", "Muon_pt_corr[i_lead]")
        .Define("mutrailpt", "Muon_pt_corr[i_trail]")
        .Define("muleadeta", "Muon_eta[i_lead]")
        .Define("mutraileta", "Muon_eta[i_trail]")
        .Define("mupospt", "Muon_pt_corr[i_pos]")
        .Define("munegpt", "Muon_pt_corr[i_neg]")
        .Define("muposeta", "Muon_eta[i_pos]")
        .Define("munegeta", "Muon_eta[i_neg]")
        .Define("muposphi", "Muon_phi[i_pos]")
        .Define("munegphi", "Muon_phi[i_neg]")
        .Define("mupos_oneOverPt", "1.0/Muon_pt_corr[i_pos]")
        .Define("muneg_oneOverPt", "1.0/Muon_pt_corr[i_neg]")
        .Define("muposcharge", "(double)Muon_charge[i_pos]")
        .Define("munegcharge", "(double)Muon_charge[i_neg]")
        .Define("mupos_nl", "(double)Muon_nTrackerLayers[i_pos]")
        .Define("muneg_nl", "(double)Muon_nTrackerLayers[i_neg]")
        .Define("mupos_masspt", "mll * Muon_pt_corr[i_pos]")
        .Define("muneg_masspt", "mll * Muon_pt_corr[i_neg]")
    )

    # ---- Build CS angles ----
    df = (
        df.Define(
            "mupos_p4",
            f"ROOT::Math::PtEtaPhiMVector(Muon_pt_corr[i_pos], Muon_eta[i_pos], Muon_phi[i_pos], {MU_MASS})",
        )
        .Define(
            "muneg_p4",
            f"ROOT::Math::PtEtaPhiMVector(Muon_pt_corr[i_neg], Muon_eta[i_neg], Muon_phi[i_neg], {MU_MASS})",
        )
        .Define("csSineCosThetaPhill", "wrem::csSineCosThetaPhi(mupos_p4, muneg_p4)")
    )
    df = df.Define("cosThetaStarll", "csSineCosThetaPhill.costheta")
    df = df.Define("phiStarll", "csSineCosThetaPhill.phi()")

    if dataset.is_data:
        df = df.Define("nominal_weight", "1.0")
    else:
        if insituMode:
            # the in-situ helpers need the gen-level uT, so the pre-FSR boson
            # comes first; iteration >= 1 folds W(theta_central) into exp_weight
            df = generator_level_definitions.define_prefsr_vars(df)
            df = define_insitu_mc_uT(df)
            if args.insituSFFile is not None:
                df = define_insitu_central_weight(df)
                df = df.Define("exp_weight", "weight*insituCentralW")
            else:
                df = df.Alias("exp_weight", "weight")
        else:
            df = df.Alias("exp_weight", "weight")
            df = generator_level_definitions.define_prefsr_vars(df)
        df = df.DefinePerSample("theory_weight_truncate", "10.0")
        # the theory-correction denominators are made with the CT18Z central
        # weight applied (gen histmaker runs with --pdfs ct18z), so the reco
        # nominal must carry it too or the pdfas/pdfvars templates pick up a
        # common CT18Z/native shape tilt relative to the nominal
        df = theory_corrections.define_central_pdf_weight(df, dataset.name, "ct18z")
        df = df.Define("nominal_weight_uncorr", "exp_weight*central_pdf_weight")
        applied_theory_corrs = []
        for theory_corr_name in theory_corrs:
            if theory_corr_name not in corr_helpers.get(dataset.name, {}):
                continue
            df = theory_corrections.define_theory_corr_weight_column(
                df, theory_corr_name
            )
            df = df.Define(
                f"{theory_corr_name}Weight_tensor",
                corr_helpers[dataset.name][theory_corr_name],
                [
                    "massVgen",
                    "absYVgen",
                    "ptVgen",
                    "chargeVgen",
                    f"{theory_corr_name}_corr_weight",
                ],
            )
            applied_theory_corrs.append(theory_corr_name)

        # The muon efficiency SF is folded into the definition of
        # nominal_weight rather than Redefine'd onto it: without theory
        # corrections nominal_weight is an Alias, and an alias cannot be
        # redefined.
        _nomw = (
            f"{applied_theory_corrs[0]}Weight_tensor[0]"
            if applied_theory_corrs
            else "exp_weight"
        )
        if muon_event_helper is not None:
            df, _evt_col = muon_efficiencies_5TeV.define_nominal_event_sf_weight(
                df, muon_event_helper
            )
            _nomw = f"{_nomw}*{_evt_col}"
        if muon_trig_helper is not None:
            # define_muon_sf_columns runs here (idempotent-unsafe), so the per-muon
            # ID SF below reuses the same columns rather than redefining them
            df, _trig_col = muon_efficiencies_5TeV.define_nominal_trigger_sf_weight(
                df, muon_trig_helper
            )
            _nomw = f"{_nomw}*{_trig_col}"
        if muon_efficiency_helper is not None:
            df, _sf_col = muon_efficiencies_5TeV.define_nominal_sf_weight(
                df, muon_efficiency_helper, skip_columns=muon_trig_helper is not None
            )
            df = df.Define("nominal_weight", f"{_nomw}*{_sf_col}")
        elif muon_trig_helper is not None or muon_event_helper is not None:
            df = df.Define("nominal_weight", _nomw)
        elif applied_theory_corrs:
            df = df.Define("nominal_weight", _nomw)
        else:
            df = df.Alias("nominal_weight", "exp_weight")

        applied_ew_corrs = []
        for ew_corr_name in ew_theory_corrs:
            helper = ew_corr_helpers.get(dataset.name, {}).get(ew_corr_name)
            if helper is None:
                continue
            if ew_corr_name == "powhegFOEW":
                # corr hist axes are named massVlhe/absYVlhe/cosThetaStarlhe,
                # but the lookup uses the pre-FSR variables (as at 13 TeV)
                ew_cols = ["massVgen", "absYVgen", "csCosThetagen", "chargeVgen"]
            else:
                df = generator_level_definitions.define_ew_vars(df)
                ew_cols = [*helper.hist.axes.name[:-2], "chargeVgen"]
            df = df.Define(
                f"{ew_corr_name}Weight_tensor",
                helper,
                [*ew_cols, "nominal_weight"],
            )
            applied_ew_corrs.append(ew_corr_name)

    # ---- Muon efficiency measurement (opt-in) ----
    if args.muonEfficiencyHists:
        # reco x idip is the factorisation the SF map uses (13 TeV parity);
        # the merged 'id' step is booked too as a free closure test, since
        # eps(id) must equal eps(reco)*eps(idip) bin by bin
        eff_steps = [
            muon_efficiencies_5TeV.cvh_step(is_data=dataset.is_data),
            muon_efficiencies_5TeV.reco_step(),
            muon_efficiencies_5TeV.idip_step(),
            muon_efficiencies_5TeV.id_step(),
        ]
        eff_steps = muon_efficiencies_5TeV.available_steps(
            df, dataset.is_data, eff_steps
        )
        # the '1HLT' step: event level, so it is booked from the event
        # dataframe rather than from a probe collection
        for _evt_step in muon_efficiencies_5TeV.available_steps(
            df, dataset.is_data, [muon_efficiencies_5TeV.trigger_step()]
        ):
            # (a) the event-level map, binned in the fit variables
            muon_efficiencies_5TeV.book_event_efficiency_hist(df, results, _evt_step)
            # (b) the muon-PAIR counts the per-muon fit needs, for full lowPU
            # ('1HLT') parity -- see the comment above trigger_bin_index_expr
            muon_efficiencies_5TeV.book_trigger_pair_hist(df, results, _evt_step)

        for _step in eff_steps:
            df_eff = muon_efficiencies_5TeV.define_probes(df, _step)
            # phi is carried for the cvh step so module-level (eta, phi)
            # structure -- David's glued-module hotspots -- stays visible
            muon_efficiencies_5TeV.book_efficiency_hist(
                df_eff,
                results,
                _step,
                with_phi=(_step.name == "cvh"),
                n_eta=args.muonEffEtaBins,
                n_pt=args.muonEffPtBins,
            )

    # ---- In-situ channels (PR #709) ----
    # Built here so the control plots below show the signal channel
    # (2HLT_passIso), which is what the fit's nominal template is.
    insitu_dfs = None
    if insituMode:
        insitu_dfs = define_insitu_channels(df, dataset, results)
        df = insitu_dfs["nominal"]["df"]

    # ---- Fill histograms ----
    hist_nLepton = df.HistoBoost(
        "nLepton", [axis_nLepton], ["nLepton", "nominal_weight"]
    )
    hist_mll = df.HistoBoost("mll", [axis_mll], ["mll", "nominal_weight"])
    hist_ptll = df.HistoBoost("ptll", [axis_ptll], ["ptll", "nominal_weight"])
    hist_yll = df.HistoBoost("yll", [axis_yll], ["yll", "nominal_weight"])
    hist_phill = df.HistoBoost("phill", [axis_phill], ["phill", "nominal_weight"])

    # Leading/trailing
    hist_mu_lead_pt = df.HistoBoost(
        "muleadpt", [axis_mu_pt], ["muleadpt", "nominal_weight"]
    )
    hist_mu_trail_pt = df.HistoBoost(
        "mutrailpt", [axis_mu_pt], ["mutrailpt", "nominal_weight"]
    )
    hist_mu_lead_eta = df.HistoBoost(
        "muleadeta", [axis_mu_eta], ["muleadeta", "nominal_weight"]
    )
    hist_mu_trail_eta = df.HistoBoost(
        "mutraileta", [axis_mu_eta], ["mutraileta", "nominal_weight"]
    )

    # Positive/negative
    hist_mu_pos_pt = df.HistoBoost(
        "mupospt", [axis_mu_pt], ["mupospt", "nominal_weight"]
    )
    hist_mu_neg_pt = df.HistoBoost(
        "munegpt", [axis_mu_pt], ["munegpt", "nominal_weight"]
    )
    hist_mu_pos_eta = df.HistoBoost(
        "muposeta", [axis_mu_eta], ["muposeta", "nominal_weight"]
    )
    hist_mu_neg_eta = df.HistoBoost(
        "munegeta", [axis_mu_eta], ["munegeta", "nominal_weight"]
    )
    hist_mu_pos_phi = df.HistoBoost(
        "muposphi", [axis_mu_phi], ["muposphi", "nominal_weight"]
    )
    hist_mu_neg_phi = df.HistoBoost(
        "munegphi", [axis_mu_phi], ["munegphi", "nominal_weight"]
    )
    hist_mu_pos_oneOverPt = df.HistoBoost(
        "mupos_oneOverPt", [axis_mu_oneOverPt], ["mupos_oneOverPt", "nominal_weight"]
    )
    hist_mu_neg_oneOverPt = df.HistoBoost(
        "muneg_oneOverPt", [axis_mu_oneOverPt], ["muneg_oneOverPt", "nominal_weight"]
    )
    hist_mu_pos_charge = df.HistoBoost(
        "muposcharge", [axis_mu_charge], ["muposcharge", "nominal_weight"]
    )
    hist_mu_neg_charge = df.HistoBoost(
        "munegcharge", [axis_mu_charge], ["munegcharge", "nominal_weight"]
    )
    hist_mu_pos_nl = df.HistoBoost(
        "mupos_nl", [axis_mu_nl], ["mupos_nl", "nominal_weight"]
    )
    hist_mu_neg_nl = df.HistoBoost(
        "muneg_nl", [axis_mu_nl], ["muneg_nl", "nominal_weight"]
    )
    hist_mu_pos_masspt = df.HistoBoost(
        "mupos_masspt", [axis_mu_masspt], ["mupos_masspt", "nominal_weight"]
    )
    hist_mu_neg_masspt = df.HistoBoost(
        "muneg_masspt", [axis_mu_masspt], ["muneg_masspt", "nominal_weight"]
    )

    # CS angles
    hist_cosThetaStarll = df.HistoBoost(
        "cosThetaStarll", [axis_cosThetaStarll], ["cosThetaStarll", "nominal_weight"]
    )
    hist_phiStarll = df.HistoBoost(
        "phiStarll", [axis_phiStarll], ["phiStarll", "nominal_weight"]
    )

    # ---- Fit channels -------------------------------------------------------
    # Following WRemnants PR #709: every fit template is booked once per channel.
    # By default there is exactly ONE channel -- today's selection with today's
    # histogram names (ptll_vs_yll, ptll_<syst>) -- so the classic analysis is
    # unchanged. The in-situ efficiency mode adds the failIso / failHLT / failID
    # control channels, each with its own axes and histogram base name.
    dfs = {
        "nominal": {
            "df": df,
            "axes": [axis_ptll, axis_yll],
            "cols": ["ptll", "yll"],
            "base": "ptll",
            "nominal_name": "ptll_vs_yll",
        }
    }
    if insitu_dfs is not None:
        dfs = insitu_dfs

    for channel, info in dfs.items():
        df_ch = info["df"]
        axes = info["axes"]
        cols = info["cols"]
        bn = info["base"]
        results.append(
            df_ch.HistoBoost(info["nominal_name"], axes, [*cols, "nominal_weight"])
        )
        # in-situ ID/HLT/Iso Chebyshev variations (pass >= 1, MC only)
        if muon_insitu_efficiency_helper is not None and not dataset.is_data:
            systematics.add_muon_insitu_efficiency_hists(
                results,
                df_ch,
                muon_insitu_efficiency_helper,
                axes,
                cols,
                category=channel,
                base_name=bn,
            )

        # SCETlib-NP param-model response R (PR #701): reco x gen joint yield on
        # the fully selected reco events. Same reco axes as the fit hist;
        # acceptance = the event falls in the gen-fiducial region N_gen is filled
        # on (load_R slices acceptance=True, so reco-passing events outside the
        # gen fiducial are excluded from the fold, as in the 13 TeV setup).
        if channel == "nominal" and not dataset.is_data and "Zmumu" in dataset.name:
            df_ch = df_ch.Define(
                "prefsr_acceptance",
                "genl.pt() > 18 && genlanti.pt() > 18 && "
                "std::fabs(genl.eta()) < 2.4 && std::fabs(genlanti.eta()) < 2.4 && "
                "massVgen > 76 && massVgen < 106",
            )
            df_ch = df_ch.Define("helicitySigUL", "int(-1)")
            results.append(
                df_ch.HistoBoost(
                    "nominal_prefsr_yieldsUnfolding",
                    [
                        axis_ptll,
                        axis_yll,
                        axis_ptVGen,
                        axis_absYVGen,
                        axis_acceptance,
                        axis_helicitySig_ul,
                    ],
                    [
                        "ptll",
                        "yll",
                        "ptVgen",
                        "absYVgen",
                        "prefsr_acceptance",
                        "helicitySigUL",
                        "nominal_weight",
                    ],
                )
            )
        # MINIMUM BIN CONTENT: 95.79483724339086 at bin (ptll index 35, yll index 6) → ptll ∈ [28, 30) GeV, yll ∈ [0.25, 0.5)
        # DATA MINIMUM BIN CONTENT: 88.0 at bin (ptll index 35, yll index 3) → ptll ∈ [28, 30) GeV, yll ∈ [-0.5, -0.25)

        if not dataset.is_data:
            # Muon efficiency template variations, on the same axes as the fit
            # templates. Booked for every MC process, signal and background alike:
            # the SF is applied to all of them, so they all respond to it.
            if muon_trig_helpers_stat:
                df_ch = muon_efficiencies_5TeV.add_muon_trigger_unc_hists_5TeV(
                    results,
                    df_ch,
                    muon_trig_helpers_stat,
                    axes,
                    cols,
                    base_name=bn,
                )

            if muon_event_helpers_stat:
                df_ch = muon_efficiencies_5TeV.add_muon_efficiency_event_unc_hists_5TeV(
                    results,
                    df_ch,
                    muon_event_helpers_stat,
                    muon_event_helper_syst,
                    axes,
                    cols,
                    base_name=bn,
                )

            if muon_efficiency_helpers_stat:
                df_ch = muon_efficiencies_5TeV.add_muon_efficiency_unc_hists_5TeV(
                    results,
                    df_ch,
                    muon_efficiency_helpers_stat,
                    muon_efficiency_helper_syst,
                    axes,
                    cols,
                    base_name=bn,
                )

            if applied_theory_corrs:
                systematics.add_theory_corr_hists(
                    results,
                    df_ch,
                    axes,
                    cols,
                    corr_helpers[dataset.name],
                    theory_corrs,
                    modify_central_weight=True,
                    isW=False,
                    base_name=bn,
                )

            if applied_ew_corrs:
                # EW/FSR variation templates from the borrowed 13 TeV ratio files
                systematics.add_theory_corr_hists(
                    results,
                    df_ch,
                    axes,
                    cols,
                    ew_corr_helpers[dataset.name],
                    applied_ew_corrs,
                    modify_central_weight=False,
                    isW=False,
                    base_name=bn,
                )

            # Helicity-decomposed QCD scale variations (angular coefficients):
            # per-helicity muR/muF envelope from the gen helicity xsecs file,
            # with a coarse ptVgen axis for nuisances decorrelated in ptV
            if qcd_helicity_helper is not None and is_z_mc:
                systematics.add_qcdScaleByHelicityUnc_hist(
                    results,
                    df_ch,
                    qcd_helicity_helper,
                    [*axes, axis_ptVgen_decorr],
                    [*cols, "ptVgen"],
                    base_name=bn,
                )

            # Z boson mass (and width-decorrelated) variations from the MiNNLO
            # Breit-Wigner reweighting weights (MEParamWeight): 21 points in
            # +-100 MeV steps of 10 MeV plus the +-2.1 MeV PDG-uncertainty
            # entries; the fit uses massShiftZ2p1MeVUp/Down as the mZ uncertainty
            if is_z_mc:
                df_ch = systematics.define_mass_width_sin2theta_weights(
                    df_ch, dataset.name
                )
                if df_ch.HasColumn("massWeight_tensor_wnom"):
                    systematics.add_massweights_hist(
                        results,
                        df_ch,
                        axes,
                        cols,
                        base_name=bn,
                        proc=dataset.name,
                    )
                if df_ch.HasColumn("widthWeight_tensor_wnom"):
                    systematics.add_widthweights_hist(
                        results,
                        df_ch,
                        axes,
                        cols,
                        base_name=bn,
                        proc=dataset.name,
                    )
                if df_ch.HasColumn("sin2thetaWeight_tensor_wnom"):
                    systematics.add_sin2thetaweights_hist(
                        results,
                        df_ch,
                        axes,
                        cols,
                        base_name=bn,
                        proc=dataset.name,
                    )

            # b,c quark mass variations (MSHT20nnlo mbrange/mcrange members from
            # LHEPdfWeightAltSet12; same menu as the 13 TeV PDFExt samples: 65
            # central + 7 alpha_s + 9 mcrange @72 + 7 mbrange @81 = 88 entries,
            # verified identical layout - the branch title claims MMHT2014 but is
            # a known gridpack mislabel at both energies). Each member is divided
            # by the range set's own central (member 0), so only the pure mass
            # variation is applied on top of the CT18Z-corrected nominal - the
            # 13 TeV from-MiNNLO scheme (pdfs msht20mb(c)range_renorm, see
            # theory_corrections.define_pdf_columns renorm branch).
            if is_z_mc:
                for pdf_key in ("msht20mbrange_renorm", "msht20mcrange_renorm"):
                    pdf_info = theory_utils.pdfMap[pdf_key]
                    n_entries = pdf_info["entries"]
                    pdf_tensor = f"{pdf_info['name']}Weights_tensor"
                    df_ch = df_ch.Define(
                        pdf_tensor,
                        f"auto res = wrem::vec_to_tensor_t<double, {n_entries}>("
                        f"{pdf_info['branch']}, {pdf_info['first_entry']}); "
                        "res = res / res(0); "
                        "res = wrem::clip_tensor(res, theory_weight_truncate); "
                        "res = res * nominal_weight; return res;",
                    )
                    axis_pdfVar = hist.axis.StrCategory(
                        [f"pdf{i}" for i in range(n_entries)], name="pdfVar"
                    )
                    results.append(
                        df_ch.HistoBoost(
                            f"{bn}_{pdf_info['name']}",
                            axes,
                            [*cols, pdf_tensor],
                            tensor_axes=[axis_pdfVar],
                        )
                    )

            # muon momentum scale/resolution statistical variations (scarekit
            # bootstrap): recompute the dimuon kinematics from the varied muon pT.
            # Selection (incl. the mll window) stays the nominal one; the residual
            # window-migration effect of these ~1e-4 pT shifts is negligible for
            # the ptll-yll templates.
            if args.muonCorr == "scarekit" and cols == ["ptll", "yll"]:
                for var, hname in [
                    ("scaleUp", f"{bn}_muonScaleUp"),
                    ("scaleDown", f"{bn}_muonScaleDown"),
                    ("resolUp", f"{bn}_muonResUp"),
                    ("resolDown", f"{bn}_muonResDown"),
                    ("scaleSystUp", f"{bn}_muonScaleSystUp"),
                    ("scaleSystDown", f"{bn}_muonScaleSystDown"),
                    ("resolSystUp", f"{bn}_muonResSystUp"),
                    ("resolSystDown", f"{bn}_muonResSystDown"),
                ]:
                    ptcol = f"Muon_pt_corr_{var}"
                    df_ch = df_ch.Define(
                        f"dimu_p4_{var}",
                        f"ROOT::Math::PtEtaPhiMVector({ptcol}[i0], Muon_eta[i0], Muon_phi[i0], {MU_MASS})"
                        f" + ROOT::Math::PtEtaPhiMVector({ptcol}[i1], Muon_eta[i1], Muon_phi[i1], {MU_MASS})",
                    )
                    df_ch = df_ch.Define(f"ptll_{var}", f"dimu_p4_{var}.Pt()")
                    df_ch = df_ch.Define(f"yll_{var}", f"dimu_p4_{var}.Rapidity()")
                    results.append(
                        df_ch.HistoBoost(
                            hname,
                            axes,
                            [f"ptll_{var}", f"yll_{var}", "nominal_weight"],
                        )
                    )

    results += [
        hist_mll,
        hist_ptll,
        hist_yll,
        hist_phill,
        hist_nLepton,
        hist_mu_lead_pt,
        hist_mu_trail_pt,
        hist_mu_lead_eta,
        hist_mu_trail_eta,
        hist_mu_pos_pt,
        hist_mu_neg_pt,
        hist_mu_pos_eta,
        hist_mu_neg_eta,
        hist_mu_pos_phi,
        hist_mu_neg_phi,
        hist_mu_pos_oneOverPt,
        hist_mu_neg_oneOverPt,
        hist_mu_pos_charge,
        hist_mu_neg_charge,
        hist_mu_pos_nl,
        hist_mu_neg_nl,
        hist_mu_pos_masspt,
        hist_mu_neg_masspt,
        hist_cosThetaStarll,
        hist_phiStarll,
    ]

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)

args.flavor = "mumu"
fout = f"{os.path.basename(__file__).replace('py', 'hdf5')}"
write_analysis_output(resultdict, fout, args)
