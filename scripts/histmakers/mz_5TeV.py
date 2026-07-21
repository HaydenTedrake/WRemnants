import math
import os

from wremnants.utilities import binning, common, parsing, samples, theory_utils
from wums import logging

analysis_label = common.analysis_label(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)
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


def build_graph(df, dataset):
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
    df = df.Filter("HLT_HIMu17")

    # available columns, see: https://cms-xpog.docs.cern.ch/autoDoc/

    # define new columns
    df = df.Define("nLepton", "nElectron + nMuon")

    # ---- Good muons (for Z->mumu selection) ----
    df = df.Define(
        "goodMu",
        "Muon_pt_corr > 18 && abs(Muon_eta) < 2.4 && Muon_mediumId && Muon_isGlobal",
    )
    df = df.Define("goodMu_idx", "ROOT::VecOps::Nonzero(goodMu)")
    df = df.Filter("goodMu_idx.size() == 2", "Exactly two good muons")

    # ---- Filter out events with extra electrons ----
    df = df.Filter("nElectron == 0", "No electrons in the event")

    # Opposite sign (or same sign for the fakes control region)
    df = df.Define("i0", "int(goodMu_idx[0])").Define("i1", "int(goodMu_idx[1])")
    if args.sameSign:
        df = df.Filter("Muon_charge[i0] * Muon_charge[i1] > 0", "Same-sign muons")
    else:
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

        if applied_theory_corrs:
            df = df.Define(
                "nominal_weight", f"{applied_theory_corrs[0]}Weight_tensor[0]"
            )
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

    # 2D histograms
    hist_ptll_vs_yll = df.HistoBoost(
        "ptll_vs_yll", [axis_ptll, axis_yll], ["ptll", "yll", "nominal_weight"]
    )

    # SCETlib-NP param-model response R (PR #701): reco x gen joint yield on
    # the fully selected reco events. Same reco axes as the fit hist;
    # acceptance = the event falls in the gen-fiducial region N_gen is filled
    # on (load_R slices acceptance=True, so reco-passing events outside the
    # gen fiducial are excluded from the fold, as in the 13 TeV setup).
    if not dataset.is_data and "Zmumu" in dataset.name:
        df = df.Define(
            "prefsr_acceptance",
            "genl.pt() > 18 && genlanti.pt() > 18 && "
            "std::fabs(genl.eta()) < 2.4 && std::fabs(genlanti.eta()) < 2.4 && "
            "massVgen > 76 && massVgen < 106",
        )
        df = df.Define("helicitySigUL", "int(-1)")
        results.append(
            df.HistoBoost(
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
        if applied_theory_corrs:
            systematics.add_theory_corr_hists(
                results,
                df,
                [axis_ptll, axis_yll],
                ["ptll", "yll"],
                corr_helpers[dataset.name],
                theory_corrs,
                modify_central_weight=True,
                isW=False,
                base_name="ptll",
            )

        if applied_ew_corrs:
            # EW/FSR variation templates from the borrowed 13 TeV ratio files
            systematics.add_theory_corr_hists(
                results,
                df,
                [axis_ptll, axis_yll],
                ["ptll", "yll"],
                ew_corr_helpers[dataset.name],
                applied_ew_corrs,
                modify_central_weight=False,
                isW=False,
                base_name="ptll",
            )

        # Helicity-decomposed QCD scale variations (angular coefficients):
        # per-helicity muR/muF envelope from the gen helicity xsecs file,
        # with a coarse ptVgen axis for nuisances decorrelated in ptV
        if qcd_helicity_helper is not None and is_z_mc:
            systematics.add_qcdScaleByHelicityUnc_hist(
                results,
                df,
                qcd_helicity_helper,
                [axis_ptll, axis_yll, axis_ptVgen_decorr],
                ["ptll", "yll", "ptVgen"],
                base_name="ptll",
            )

        # Z boson mass (and width-decorrelated) variations from the MiNNLO
        # Breit-Wigner reweighting weights (MEParamWeight): 21 points in
        # +-100 MeV steps of 10 MeV plus the +-2.1 MeV PDG-uncertainty
        # entries; the fit uses massShiftZ2p1MeVUp/Down as the mZ uncertainty
        if is_z_mc:
            df = systematics.define_mass_width_sin2theta_weights(df, dataset.name)
            if df.HasColumn("massWeight_tensor_wnom"):
                systematics.add_massweights_hist(
                    results,
                    df,
                    [axis_ptll, axis_yll],
                    ["ptll", "yll"],
                    base_name="ptll",
                    proc=dataset.name,
                )
            if df.HasColumn("widthWeight_tensor_wnom"):
                systematics.add_widthweights_hist(
                    results,
                    df,
                    [axis_ptll, axis_yll],
                    ["ptll", "yll"],
                    base_name="ptll",
                    proc=dataset.name,
                )
            if df.HasColumn("sin2thetaWeight_tensor_wnom"):
                systematics.add_sin2thetaweights_hist(
                    results,
                    df,
                    [axis_ptll, axis_yll],
                    ["ptll", "yll"],
                    base_name="ptll",
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
                df = df.Define(
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
                    df.HistoBoost(
                        f"ptll_{pdf_info['name']}",
                        [axis_ptll, axis_yll],
                        ["ptll", "yll", pdf_tensor],
                        tensor_axes=[axis_pdfVar],
                    )
                )

        # muon momentum scale/resolution statistical variations (scarekit
        # bootstrap): recompute the dimuon kinematics from the varied muon pT.
        # Selection (incl. the mll window) stays the nominal one; the residual
        # window-migration effect of these ~1e-4 pT shifts is negligible for
        # the ptll-yll templates.
        if args.muonCorr == "scarekit":
            for var, hname in [
                ("scaleUp", "ptll_muonScaleUp"),
                ("scaleDown", "ptll_muonScaleDown"),
                ("resolUp", "ptll_muonResUp"),
                ("resolDown", "ptll_muonResDown"),
                ("scaleSystUp", "ptll_muonScaleSystUp"),
                ("scaleSystDown", "ptll_muonScaleSystDown"),
                ("resolSystUp", "ptll_muonResSystUp"),
                ("resolSystDown", "ptll_muonResSystDown"),
            ]:
                ptcol = f"Muon_pt_corr_{var}"
                df = df.Define(
                    f"dimu_p4_{var}",
                    f"ROOT::Math::PtEtaPhiMVector({ptcol}[i0], Muon_eta[i0], Muon_phi[i0], {MU_MASS})"
                    f" + ROOT::Math::PtEtaPhiMVector({ptcol}[i1], Muon_eta[i1], Muon_phi[i1], {MU_MASS})",
                )
                df = df.Define(f"ptll_{var}", f"dimu_p4_{var}.Pt()")
                df = df.Define(f"yll_{var}", f"dimu_p4_{var}.Rapidity()")
                results.append(
                    df.HistoBoost(
                        hname,
                        [axis_ptll, axis_yll],
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
        hist_ptll_vs_yll,
    ]

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)

args.flavor = "mumu"
fout = f"{os.path.basename(__file__).replace('py', 'hdf5')}"
write_analysis_output(resultdict, fout, args)
