"""
This is the Low PU data set taken at 5.020 TeV with an integrated luminosity of about 300/pb
"""

from wremnants.utilities import common

lumicsv = f"{common.data_dir}/bylsoutput_2017G.csv"
lumijson = (
    f"{common.data_dir}/Cert_306546-306826_5TeV_EOY2017ReReco_Collisions17_JSON.txt"
)

# from GenXSecAnalyzer
# Paths carry {NANO_PROD_TAG}; the histmaker picks exactly ONE production via
# --nanoVersion (NANO_PROD_TAGS for MC, NANO_DATA_TAGS for data below), so v9 and
# v15 files are never mixed.
#   v9  NanoV9MC2017_TrackFitV722_NanoProdv3  original production (no muon TrigObj)
#   v15 NanoV15MCLowPU5TeV_v1                 Oct 2026 reproduction: muon TrigObj
#       (HLT_HIMu17 objects, filterBits bit 0) + standalone-muon branches;
#       MET_* renamed PFMET_* (not used by mz_5TeV.py)
NANO_PROD_TAGS = {
    "v9": "NanoV9MC2017_TrackFitV722_NanoProdv3",
    "v15": "NanoV15MCLowPU5TeV_v1",
}
# data lives under a different directory layout per production
NANO_DATA_TAGS = {
    "v9": "LowPU/2017G/SingleMuon/Run2017G-UL2017_MiniAODv2_NanoAODv9_GT36-v2",
    "v15": "SingleMuon/NanoV15Run2017GDataLowPU5TeV_v2",
}

xsec_DYJetsToLL = 698.3  # +/- 2.133
xsec_WplusJetsToLNu = 4477  # +/- 17.27
xsec_WminusJetsToLL = 2940  # +/- 9.153

dataDict = {
    "SingleMuon_2017G": {
        "filepaths": [
            "{BASE_PATH}/{NANO_PROD_TAG}",
        ],
        "group": "Data",
        "lumicsv": lumicsv,
        "lumijson": lumijson,
    },
    "Zmumu_2017G": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToMuMu_H2ErratumFix_PDFExt_TuneCP5_5020GeV-powhegMiNNLO-pythia8-photos/{NANO_PROD_TAG}",
        ],
        "xsec": xsec_DYJetsToLL,
        "group": "Zmumu",
    },
    "Ztautau_2017G": {
        "filepaths": [
            "{BASE_PATH}/DYJetsToTauTau_TauToMuorE_H2ErratumFix_PDFExt_TuneCP5_5020GeV-powhegMiNNLO-pythia8-photos/{NANO_PROD_TAG}",
        ],
        "xsec": xsec_DYJetsToLL * common.Z_TAU_TO_LEP_RATIO,
        "group": "Ztautau",
    },
    "Wplusmunu_2017G": {
        "filepaths": [
            "{BASE_PATH}/WplusJetsToMuNu_H2ErratumFix_PDFExt_TuneCP5_5020GeV-powhegMiNNLO-pythia8-photos/{NANO_PROD_TAG}",
        ],
        "xsec": xsec_WplusJetsToLNu,
        "group": "Wmunu",
    },
    "Wminusmunu_2017G": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToMuNu_H2ErratumFix_PDFExt_TuneCP5_5020GeV-powhegMiNNLO-pythia8-photos/{NANO_PROD_TAG}",
        ],
        "xsec": xsec_WminusJetsToLL,
        "group": "Wmunu",
    },
    "Wplustaunu_2017G": {
        "filepaths": [
            "{BASE_PATH}/WplusJetsToTauNu_TauToMuorE_H2ErratumFix_PDFExt_TuneCP5_5020GeV-powhegMiNNLO-pythia8-photos/{NANO_PROD_TAG}",
        ],
        "xsec": xsec_WplusJetsToLNu * (common.BR_TAUToMU + common.BR_TAUToE),
        "group": "Wtaunu",
    },
    "Wminustaunu_2017G": {
        "filepaths": [
            "{BASE_PATH}/WminusJetsToTauNu_TauToMuorE_H2ErratumFix_PDFExt_TuneCP5_5020GeV-powhegMiNNLO-pythia8-photos/{NANO_PROD_TAG}",
        ],
        "xsec": xsec_WminusJetsToLL * (common.BR_TAUToMU + common.BR_TAUToE),
        "group": "Wtaunu",
    },
}

# The 2017G MC production is PDF-extended by construction (PDFExt samples
# carry the full LHEPdfWeightAltSet* branches, incl. CT18Z in AltSet11),
# so the extended dataset dict is the same as the default one.
dataDict_extended = dataDict
