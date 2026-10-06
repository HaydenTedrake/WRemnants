"""Make newer NanoAOD productions look like NanoAOD v9 to the histmakers.

From NanoAOD v12 on, CMS stores many integer branches in narrower types (counts
as Int_t instead of UInt_t, indices as Short_t, IDs/flags as UChar_t/UShort_t,
TrigObj_filterBits as ULong64_t). The C++ helpers here are written for the v9
types (e.g. RVec<int> for Muon_nTrackerLayers), and TTreeReader refuses the
mismatch ("Type ambiguity (want unsigned char, have int)"). harmonize_nano_types
redefines exactly the affected columns back to their v9 type, and only when the
type differs, so it is a no-op on v9 input.

Found by comparing v9 and v15 (NanoV15MCLowPU5TeV_v1, Oct 2026) 5.02 TeV MC.
"""

from wums import logging

logger = logging.child_logger(__name__)

# column -> v9 type (as reported by RDataFrame.GetColumnType)
NANO_V9_TYPES = {
    "Electron_cutBased": "ROOT::VecOps::RVec<Int_t>",
    "GenPart_genPartIdxMother": "ROOT::VecOps::RVec<Int_t>",
    "GenPart_statusFlags": "ROOT::VecOps::RVec<Int_t>",
    "Muon_genPartIdx": "ROOT::VecOps::RVec<Int_t>",
    "Muon_nTrackerLayers": "ROOT::VecOps::RVec<Int_t>",
    "Muon_svIdx": "ROOT::VecOps::RVec<Int_t>",
    # only bit 0 (HLT_HIMu17 at 5.02 TeV) is read; narrowing drops bits 32-63
    "TrigObj_filterBits": "ROOT::VecOps::RVec<Int_t>",
    "TrigObj_id": "ROOT::VecOps::RVec<Int_t>",
    "PV_npvsGood": "Int_t",
    "nElectron": "UInt_t",
    "nGenPart": "UInt_t",
    "nLHEPdfWeight": "UInt_t",
    "nMuon": "UInt_t",
    "nTrigObj": "UInt_t",
}

_CPP_TYPE = {"Int_t": "int", "UInt_t": "unsigned int"}


def _norm(typename):
    # RDF spells dataset columns with ROOT typedefs, defined ones with C++ names
    return typename.replace("UInt_t", "unsigned int").replace("Int_t", "int")


def harmonize_nano_types(df):
    """Redefine NanoAOD columns whose type differs from v9 back to the v9 type."""
    columns = {str(c) for c in df.GetColumnNames()}
    changed = []
    for col, v9_type in NANO_V9_TYPES.items():
        if col not in columns:
            continue
        have = str(df.GetColumnType(col))
        if _norm(have) == _norm(v9_type):
            continue
        if v9_type.startswith("ROOT::VecOps::RVec<"):
            elem = _CPP_TYPE[v9_type[len("ROOT::VecOps::RVec<") : -1]]
            expr = f"ROOT::VecOps::RVec<{elem}>({col}.begin(), {col}.end())"
        else:
            expr = f"static_cast<{_CPP_TYPE[v9_type]}>({col})"
        df = df.Redefine(col, expr)
        changed.append(f"{col} ({have} -> {v9_type})")
    if changed:
        logger.info(f"NanoAOD types harmonized to v9: {', '.join(changed)}")
    return df
