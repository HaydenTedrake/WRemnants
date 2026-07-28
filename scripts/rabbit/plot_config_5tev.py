# Style config for rabbit_plot_pulls_and_impacts.py (--config), 5 TeV alpha_s fit.
# Reuses the 13 TeV impact labels and defines a grouping matching AN Fig. 23.

from wremnants.utilities.styles.styles import impact_labels as _wremnants_labels

impact_labels = dict(_wremnants_labels)
impact_labels.update(
    {
        "massShift": "<i>m</i><sub>Z</sub>",
        "widthZ": "Γ<sub>Z</sub>",
        "sin2thetaZ": "sin<sup>2</sup><i>θ</i><sub>eff</sub>",
        "norm": "Bkg. normalization",
    }
)

nuisance_grouping = {
    # rows of the 13 TeV AN grouped-impacts figure, restricted to groups
    # present in the 5 TeV model (no b,c masses, eff. SFs yet)
    "an13": [
        "Total",
        "theory_ew",
        "stat",
        "binByBinStat",
        "pdfCT18Z",
        "resumTNP",
        "resumNonpert",
        "resumTransitionFOScale",
        "angularCoeffs",
        "bcQuarkMass",
        "luminosity",
        "muonScale",
        "resolutionCrctn",
        "massShift",
        "widthZ",
        "sin2thetaZ",
        "norm",
    ],
}
