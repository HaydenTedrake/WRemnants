#ifndef WREMNANTS_MUON_EFFICIENCIES_5TEV_H
#define WREMNANTS_MUON_EFFICIENCIES_5TEV_H

#include <ROOT/RVec.hxx>
#include <algorithm>
#include <boost/histogram/axis.hpp>
#include <cmath>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <string>

// Muon efficiency scale factors and their template variations for the 5.02 TeV
// (2017G) low-PU Z -> mumu analysis.
//
// Same three-helper structure as muon_efficiencies_binned.hpp / _smooth.hpp --
// nominal weight, effSystTnP, effStatTnP -- so the downstream booking and the
// datacard nuisance naming carry over unchanged. What differs, and why:
//
//  * ONE SF MAP, indexed by a step axis, instead of the three histograms
//    (reco / tracking / idip-trigger-iso) the 13 TeV helpers juggle. Those three
//    exist only because the 13 TeV steps are measured on different pt binnings;
//    here every step shares the binning of muon_efficiencies_5TeV.build_axes, so
//    a step axis is enough and the number of steps is a runtime property of the
//    histogram rather than a template parameter.
//
//  * NO uT AXIS. The 13 TeV smooth SFs can bin trigger and isolation in the
//    hadronic recoil (smooth3D); no entry point here takes a uT.
//
//  * NO tag/probe ASYMMETRY. The 13 TeV dilepton helpers take trigMuons_* and
//    nonTrigMuons_* separately and skip the non-triggering muon for the trigger
//    step. The steps measured here are uncorrelated with the trigger (see the
//    module docstring), so both muons of the Z are ordinary probes and their SFs
//    simply multiply.
//
//  * NO CHARGE SPLIT FORCED. NCharges is a template parameter: pass 1 with a
//    single-bin charge axis for a charge-inclusive measurement, 2 to let the
//    charges vary independently. 13 TeV makes that choice per step through
//    common.muonEfficiency_chargeDependentSteps; at 0.3/fb there are not enough
//    probes to split, so 1 is the expected value.
//
// The SF map is written by scripts/analysisTools/make_muon_efficiency_sf_5TeV.py
// out of the pass/fail histograms that mz_5TeV.py --muonEfficiencyHists fills.
// Its cell value is SF = eps_data/eps_MC and its variance is the propagated
// binomial variance of that ratio, which is what sf_stat_var reads below.
//
// Axis layout, fixed by that script:
//   <0> eta   <1> pt   <2> charge   <3> step (string category)   <4> nom/alt
// Slot 0 of <4> is the nominal map; slot 1 is an alternative measurement used by
// the syst helper. When no alternative exists the factory refuses to build the
// syst helper at all rather than handing the fit a nuisance that does nothing.

namespace wrem {

template <typename HIST_SF> class muon_efficiency_5TeV_helper_base {
public:
  muon_efficiency_5TeV_helper_base(HIST_SF &&sf)
      : sf_(std::make_shared<const HIST_SF>(std::move(sf))) {}

  // Bin indices of one muon. Kept separate from the lookup so that the several
  // per-step lookups of one muon pay for the axis search only once.
  struct MuonBins {
    int eta_idx, pt_idx, charge_idx;
  };

  MuonBins muon_bins(float pt, float eta, int charge) const {
    return MuonBins{sf_->template axis<0>().index(eta),
                    sf_->template axis<1>().index(pt),
                    // a single-bin charge axis maps both charges onto bin 0
                    nCharges() == 1 ? 0 : sf_->template axis<2>().index(charge)};
  }

  int nSteps() const { return sf_->template axis<3>().size(); }
  int nCharges() const { return sf_->template axis<2>().size(); }

  double sf_one_step(const MuonBins &b, int step_idx, int nom_alt) const {
    return sf_->at(b.eta_idx, b.pt_idx, b.charge_idx, step_idx, nom_alt).value();
  }

  // All steps of one muon, multiplied.
  double sf_one_muon(const MuonBins &b, int nom_alt) const {
    double sf = 1.0;
    for (int s = 0; s < nSteps(); ++s)
      sf *= sf_one_step(b, s, nom_alt);
    return sf;
  }

  // The event weight: every step, both muons of the Z candidate.
  double sf_product(float pt1, float eta1, int charge1, float pt2, float eta2,
                    int charge2, int nom_alt) const {
    return sf_one_muon(muon_bins(pt1, eta1, charge1), nom_alt) *
           sf_one_muon(muon_bins(pt2, eta2, charge2), nom_alt);
  }

  // Index of a named step, -1 when absent. The string category lookup is slow,
  // so callers cache the result.
  int step_index(const std::string &name) const {
    const auto &ax = sf_->template axis<3>();
    for (int i = 0; i < ax.size(); ++i)
      if (ax.value(i) == name)
        return i;
    return -1;
  }

protected:
  std::shared_ptr<const HIST_SF> sf_;
  static constexpr int idx_nom_ = 0;
  static constexpr int idx_alt_ = 1;
};

//// NOMINAL: the per-event scale factor, to be multiplied into nominal_weight.

template <typename HIST_SF>
class muon_efficiency_5TeV_helper
    : public muon_efficiency_5TeV_helper_base<HIST_SF> {
public:
  using base_t = muon_efficiency_5TeV_helper_base<HIST_SF>;
  using base_t::base_t;

  double operator()(float pt1, float eta1, int charge1, float pt2, float eta2,
                    int charge2) const {
    return base_t::sf_product(pt1, eta1, charge1, pt2, eta2, charge2,
                              base_t::idx_nom_);
  }
};

//// EFFSYST: one fully correlated variation per step.
//
// Element s is the event weight ratio when step s alone is moved to the
// alternative map, all other steps staying nominal. Correlated across (eta, pt)
// and across the two muons by construction, which is the whole point of
// separating it from the per-bin stat variations below.

template <int NSteps, typename HIST_SF>
class muon_efficiency_5TeV_helper_syst
    : public muon_efficiency_5TeV_helper_base<HIST_SF> {
public:
  using base_t = muon_efficiency_5TeV_helper_base<HIST_SF>;
  using base_t::base_t;
  using tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NSteps>>;

  tensor_t operator()(float pt1, float eta1, int charge1, float pt2, float eta2,
                      int charge2, double nominal_weight = 1.0) const {
    tensor_t res;
    res.setConstant(nominal_weight);

    const auto b1 = base_t::muon_bins(pt1, eta1, charge1);
    const auto b2 = base_t::muon_bins(pt2, eta2, charge2);

    for (int s = 0; s < NSteps; ++s) {
      const double nom = base_t::sf_one_step(b1, s, base_t::idx_nom_) *
                         base_t::sf_one_step(b2, s, base_t::idx_nom_);
      const double alt = base_t::sf_one_step(b1, s, base_t::idx_alt_) *
                         base_t::sf_one_step(b2, s, base_t::idx_alt_);
      // only step s moves, so the other steps cancel in the ratio and never
      // have to be looked up
      if (nom > 0.0)
        res(s) *= alt / nom;
    }
    return res;
  }
};

//// EFFSTAT: one variation per (eta, pt, charge) bin of ONE step.
//
// One helper instance per step, the step being fixed at construction, so each
// step gets its own independent block of nuisances -- the same layout as
// helper_stat being a dict keyed by step in the 13 TeV factories.
//
// Element (i, j, k) is the event weight ratio when the SF of bin (i, j, k) is
// raised by its statistical error. A muon outside that bin contributes 1, so
// for most bins the tensor is 1 and only the two bins the muons fall into move;
// when BOTH muons land in the same bin the shift is applied twice, which is the
// correct response of a product of two SFs to one shifted map cell.

template <int NEtaBins, int NPtBins, int NCharges, typename HIST_SF>
class muon_efficiency_5TeV_helper_stat
    : public muon_efficiency_5TeV_helper_base<HIST_SF> {
public:
  using base_t = muon_efficiency_5TeV_helper_base<HIST_SF>;
  using tensor_t =
      Eigen::TensorFixedSize<double, Eigen::Sizes<NEtaBins, NPtBins, NCharges>>;

  muon_efficiency_5TeV_helper_stat(HIST_SF &&sf, int step_idx)
      : base_t(std::move(sf)), step_idx_(step_idx) {}

  tensor_t operator()(float pt1, float eta1, int charge1, float pt2, float eta2,
                      int charge2, double nominal_weight = 1.0) const {
    tensor_t res;
    res.setConstant(1.0);
    fill_one_muon(res, pt1, eta1, charge1);
    fill_one_muon(res, pt2, eta2, charge2);
    return nominal_weight * res;
  }

private:
  // *=, not =, so that two muons in the same bin both shift it.
  void fill_one_muon(tensor_t &res, float pt, float eta, int charge) const {
    const auto b = base_t::muon_bins(pt, eta, charge);
    const auto &cell = base_t::sf_->at(b.eta_idx, b.pt_idx, b.charge_idx,
                                       step_idx_, base_t::idx_nom_);
    const double value = cell.value();
    if (!(value > 0.0))
      return; // empty map cell: leave the variation at 1
    const double shifted = value + std::sqrt(cell.variance());
    // overflow and underflow are folded onto the adjacent bin, as the tensor
    // has no flow bins to put them in
    res(std::clamp(b.eta_idx, 0, NEtaBins - 1),
        std::clamp(b.pt_idx, 0, NPtBins - 1),
        std::clamp(b.charge_idx, 0, NCharges - 1)) *= shifted / value;
  }

  int step_idx_;
};

//// EVENT-LEVEL STEPS (the single-muon trigger).
//
// HLT_HIMu17 is required of the EVENT (mz_5TeV.py: df.Filter("HLT_HIMu17")), not
// of either muon, so the quantity the analysis needs is
//     P(event fires | Z event with these two muons),
// one factor per event rather than one per muon. That is why these three classes
// exist alongside the per-muon ones above: same three-helper contract, same SF
// map layout, but ONE lookup instead of two.
//
// WHY NOT A PER-MUON TRIGGER EFFICIENCY: assigning "this muon fired" needs the
// HLT object that caused the decision, matched in (eta, phi). This production
// stores no muon trigger objects at all (TrigObj_id has no entry 13 in 5.16M
// events; only id 11, electrons). Muon_triggerIdLoose is an offline QUALITY flag
// chosen to mirror the HLT muon ID, not a record that the muon fired, and the
// L1_* branches are event-level seed decisions with no per-muon objects.
//
// The map is therefore binned in the FIT variables rather than in muon
// kinematics: axis<0> and axis<1> hold (yll, ptll) in place of (eta, pt), and the
// charge axis carries a single dummy bin. That keeps the base class, the SF file
// format and the stat/syst machinery identical, and it bins the correction in
// exactly the variables whose templates it can distort.

template <typename HIST_SF>
class muon_efficiency_5TeV_event_helper
    : public muon_efficiency_5TeV_helper_base<HIST_SF> {
public:
  using base_t = muon_efficiency_5TeV_helper_base<HIST_SF>;
  using base_t::base_t;

  // x0, x1 are whatever the map's first two axes are -- (yll, ptll) as built by
  // make_muon_efficiency_event_helpers_5TeV. Note the argument order matches the
  // per-muon helper's (pt, eta) convention: x0 goes to axis<1>, x1 to axis<0>.
  double operator()(float x0, float x1) const {
    return base_t::sf_one_muon(base_t::muon_bins(x0, x1, 0), base_t::idx_nom_);
  }
};

template <int NSteps, typename HIST_SF>
class muon_efficiency_5TeV_event_helper_syst
    : public muon_efficiency_5TeV_helper_base<HIST_SF> {
public:
  using base_t = muon_efficiency_5TeV_helper_base<HIST_SF>;
  using base_t::base_t;
  using tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NSteps>>;

  tensor_t operator()(float x0, float x1, double nominal_weight = 1.0) const {
    tensor_t res;
    res.setConstant(nominal_weight);
    const auto b = base_t::muon_bins(x0, x1, 0);
    for (int s = 0; s < NSteps; ++s) {
      const double nom = base_t::sf_one_step(b, s, base_t::idx_nom_);
      const double alt = base_t::sf_one_step(b, s, base_t::idx_alt_);
      if (nom > 0.0)
        res(s) *= alt / nom;
    }
    return res;
  }
};

template <int N0, int N1, typename HIST_SF>
class muon_efficiency_5TeV_event_helper_stat
    : public muon_efficiency_5TeV_helper_base<HIST_SF> {
public:
  using base_t = muon_efficiency_5TeV_helper_base<HIST_SF>;
  using tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<N0, N1>>;

  muon_efficiency_5TeV_event_helper_stat(HIST_SF &&sf, int step_idx)
      : base_t(std::move(sf)), step_idx_(step_idx) {}

  tensor_t operator()(float x0, float x1, double nominal_weight = 1.0) const {
    tensor_t res;
    res.setConstant(1.0);
    const auto b = base_t::muon_bins(x0, x1, 0);
    const auto &cell =
        base_t::sf_->at(b.eta_idx, b.pt_idx, 0, step_idx_, base_t::idx_nom_);
    const double value = cell.value();
    if (value > 0.0) {
      // one factor per event, so = rather than the *= the per-muon helper needs
      res(std::clamp(b.eta_idx, 0, N0 - 1), std::clamp(b.pt_idx, 0, N1 - 1)) =
          (value + std::sqrt(cell.variance())) / value;
    }
    return nominal_weight * res;
  }

private:
  int step_idx_;
};

//// PER-MUON TRIGGER (full lowPU / '1HLT' parity).
//
// lowPU (lowpu_efficiencies.hpp::lepSF_HLT_q) applies a per-muon trigger
// efficiency through
//     corr = (1 - prod_i (1 - eps_data_i)) / (1 - prod_i (1 - eps_MC_i))
//          = P(at least one muon fired)_data / P(...)_MC
// which is NON-LINEAR in eps, so eps_data and eps_MC have to be carried
// separately -- a single SF ratio per bin cannot reproduce it. These classes do
// exactly that combination for the two muons of the Z.
//
// The maps come from trigger_efficiency_fit_5TeV.py, which recovers eps per muon
// from the event decision alone (no muon trigger objects exist in this
// production). Because only the event decision is observed, the per-bin
// efficiencies are strongly correlated, so the STATISTICAL variations are
// EIGENVECTORS of the fit covariance rather than one nuisance per bin -- the
// representation the 13 TeV smooth scale factors also use. Each eigen mode is a
// full (eta, pt, charge) shift vector applied to BOTH muons coherently, which is
// what makes it a single nuisance.
//
// Two histograms are held:
//   eps : axes <0> eta <1> pt <2> charge <3> kind(data,MC)   value = eps
//   eig : axes <0> eta <1> pt <2> charge <3> kind <4> mode   value = +1 sigma
//         shift of eps along that mode

template <typename HIST_EPS, typename HIST_EIG>
class muon_efficiency_5TeV_trigger_base {
public:
  muon_efficiency_5TeV_trigger_base(HIST_EPS &&eps, HIST_EIG &&eig)
      : eps_(std::make_shared<const HIST_EPS>(std::move(eps))),
        eig_(std::make_shared<const HIST_EIG>(std::move(eig))) {}

  static constexpr int kData = 0;
  static constexpr int kMC = 1;

  struct Bins {
    int eta_idx, pt_idx, charge_idx;
  };

  Bins bins(float pt, float eta, int charge) const {
    return Bins{eps_->template axis<0>().index(eta),
                eps_->template axis<1>().index(pt),
                eps_->template axis<2>().size() == 1
                    ? 0
                    : eps_->template axis<2>().index(charge)};
  }

  double eps_at(const Bins &b, int kind) const {
    return eps_->at(b.eta_idx, b.pt_idx, b.charge_idx, kind).value();
  }

  double shift_at(const Bins &b, int kind, int mode) const {
    return eig_->at(b.eta_idx, b.pt_idx, b.charge_idx, kind, mode).value();
  }

  // P(at least one of the two fired), clamped so a pathological map cannot
  // produce a non-positive denominator downstream
  static double at_least_one(double e1, double e2) {
    return std::clamp(1.0 - (1.0 - e1) * (1.0 - e2), 1e-9, 1.0);
  }

  double correction(const Bins &b1, const Bins &b2, double d1 = 0.0,
                    double d2 = 0.0, double m1 = 0.0, double m2 = 0.0) const {
    // d*, m* are additive shifts on eps_data / eps_MC for the two muons
    const double pd =
        at_least_one(std::clamp(eps_at(b1, kData) + d1, 0.0, 1.0),
                     std::clamp(eps_at(b2, kData) + d2, 0.0, 1.0));
    const double pm =
        at_least_one(std::clamp(eps_at(b1, kMC) + m1, 0.0, 1.0),
                     std::clamp(eps_at(b2, kMC) + m2, 0.0, 1.0));
    return pd / pm;
  }

protected:
  std::shared_ptr<const HIST_EPS> eps_;
  std::shared_ptr<const HIST_EIG> eig_;
};

template <typename HIST_EPS, typename HIST_EIG>
class muon_efficiency_5TeV_trigger_helper
    : public muon_efficiency_5TeV_trigger_base<HIST_EPS, HIST_EIG> {
public:
  using base_t = muon_efficiency_5TeV_trigger_base<HIST_EPS, HIST_EIG>;
  using base_t::base_t;

  double operator()(float pt1, float eta1, int charge1, float pt2, float eta2,
                    int charge2) const {
    return base_t::correction(base_t::bins(pt1, eta1, charge1),
                              base_t::bins(pt2, eta2, charge2));
  }
};

// NModes eigen variations of eps_data and, separately, of eps_MC. Two helper
// instances are made -- one per kind -- so the data and MC statistical
// uncertainties stay independent nuisances, as they are in lowPU.
template <int NModes, typename HIST_EPS, typename HIST_EIG>
class muon_efficiency_5TeV_trigger_helper_stat
    : public muon_efficiency_5TeV_trigger_base<HIST_EPS, HIST_EIG> {
public:
  using base_t = muon_efficiency_5TeV_trigger_base<HIST_EPS, HIST_EIG>;
  using tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NModes>>;

  muon_efficiency_5TeV_trigger_helper_stat(HIST_EPS &&eps, HIST_EIG &&eig,
                                           int kind)
      : base_t(std::move(eps), std::move(eig)), kind_(kind) {}

  tensor_t operator()(float pt1, float eta1, int charge1, float pt2, float eta2,
                      int charge2, double nominal_weight = 1.0) const {
    tensor_t res;
    const auto b1 = base_t::bins(pt1, eta1, charge1);
    const auto b2 = base_t::bins(pt2, eta2, charge2);
    const double nom = base_t::correction(b1, b2);
    for (int k = 0; k < NModes; ++k) {
      const double s1 = base_t::shift_at(b1, kind_, k);
      const double s2 = base_t::shift_at(b2, kind_, k);
      // the mode shifts BOTH muons coherently -- that is what makes it one
      // nuisance rather than one per bin
      const double var =
          (kind_ == base_t::kData)
              ? base_t::correction(b1, b2, s1, s2, 0.0, 0.0)
              : base_t::correction(b1, b2, 0.0, 0.0, s1, s2);
      res(k) = nominal_weight * (nom > 0.0 ? var / nom : 1.0);
    }
    return res;
  }

private:
  int kind_;
};

} // namespace wrem

#endif
