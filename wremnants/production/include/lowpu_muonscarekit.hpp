#ifndef WREMNANTS_LOWPU_MUONSCAREKIT_H
#define WREMNANTS_LOWPU_MUONSCAREKIT_H

#include "defines.hpp"
#include <TFile.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TMath.h>
#include <boost/math/special_functions/erf.hpp>
#include <random>
#include <string>
#include <vector>

namespace wrem {

struct MuonScarekitCB {
  static const double pi;
  static const double sqrtPiOver2;
  static const double sqrt2;

  double m, s, a, n;
  double B, C, D, N, NA, Ns, NC, F, G, k;
  double cdfMa, cdfPa;

  MuonScarekitCB(double mean, double sigma, double alpha, double nn)
      : m(mean), s(sigma), a(alpha), n(nn) {
    init();
  }

  void init() {
    double fa = fabs(a);
    double ex = exp(-fa * fa / 2);
    double A = pow(n / fa, n) * ex;
    double C1 = n / fa / (n - 1) * ex;
    double D1 = 2 * sqrtPiOver2 * boost::math::erf(fa / sqrt2);
    B = n / fa - fa;
    C = (D1 + 2 * C1) / C1;
    D = (D1 + 2 * C1) / 2;
    N = 1.0 / s / (D1 + 2 * C1);
    k = 1.0 / (n - 1);
    NA = N * A;
    Ns = N * s;
    NC = Ns * C1;
    F = 1 - fa * fa / n;
    G = s * n / fa;
    cdfMa = cdf(m - a * s);
    cdfPa = cdf(m + a * s);
  }

  double cdf(double x) const {
    double d = (x - m) / s;
    if (d < -a)
      return NC / pow(F - s * d / G, n - 1);
    if (d > a)
      return NC * (C - pow(F + s * d / G, 1 - n));
    return Ns * (D - sqrtPiOver2 * boost::math::erf(-d / sqrt2));
  }

  double invcdf(double u) const {
    if (u < cdfMa)
      return m + G * (F - pow(NC / u, k));
    if (u > cdfPa)
      return m - G * (F - pow(C - u / NC, -k));
    return m - sqrt2 * s * boost::math::erf_inv((D - u / Ns) / sqrtPiOver2);
  }
};

const double MuonScarekitCB::pi = 3.14159265358979;
const double MuonScarekitCB::sqrtPiOver2 = sqrt(MuonScarekitCB::pi / 2.0);
const double MuonScarekitCB::sqrt2 = sqrt(2.0);

namespace muonscarekit_impl {
TFile *tf_scale = TFile::Open(
    "wremnants-data/data/lowPU/muonscarekit/step3_correction.root", "READ");
TH2D *h_M_DATA = (TH2D *)tf_scale->Get("M_DATA");
TH2D *h_A_DATA = (TH2D *)tf_scale->Get("A_DATA");
TH2D *h_M_SIG = (TH2D *)tf_scale->Get("M_SIG");
TH2D *h_A_SIG = (TH2D *)tf_scale->Get("A_SIG");

TFile *tf_cb = TFile::Open(
    "wremnants-data/data/lowPU/muonscarekit/step2_fitresults.root", "READ");
TH3D *h_cb = (TH3D *)tf_cb->Get("h_results_cb");
TH3D *h_poly = (TH3D *)tf_cb->Get("h_results_poly");

TFile *tf_k =
    TFile::Open("wremnants-data/data/lowPU/muonscarekit/step4_k.root", "READ");
TH2D *h_k_data = (TH2D *)tf_k->Get("k_hist_DATA");
TH2D *h_k_sig = (TH2D *)tf_k->Get("k_hist_SIG");

// Bootstrap (statistical) uncertainties of the corrections; optional inputs,
// variations are disabled (return nominal) when the files are absent.
// Contents follow muonscarekit python/corrections/uncertainties.py: bin error
// = bootstrap std of the sim->data kappa (M) / lambda (A) corrections and of
// the resolution factor k; "correlation" holds the per-bin M-A correlation.
TFile *tf_scale_unc = TFile::Open("wremnants-data/data/lowPU/muonscarekit/"
                                  "scale_step3_correction_uncertainty.root",
                                  "READ");
TH2D *h_M_unc = tf_scale_unc ? (TH2D *)tf_scale_unc->Get("M") : nullptr;
TH2D *h_A_unc = tf_scale_unc ? (TH2D *)tf_scale_unc->Get("A") : nullptr;
TH2D *h_rho_unc =
    tf_scale_unc ? (TH2D *)tf_scale_unc->Get("correlation") : nullptr;

TFile *tf_res_unc = TFile::Open(
    "wremnants-data/data/lowPU/muonscarekit/resolution_uncertainty.root",
    "READ");
TH1D *h_k_unc = tf_res_unc ? (TH1D *)tf_res_unc->Get("k_hist") : nullptr;

// Systematic uncertainties of the corrections, same file layout as the
// bootstrap ones: scale from the scarekit --syst 3 campaign (spread of the
// step3 kappa/lambda corrections over scale-fit mass-window variations),
// resolution from --syst 4 (spread of k over resolution-fit window
// variations). Optional inputs like the stat ones.
TFile *tf_scale_unc_syst =
    TFile::Open("wremnants-data/data/lowPU/muonscarekit/"
                "syst3_scale_step3_correction_uncertainty.root",
                "READ");
TH2D *h_M_unc_syst =
    tf_scale_unc_syst ? (TH2D *)tf_scale_unc_syst->Get("M") : nullptr;
TH2D *h_A_unc_syst =
    tf_scale_unc_syst ? (TH2D *)tf_scale_unc_syst->Get("A") : nullptr;
TH2D *h_rho_unc_syst =
    tf_scale_unc_syst ? (TH2D *)tf_scale_unc_syst->Get("correlation") : nullptr;

TFile *tf_res_unc_syst = TFile::Open(
    "wremnants-data/data/lowPU/muonscarekit/syst4_resolution_uncertainty.root",
    "READ");
TH1D *h_k_unc_syst =
    tf_res_unc_syst ? (TH1D *)tf_res_unc_syst->Get("k_hist") : nullptr;
} // namespace muonscarekit_impl

Vec_f applyMuonScarekitData(Vec_f pt, Vec_f eta, Vec_f phi, Vec_i charge) {
  using namespace muonscarekit_impl;
  unsigned int size = pt.size();
  Vec_f res(size);
  for (unsigned int i = 0; i < size; ++i) {
    double M = h_M_DATA->GetBinContent(h_M_DATA->FindBin(eta[i], phi[i]));
    double A = h_A_DATA->GetBinContent(h_A_DATA->FindBin(eta[i], phi[i]));
    res[i] = static_cast<float>(1.0 / (M / pt[i] + charge[i] * A));
  }
  return res;
}

// Scale (kappa/lambda) statistical variation on the fully corrected MC pt,
// following muonscarekit apply_corrections.uncertainties: the pt shift from
// the bootstrap stds of M and A with their per-bin correlation,
// delta(pt) = pt^2 * sqrt(dM^2/pt^2 + dA^2 + 2*q*rho*dM*dA/pt).
// updn = +1/-1 selects the up/down variation; syst=true reads the syst3
// (mass-window) spread instead of the bootstrap one; nominal pt is returned
// when the uncertainty file is not available.
Vec_f varyMuonScarekitScaleMC(Vec_f pt_corr, Vec_f eta, Vec_f phi, Vec_i charge,
                              double updn, bool syst = false) {
  using namespace muonscarekit_impl;
  TH2D *hM = syst ? h_M_unc_syst : h_M_unc;
  TH2D *hA = syst ? h_A_unc_syst : h_A_unc;
  TH2D *hR = syst ? h_rho_unc_syst : h_rho_unc;
  unsigned int size = pt_corr.size();
  Vec_f res(size);
  for (unsigned int i = 0; i < size; ++i) {
    if (hM == nullptr || hA == nullptr || hR == nullptr) {
      res[i] = pt_corr[i];
      continue;
    }
    const int etabin = hM->GetXaxis()->FindBin(eta[i]);
    const int phibin = hM->GetYaxis()->FindBin(phi[i]);
    const double dM = hM->GetBinError(etabin, phibin);
    const double dA = hA->GetBinError(etabin, phibin);
    const double rho = hR->GetBinContent(etabin, phibin);
    const double pt = pt_corr[i];
    const double num =
        dM * dM / (pt * pt) + dA * dA + 2.0 * charge[i] * rho * dM / pt * dA;
    const double dpt = pt * pt * std::sqrt(std::max(num, 0.0));
    res[i] = static_cast<float>(pt + updn * dpt);
  }
  return res;
}

class MuonScarekitMCHelper {

public:
  // k_unc_shift shifts the MC smearing factor k by that many stds of the
  // stat (bootstrap, resolution_uncertainty.root) or, with k_unc_syst=true,
  // syst4 (window-variation) uncertainty; the per-event RNG seeding
  // guarantees the same smearing random number as the nominal helper, so the
  // shifted helper yields a consistent resolution up/down variation.
  MuonScarekitMCHelper(const std::size_t seed = 0,
                       const double k_unc_shift = 0.0,
                       const bool k_unc_syst = false)
      : hash_(std::hash<std::string>()("MuonScarekitMCHelper")), seed_(seed),
        k_unc_shift_(k_unc_shift), k_unc_syst_(k_unc_syst) {}

  // Per-event seeding (run, lumi, event): reproducible across runs and thread
  // counts, and thread-safe (RNG is local to each call). Mirrors
  // wrem::SmearingHelper in muon_calibration.hpp.
  Vec_f operator()(const unsigned int run, const unsigned int lumi,
                   const unsigned long long event, Vec_f pt, Vec_f eta,
                   Vec_f phi, Vec_i charge, Vec_i nTrackerLayers) const {
    using namespace muonscarekit_impl;
    std::seed_seq seq{hash_, seed_, std::size_t(run), std::size_t(lumi),
                      std::size_t(event)};
    std::mt19937 rng(seq);
    std::uniform_real_distribution<double> unif(0., 1.);

    unsigned int size = pt.size();
    Vec_f res(size);

    for (unsigned int i = 0; i < size; ++i) {
      double M = h_M_SIG->GetBinContent(h_M_SIG->FindBin(eta[i], phi[i]));
      double A = h_A_SIG->GetBinContent(h_A_SIG->FindBin(eta[i], phi[i]));
      double pt_scale = 1.0 / (M / pt[i] + charge[i] * A);

      Int_t etabin = h_cb->GetXaxis()->FindBin(fabs((double)eta[i]));
      Int_t nlbin = h_cb->GetYaxis()->FindBin((double)nTrackerLayers[i]);

      double mean_cb = h_cb->GetBinContent(etabin, nlbin, 1);
      double sig_cb = h_cb->GetBinContent(etabin, nlbin, 2);
      double n_cb = h_cb->GetBinContent(etabin, nlbin, 3);
      double alpha_cb = h_cb->GetBinContent(etabin, nlbin, 4);

      double a_poly = h_poly->GetBinContent(etabin, nlbin, 1);
      double b_poly = h_poly->GetBinContent(etabin, nlbin, 2);
      double c_poly = h_poly->GetBinContent(etabin, nlbin, 3);
      double sigma_poly =
          a_poly + b_poly * pt_scale + c_poly * pt_scale * pt_scale;
      if (sigma_poly < 0.0)
        sigma_poly = 0.0;

      Int_t absetabin = h_k_data->GetXaxis()->FindBin(fabs((double)eta[i]));
      double k_data_v = h_k_data->GetBinContent(absetabin, 3);
      double k_sig_v = h_k_sig->GetBinContent(absetabin, 3);
      double k_mc = (k_sig_v < k_data_v)
                        ? sqrt(k_data_v * k_data_v - k_sig_v * k_sig_v)
                        : 0.0;

      TH1D *h_k_unc_src = k_unc_syst_ ? h_k_unc_syst : h_k_unc;
      if (k_unc_shift_ != 0.0 && k_mc > 0.0 && h_k_unc_src != nullptr) {
        const int kbin = h_k_unc_src->GetXaxis()->FindBin(fabs((double)eta[i]));
        k_mc =
            std::max(0.0, k_mc + k_unc_shift_ * h_k_unc_src->GetBinError(kbin));
      }

      if (k_mc == 0.0 || sigma_poly == 0.0 || n_cb <= 1.0 + 1e-6 ||
          sig_cb <= 0.0 || alpha_cb <= 0.0) {
        res[i] = static_cast<float>(pt_scale);
        continue;
      }

      MuonScarekitCB cb(mean_cb, sig_cb, alpha_cb, n_cb);
      double rndm_cb = cb.invcdf(unif(rng));

      res[i] =
          static_cast<float>(pt_scale * (1.0 + k_mc * sigma_poly * rndm_cb));
    }
    return res;
  }

private:
  const std::size_t hash_;
  std::size_t seed_;
  double k_unc_shift_ = 0.0;
  bool k_unc_syst_ = false;
};

} // namespace wrem
#endif
