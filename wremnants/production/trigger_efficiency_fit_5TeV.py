"""Per-muon trigger efficiency from event-level pair counts.

Inverts the lowPU combination formula (lowpu_efficiencies.hpp::lepSF_HLT_q)

    P(event fires | muons in bins a, b) = 1 - (1 - eps_a)(1 - eps_b)

by maximum likelihood over the per-muon efficiencies eps, given the observed
(bin_a, bin_b, pass) counts. That formula is what lowPU uses to APPLY a tagged
tag-and-probe map; used backwards it EXTRACTS the map from the event decision
alone, which is the only thing observable without muon trigger objects.

Binomial log-likelihood, with n_pass(a,b) and n_fail(a,b) the weighted counts:

    -lnL = -sum_ab [ n_pass ln P_ab + n_fail ln(1 - P_ab) ]

and, using 1 - P_ab = (1 - eps_a)(1 - eps_b), the gradient collapses to

    d(-lnL)/d eps_a = sum_b [ -n_pass(a,b) (1 - eps_b) / P_ab
                              + n_fail(a,b) / (1 - eps_a) ]

which is exact and cheap -- no numerical differentiation.

Per-bin uncertainties come from the inverse Hessian diagonal, and the FULL
covariance is returned, because here it cannot be thrown away.

WHY THE NUISANCES CANNOT BE INDEPENDENT PER BIN, unlike lowPU's. lowPU's map
comes from a tagged tag-and-probe measurement, where each muon is individually
labelled as having fired, so neighbouring bins are nearly uncorrelated and one
nuisance per bin is a fair representation. Here only the EVENT decision is
observable, and a single event constrains the PRODUCT (1-eps_a)(1-eps_b). The
overall scale of (1 - eps) is therefore weakly determined while relative
variations between bins are well determined -- a near-degenerate shared mode.
Measured on toys: off-diagonal correlations reach |rho| ~ 0.7, and pulls computed
from the marginal diagonal errors come out with width 0.46 (flat truth) to 0.84
(structured truth) instead of 1, i.e. the per-bin errors are dominated by that
shared mode and treating 240 of them as independent would badly mis-model the
uncertainty.

The right representation is the eigendecomposition of the covariance, which is
what the 13 TeV SMOOTH scale factors already use for their stat nuisances (see
the "number of eigen variations" comment in muon_efficiencies_binned.hpp).
eigen_variations() below returns those decorrelated shifts, ordered by decreasing
eigenvalue, so a small number of them captures the uncertainty faithfully.
"""

import numpy as np


def _pack(npass, nfail):
    """Symmetrise the pair counts: only a <= b is independent."""
    n = npass.shape[0]
    iu = np.triu_indices(n)
    # a != b appears twice in the histogram (bin1,bin2) and (bin2,bin1)
    P = npass + npass.T - np.diag(np.diag(npass))
    F = nfail + nfail.T - np.diag(np.diag(nfail))
    return iu, P[iu], F[iu]


def nll_and_grad(eps, ia, ib, npass, nfail):
    ea, eb = eps[ia], eps[ib]
    one_m = (1.0 - ea) * (1.0 - eb)
    P = 1.0 - one_m
    P = np.clip(P, 1e-12, 1.0 - 1e-12)
    one_m = np.clip(one_m, 1e-12, 1.0 - 1e-12)
    nll = -(npass * np.log(P) + nfail * np.log(one_m)).sum()

    g = np.zeros_like(eps)
    # d/d eps_a of -n_pass ln P = -n_pass (1 - eps_b) / P
    ta = -npass * (1.0 - eb) / P
    tb = -npass * (1.0 - ea) / P
    # d/d eps_a of -n_fail ln[(1-eps_a)(1-eps_b)] = +n_fail / (1 - eps_a)
    ua = nfail / np.maximum(1.0 - ea, 1e-12)
    ub = nfail / np.maximum(1.0 - eb, 1e-12)
    np.add.at(g, ia, ta + ua)
    np.add.at(g, ib, tb + ub)
    # a == b entries were counted once but contribute to both slots; the two
    # np.add.at calls above already handle that correctly
    return nll, g


def hessian(eps, ia, ib, npass, nfail, n):
    """Exact Hessian of the NLL. Small (n x n), so build it densely."""
    ea, eb = eps[ia], eps[ib]
    P = np.clip(1.0 - (1.0 - ea) * (1.0 - eb), 1e-12, 1.0 - 1e-12)
    H = np.zeros((n, n))
    # d2/deps_a2 : n_pass (1-eps_b)^2 / P^2 + n_fail / (1-eps_a)^2
    daa = npass * (1.0 - eb) ** 2 / P**2 + nfail / np.maximum((1.0 - ea) ** 2, 1e-12)
    dbb = npass * (1.0 - ea) ** 2 / P**2 + nfail / np.maximum((1.0 - eb) ** 2, 1e-12)
    np.add.at(H, (ia, ia), daa)
    np.add.at(H, (ib, ib), dbb)
    # cross term d2/deps_a deps_b = n_pass[(1-eps_a)(1-eps_b)/P^2 - 1/P]
    dab = npass * ((1.0 - ea) * (1.0 - eb) / P**2 - 1.0 / P)
    np.add.at(H, (ia, ib), dab)
    np.add.at(H, (ib, ia), dab)
    return H


def fit_per_muon_efficiency(npass, nfail, eps0=None, verbose=True):
    """Return (eps, sigma, info) for the per-muon efficiencies.

    npass, nfail: (n_bins, n_bins) weighted counts from the pair histogram.
    Bins with no entries at all are left at eps = 0 with sigma = 0 and flagged in
    info["empty"], so downstream code can prune them exactly as it prunes empty
    cells of the counted maps.
    """
    from scipy.optimize import minimize

    n = npass.shape[0]
    iu, P, F = _pack(npass, nfail)
    ia, ib = iu
    keep = (P + F) > 0
    ia, ib, P, F = ia[keep], ib[keep], P[keep], F[keep]

    occupancy = np.zeros(n)
    np.add.at(occupancy, ia, P + F)
    np.add.at(occupancy, ib, P + F)
    empty = occupancy <= 0

    # start from the naive single-bin solution: 1 - eps = sqrt(fail fraction)
    if eps0 is None:
        tot = P.sum() + F.sum()
        frac_fail = F.sum() / max(tot, 1.0)
        eps0 = np.full(n, 1.0 - np.sqrt(max(frac_fail, 1e-6)))
    x0 = np.clip(eps0, 1e-3, 1.0 - 1e-3)

    res = minimize(
        lambda e: nll_and_grad(e, ia, ib, P, F),
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(1e-4, 1.0 - 1e-4)] * n,
        options=dict(maxiter=20000, ftol=1e-14, gtol=1e-10),
    )
    eps = res.x
    # Bins pushed to a bound are NOT determined: the event observable only
    # constrains the DOUBLE-failure rate (1-eps_a)(1-eps_b), so a bin with no
    # double failures has no curvature and eps runs to 1. Including such bins in
    # the covariance makes it singular and produces meaningless correlations
    # (observed: |rho| ~ 1e14). They are reported and excluded instead.
    at_bound = (eps > 1.0 - 1e-3) | (eps < 1e-3)
    H = hessian(eps, ia, ib, P, F, n)
    sigma = np.zeros(n)
    free = ~(empty | at_bound)
    if free.sum():
        Hf = H[np.ix_(free, free)]
        # ridge for numerical safety; tiny compared with the diagonal
        Hf = Hf + np.eye(Hf.shape[0]) * 1e-9 * np.trace(Hf) / Hf.shape[0]
        try:
            C = np.linalg.inv(Hf)
            d = np.clip(np.diag(C), 0.0, None)
            sigma[free] = np.sqrt(d)
            # largest off-diagonal correlation, to judge the lowPU approximation
            sd = np.sqrt(np.maximum(np.diag(C), 1e-30))
            corr = C / np.outer(sd, sd)
            np.fill_diagonal(corr, 0.0)
            max_corr = float(np.abs(corr).max())
        except np.linalg.LinAlgError:
            max_corr = float("nan")
    else:
        max_corr = float("nan")
    eps[empty] = 0.0
    cov = np.zeros((n, n))
    if free.sum():
        cov[np.ix_(free, free)] = C
    n_fail_tot = float(F.sum())
    identifiable = bool(res.success and float(np.linalg.norm(res.jac[free])) < 1e-2
                        if free.sum() else False)
    info = dict(success=bool(res.success), nll=float(res.fun), nfev=int(res.nfev),
                empty=empty, at_bound=at_bound, identifiable=identifiable,
                n_double_fail=n_fail_tot, max_corr=max_corr,
                occupancy=occupancy, cov=cov,
                grad_norm=float(np.linalg.norm(res.jac[free])) if free.sum() else 0.0)
    if verbose:
        print(f"    fit: success={info['success']} nll={info['nll']:.1f} "
              f"|grad|={info['grad_norm']:.2e} nfev={info['nfev']}")
        print(f"    {int(at_bound.sum())}/{n} bins AT A BOUND (undetermined: no"
              f" double failures); {n_fail_tot:.0f} double failures in total")
        print(f"    identifiable = {identifiable}")
        print(f"    {int(empty.sum())}/{n} bins empty; "
              f"eps range [{eps[free].min():.4f}, {eps[free].max():.4f}]; "
              f"median sigma {np.median(sigma[free]):.4f}; "
              f"max |off-diagonal correlation| {max_corr:.3f}"
              if free.sum() else "    no determined bins")
    return eps, sigma, info


def eigen_variations(cov, n_keep=None, tol=1e-12):
    """Decorrelated +1 sigma shifts from a covariance matrix.

    Returns (shifts, eigenvalues) with shifts[k] the vector to ADD to eps for the
    k-th nuisance, ordered by decreasing eigenvalue. Summing shifts[k] in
    quadrature reproduces the diagonal errors exactly, so nothing is lost, while
    the nuisances are independent by construction -- which the per-bin ones are
    not (see the module docstring).

    n_keep truncates to the leading modes. The dropped variance is reported by the
    caller from the returned eigenvalues, so truncation is never silent.
    """
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    keep = w > tol
    w, V = w[keep], V[:, keep]
    if n_keep is not None:
        w, V = w[:n_keep], V[:, :n_keep]
    # column k scaled by sqrt(eigenvalue) is the +1 sigma shift along mode k
    shifts = (V * np.sqrt(w)).T
    return shifts, w
