"""PROTOTYPE — joint-conditional (RTS/FFBS) observational Shapley.

Standalone proof-of-concept, NOT wired into `analytic_estimates._observational`
and NOT production code. It demonstrates the joint-smoother approach that fixes
the coupled-feature scenarios (e.g. base_chain), which the per-frame self-chain
bridge in `analytic_estimates` cannot (that one is exact only for decoupled
observed subgraphs: base_linear / base_multiplicative / base_ratio).

Method
------
Proposal: sample the free past-nodes from the EKF-linearised Gaussian conditional
    p(free | pinned = base). Each observed edge's coefficient in the precision is
    the Jacobian of its rule evaluated at the base trajectory, per frame
    (`edge_coeffs_at_base`); for a linear edge this is the exact constant.
Correction: importance-weight each of the B samples by  p_true / q_linearised,
    where p_true uses the ACTUAL (nonlinear) rule residuals (`true_log_prior`).
    Self-normalised over the samples. For a linear observed subgraph p_true == q,
    so the weight is constant and v(S) is a plain average (no weighting).
Common random numbers (`base_normals`, shared across coalitions in a permutation)
    make a dummy feature's marginal *identically* zero rather than zero-in-mean.

Verified on a small A->B->C->Hazard chain (see __main__):
    linear edge  (0.8*A)        -> ESS 16/16,  D signed/|.| = 0.0000 (exact)
    tanh  edge   (0.8*tanh(A))  -> ESS ~10/16, D signed/|.| = 0.0000, weights active

Known limitations (production work, not yet done)
-------------------------------------------------
* Speed: per-coalition DENSE Cholesky in a Python loop. Real lf needs the banded
  precision + incremental (rank-1) updates along the nested coalitions.
* Linearisation: EKF (Jacobian via Expr.partial). For edges too sharp for a
  single Gaussian, swap in a UKF sigma-point slope in `edge_coeffs_at_base`, or
  a 1-D grid; ESS is the diagnostic for when that is needed.
* Not integrated: does not read/write the shap_samples.h5 pipeline or share the
  ScenarioCfg/hazard-calibration plumbing used by the real estimators.
"""
import numpy as np
from scipy.linalg import solve_triangular, cholesky

from PhagoPred.survival_v2.data.graph_synthetic.rules import (
    Rule, ReLU, Var, Apply, AutoCorrelationRule)
from PhagoPred.survival_v2.data.graph_synthetic.graph import CausalGraph, Feature
from PhagoPred.survival_v2.data.graph_synthetic import noise_funcs

S = 0.5
RIDGE = 1e-8


def lp(x, sc):
    return -0.5 * (x / sc) ** 2 - np.log(sc * np.sqrt(2 * np.pi))


def chain(lag=5, nonlinear=False):
    feats = [Feature(n, pre_noise=noise_funcs.GaussianNoise(S),
                     post_noise=noise_funcs.NoNoise())
             for n in ('A', 'B', 'C', 'D')]
    feats.append(Feature('Hazard', pre_noise=noise_funcs.NoNoise()))
    a_edge = (0.8 * Apply(np.tanh, Var('A', lag))) if nonlinear else (0.8 * Var('A', lag))
    rules = [Rule('B', a_edge),
             Rule('C', 0.8 * Var('B', lag)),
             Rule('Hazard', ReLU(Var('C', lag)))]
    rules += [AutoCorrelationRule(k, 1 - 1e-5) for k in ('A', 'B', 'C', 'D')]
    return CausalGraph(feats, rules, 500)


def edge_coeffs_at_base(graph, base_sig, lf):
    """Per-target list of (parent, lag, coef_array[lf]) — EKF Jacobian at base."""
    obs = [f.name for f in graph.features if f.name != 'Hazard']
    edges = {f: [] for f in obs}
    for rule in graph.rules:
        if rule.target == 'Hazard':
            continue
        for n, lags in rule.inputs.items():
            for lag in lags:
                coef = np.zeros(lf)
                for t in range(lf):
                    if t - lag < 0:
                        continue
                    ctx = {(nm, lg): base_sig[nm][t - lg]
                           for nm, lgs in rule.inputs.items() for lg in lgs
                           if t - lg >= 0}
                    coef[t] = rule.expr.partial(n, lag, ctx)
                edges[rule.target].append((n, lag, coef))
    return obs, edges


def build_precision(obs, edges, lf, idx):
    n = len(idx)
    Q = np.zeros((n, n))
    iv = 1.0 / S ** 2
    for f in obs:
        for t in range(lf):
            terms = [(idx[(f, t)], 1.0)]
            for (p, lag, coef) in edges[f]:
                if t - lag >= 0:
                    terms.append((idx[(p, t - lag)], -coef[t]))
            for i, ci in terms:
                for j, cj in terms:
                    Q[i, j] += iv * ci * cj
    return Q


def true_log_prior(graph, past, obs, scales, lf, B):
    """Sum of true (nonlinear) residual log-densities over past observed nodes."""
    logp = np.zeros(B)
    for f in obs:
        sc = scales[f]
        for t in range(lf):
            drift = np.zeros(B)
            for rule in graph.rules:
                if rule.target != f or t < rule.max_lag:
                    continue
                ctx = {(nm, lg): past[nm][t - lg]
                       for nm, lgs in rule.inputs.items() for lg in lgs}
                drift = drift + rule.expr(ctx)
            logp += lp(past[f][t] - drift, sc)
    return logp


def propagate_forward(graph, past, base_noise, lf, horizon, B):
    seq = lf + horizon
    sig = {f.name: np.zeros((seq, B)) for f in graph.features}
    for f in graph.features:
        if f.name in past:
            sig[f.name][:lf] = past[f.name]
        sig[f.name][lf:] = base_noise[f.name][lf:seq, None]
    for t in range(lf, seq):
        for rule in graph.rules:
            rule.apply_step(sig, t)
    return sig


def hazard_target(sig, lf, horizon):
    hz = np.maximum(sig['Hazard'][lf:lf + horizon], 0.0)
    h = 1.0 - np.exp(-0.05 * hz)
    return 1.0 - np.prod(1.0 - h, axis=0)


def joint_observational(graph, base_sig, base_noise, lf, horizon, nperm, B, seed):
    rng = np.random.default_rng(seed)
    obs, edges = edge_coeffs_at_base(graph, base_sig, lf)
    scales = {f.name: float(getattr(f.pre_noise, 'sigma', 1.0))
              for f in graph.features}
    idx = {(f, t): fi * lf + t for fi, f in enumerate(obs) for t in range(lf)}
    inv_idx = {v: k for k, v in idx.items()}
    n = len(idx)
    Q = build_precision(obs, edges, lf, idx) + np.eye(n) * RIDGE
    basevec = np.array([base_sig[f][t] for (f, t) in idx])
    acc = np.zeros(n)
    ess_log = []

    for _ in range(nperm):
        perm = rng.permutation(n)
        inv = np.argsort(perm)
        present = inv[:, None] < np.arange(n + 1)[None, :]
        base_normals = rng.normal(size=(n, B))          # common random numbers
        v = np.zeros(n + 1)
        for k in range(n + 1):
            pin = present[:, k]
            F = np.where(~pin)[0]
            P = np.where(pin)[0]
            past = {f: np.repeat(base_sig[f][:lf, None], B, axis=1) for f in obs}
            if len(F) > 0:
                Qff = Q[np.ix_(F, F)]
                Qfp = Q[np.ix_(F, P)]
                Lc = cholesky(Qff, lower=True)
                mean = (np.linalg.solve(Qff, -(Qfp @ basevec[P]))
                        if len(P) else np.zeros(len(F)))
                Z = mean[:, None] + solve_triangular(Lc.T, base_normals[F])
                for row, i in enumerate(F):
                    f, t = inv_idx[i]
                    past[f][t] = Z[row]
                # log q (linearised conditional density) per sample
                d = Z - mean[:, None]
                logdet = 2 * np.sum(np.log(np.diag(Lc)))
                logq = 0.5 * logdet - 0.5 * np.einsum('ib,ib->b', d, Qff @ d) \
                    - 0.5 * len(F) * np.log(2 * np.pi)
                logp = true_log_prior(graph, past, obs, scales, lf, B)
                logw = logp - logq
            else:
                logw = np.zeros(B)
            sig = propagate_forward(graph, past, base_noise, lf, horizon, B)
            tgt = hazard_target(sig, lf, horizon)
            ww = np.exp(logw - logw.max())
            v[k] = (ww * tgt).sum() / ww.sum()
            if 0 < len(F) < n:
                ess_log.append((ww.sum() ** 2) / (ww ** 2).sum())
        marg = np.diff(v)
        contrib = np.zeros(n)
        contrib[perm] = marg
        acc += contrib
    return (acc / nperm).reshape(len(obs), lf), obs, np.mean(ess_log)


def run(nonlinear, seed=1):
    np.random.seed(seed)
    g = chain(lag=4, nonlinear=nonlinear)
    lf, horizon, B, nperm = 20, 8, 16, 40
    from PhagoPred.survival_v2.data.graph_synthetic.analytic_estimates import _apply_rules
    bn = {f.name: f.generate_signal(lf + horizon) for f in g.features}
    base = _apply_rules(g, bn, lf + horizon)
    m, obs, ess = joint_observational(g, base, bn, lf, horizon, nperm, B, seed)
    tag = 'NONLINEAR (0.8*tanh(A))' if nonlinear else 'LINEAR (0.8*A)'
    print('=== %s edge,  mean ESS=%.2f / %d ===' % (tag, ess, B))
    for j, f in enumerate(obs):
        print('   %s  signed=%+.4f   |.|=%.4f' % (f, m[j].sum(), np.abs(m[j]).sum()))
    print()


if __name__ == '__main__':
    run(nonlinear=False)
    run(nonlinear=True)
