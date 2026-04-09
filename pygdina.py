"""
gdina.py — Pure-Python GDINA estimation (identity link, MMLE/EM)

Implements the Generalized Deterministic Inputs Noisy And gate model
(de la Torre, 2011) via marginal maximum likelihood with EM. Covers all
CDMs expressible under the GDINA framework (DINA, DINO, A-CDM, etc.).

Design alignment
----------------
- Attribute patterns : binary counting order, matching R GDINA / CDM conventions
- Initialization     : start 0 replicates R CDM seed=0 (ANOVA delta, deterministic);
                       start 1+ use random DINA / DINO / A-CDM profiles
- Multi-start        : compare obs-LL across starts, run full EM on best
                       → O(nstarts × 1 E-step) + O(maxitr), not O(nstarts × EM)
                       Mirrors R GDINA package (Ma & de la Torre, 2020)
- Convergence        : |ΔLL| < conv_crit (ΔLL between successive E-steps)
- Best-solution save : roll back to the iteration with lowest deviance within the
                       single EM run, guarding against transient upswings near
                       convergence. Mirrors R CDM save.devmin=TRUE (Robitzsch et al.)

References
----------
de la Torre, J. (2011). The generalized DINA model framework.
    Psychometrika, 76(2), 179–199.
Ma, W., & de la Torre, J. (2020). GDINA: An R package for cognitive diagnosis
    modeling. Journal of Statistical Software, 93(14), 1–26.
Robitzsch, A., Kiefer, T., & George, A. C. (2023). CDM: Cognitive Diagnosis
    Modeling. R package version 8.x.
"""

from __future__ import annotations

import numpy as np
from itertools import combinations
from scipy.special import logsumexp


# ─────────────────────────────────────────────────────────────
# Module-level helper
# ─────────────────────────────────────────────────────────────

def _skill_profiles(K: int) -> np.ndarray:
    """
    Return the (2^K, K) matrix of all binary attribute profiles.

    Profiles are ordered by binary counting (bit 0 = attribute 1, bit 1 =
    attribute 2, …), consistent with R GDINA / CDM:

        K=2 → [[0,0], [1,0], [0,1], [1,1]]

    This ordering is critical: any deviation causes parloc to mis-map
    latent classes to reduced groups, silently corrupting all posteriors.
    """
    L = 2 ** K
    return np.array([[(l >> k) & 1 for k in range(K)] for l in range(L)], dtype=int)


# ─────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────

class GDINA:
    """
    GDINA model — identity link, MMLE/EM estimation.

    Parameters
    ----------
    att_dist : {'saturated', 'independent'}, default 'saturated'
        Attribute distribution structure.

        'saturated'   — unconstrained mixing proportions over all 2^K
                        latent classes (equivalent to R GDINA att.dist =
                        "saturated" and R CDM att_dist = "unstructured").
        'independent' — K independent Bernoulli marginals; class probabilities
                        are their products. Reduces free prior parameters
                        from 2^K − 1 to K.

    max_iter : int, default 2000
        Maximum EM iterations.

    conv_crit : float, default 1e-5
        Convergence threshold on |ΔLL| between successive iterations.
        Tighter values increase precision at the cost of more iterations.

    n_starts : int, default 3
        Number of starting values. Start 0 is deterministic (R CDM
        compatible); starts 1+ are random (DINA / DINO / A-CDM type).
        Only the start with the highest initial observed log-likelihood
        proceeds to full EM — O(n_starts × 1 E-step) overhead, not
        O(n_starts × max_iter).

    lower_p, upper_p : float, default 1e-4 / 1−1e-4
        Box constraints on item success probabilities, preventing boundary
        degeneracy during M-step.

    random_seed : int, default 123456
        Seed passed to numpy before generating random starting values.

    verbose : bool, default True
        Print per-iteration progress (deviance, |ΔLL|).

    Fitted attributes (available after .fit)
    -----------------------------------------
    item_parm : ndarray (J, Lj_max)
        Item success probabilities per reduced latent group.
        item_parm[j, k] = P(X_j=1 | reduced group k+1 of item j).
        Columns beyond item j's 2^Kj active groups are zero-padded.

    log_prior : ndarray (L,)
        Log mixing proportions over all L = 2^K attribute profiles.

    att_pattern : ndarray (L, K)
        All attribute profiles (rows), attributes (columns). Output of
        _skill_profiles(K); stored for downstream posterior computation.

    parloc : ndarray (J, L)  — 1-indexed
        parloc[j, l] is the column index (1-based) into item_parm[j]
        for a person belonging to global latent class l on item j.

    log_post : ndarray (N, L)
        Log posterior P(α_l | X_n) for every person n and class l.

    deviance : float
        −2 × log-likelihood at the saved (lowest-deviance) solution.
    """

    def __init__(
        self,
        att_dist: str = "saturated",
        max_iter: int = 2000,
        conv_crit: float = 1e-5,
        n_starts: int = 3,
        lower_p: float = 1e-4,
        upper_p: float = 1 - 1e-4,
        random_seed: int = 123456,
        verbose: bool = True,
    ) -> None:
        if att_dist not in ("saturated", "independent"):
            raise ValueError("att_dist must be 'saturated' or 'independent'")
        self.att_dist    = att_dist
        self.max_iter    = max_iter
        self.conv_crit   = conv_crit
        self.n_starts    = n_starts
        self.lower_p     = lower_p
        self.upper_p     = upper_p
        self.random_seed = random_seed
        self.verbose     = verbose

        # set after .fit
        self.item_parm  : np.ndarray | None = None
        self.log_prior  : np.ndarray | None = None
        self.att_pattern: np.ndarray | None = None
        self.parloc     : np.ndarray | None = None
        self.log_post   : np.ndarray | None = None
        self.deviance   : float | None      = None

    # ── private: structural helpers ──────────────────────────

    def _build_parloc(self, Q: np.ndarray) -> np.ndarray:
        """
        Build parloc[j, l]: for item j and global class l, return the
        1-indexed position within item j's reduced group table.

        Each item j requires Kj skills (non-zero columns of Q[j]).  The
        full 2^K global classes collapse onto 2^Kj reduced groups based
        on the Kj required skills.  parloc encodes this mapping so that
        mP[j, l] = item_parm[j, parloc[j,l] − 1] everywhere.
        """
        J, L = Q.shape[0], len(self.att_pattern)
        parloc = np.zeros((J, L), dtype=int)
        for j in range(J):
            req   = np.where(Q[j] > 0)[0]             # required skill indices
            Kj    = len(req)
            reduc = _skill_profiles(Kj)                # (2^Kj, Kj) reduced patterns
            proj  = self.att_pattern[:, req]           # (L, Kj) global → reduced projection
            # match[l, r] is True when global class l maps to reduced group r
            match    = (proj[:, None, :] == reduc[None, :, :]).all(axis=2)
            parloc[j] = match.argmax(axis=1) + 1       # 1-indexed
        return parloc

    @staticmethod
    def _design_matrix(Kj: int) -> np.ndarray:
        """
        ANOVA design matrix M for Kj required skills (identity link).

        Shape: (2^Kj, 2^Kj).  Columns: intercept, Kj main effects,
        C(Kj,2) two-way interactions, …, one Kj-way interaction — the
        complete ANOVA parameterisation used by R CDM / GDINA.

        Satisfies: P_j = M @ delta_j, where delta_j are the GDINA
        delta parameters (intercept + interaction effects).
        """
        ap = _skill_profiles(Kj)
        col_defs = [()]  # intercept first
        for r in range(1, Kj + 1):
            for combo in combinations(range(Kj), r):
                col_defs.append(combo)
        M = np.zeros((len(ap), len(col_defs)))
        for c, cols in enumerate(col_defs):
            M[:, c] = 1.0 if not cols else np.prod(ap[:, list(cols)], axis=1)
        return M

    @staticmethod
    def _compress(dat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reduce N×J response matrix to unique patterns + frequencies.

        Returns
        -------
        Y_uniq     : (n_uniq, J) unique response vectors
        freq       : (n_uniq,)  count of each unique pattern in dat
        raw2unique : (N,)       index mapping each row of dat to Y_uniq
        """
        patterns = [tuple(row) for row in dat.astype(int)]
        seen: dict = {}
        order: list = []
        for p in patterns:
            if p not in seen:
                seen[p] = len(order)
                order.append(p)
        Y_uniq     = np.array(order, dtype=float)
        raw2unique = np.array([seen[p] for p in patterns], dtype=int)
        freq       = np.bincount(raw2unique, minlength=len(order)).astype(float)
        return Y_uniq, freq, raw2unique

    # ── private: EM components ───────────────────────────────

    def _mP(self, item_parm: np.ndarray) -> np.ndarray:
        """
        Expand item_parm (J, Lj_max) to mP (J, L) using parloc.

        mP[j, l] = P(X_j=1 | global class l), derived by looking up
        item_parm[j, parloc[j,l]−1] for every (j, l) pair.
        """
        J, L = self.parloc.shape
        mP = np.empty((J, L))
        for j in range(J):
            mP[j] = item_parm[j, self.parloc[j] - 1]
        return mP

    def _estep(
        self,
        item_parm: np.ndarray,
        log_prior: np.ndarray,
        Y_uniq: np.ndarray,
        freq: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        E-step: compute posterior and sufficient statistics.

        For each unique response pattern p and latent class l:

            log P(X=p, α_l) = Σ_j [x_j log P_jl + (1−x_j) log(1−P_jl)]
                               + log π_l

            log P(α_l | X=p) = log P(X=p, α_l) − log P(X=p)

        Sufficient statistics collected for the M-step:

            Ng[j, k] = Σ_p freq_p * P(α_k* | X=p)   (expected N in reduced group k)
            Rg[j, k] = Σ_p freq_p * x_pj * P(α_k* | X=p)   (expected correct count)

        Returns
        -------
        LL       : observed log-likelihood Σ_p freq_p log P(X=p)
        log_post : (n_uniq, L) log posterior per unique pattern
        Ng, Rg   : (J, Lj_max) sufficient statistics
        """
        eps = 1e-16
        J   = self.parloc.shape[0]
        Lj_max = int(self.parloc.max())

        mP      = np.clip(self._mP(item_parm), eps, 1 - eps)
        log_lik = Y_uniq @ np.log(mP) + (1 - Y_uniq) @ np.log(1 - mP)  # (n_uniq, L)
        log_jnt = log_lik + log_prior[None, :]                           # (n_uniq, L)

        log_marg = logsumexp(log_jnt, axis=1, keepdims=True)             # (n_uniq, 1)
        log_post = log_jnt - log_marg                                     # (n_uniq, L)
        LL       = float((freq @ log_marg.squeeze()))

        wpost   = np.exp(log_post) * freq[:, None]  # (n_uniq, L) weighted posterior
        expN_lc = wpost.sum(axis=0)                 # (L,) expected N per global class
        expR    = Y_uniq.T @ wpost                  # (J, L) expected correct responses

        Ng = np.zeros((J, Lj_max))
        Rg = np.zeros((J, Lj_max))
        for j in range(J):
            for k in range(int(self.parloc[j].max())):
                mask    = self.parloc[j] == (k + 1)
                Ng[j, k] = expN_lc[mask].sum()
                Rg[j, k] = expR[j, mask].sum()

        return LL, log_post, Ng, Rg

    def _mstep_prior(self, wpost_colsum: np.ndarray, N: float) -> np.ndarray:
        """
        Compute updated log_prior from column sums of weighted posterior.

        Parameters
        ----------
        wpost_colsum : (L,) — Σ_p freq_p P(α_l | X_p) for each class l
        N            : total sample size (= freq.sum())
        """
        eps = 1e-16
        if self.att_dist == "saturated":
            return np.log(np.clip(wpost_colsum / N, eps, None))
        # independent
        marg = (self.att_pattern.T @ wpost_colsum) / N    # (K,) marginal mastery
        marg = np.clip(marg, eps, 1 - eps)
        return (self.att_pattern @ np.log(marg)
                + (1 - self.att_pattern) @ np.log(1 - marg))

    def _mstep_items(
        self, Ng: np.ndarray, Rg: np.ndarray, Lj: np.ndarray
    ) -> np.ndarray:
        """
        Item M-step (identity link, closed form).

        Under the identity link the GDINA log-likelihood factorises over
        items and reduced groups, giving closed-form updates:

            P_jk = Rg[j,k] / Ng[j,k]

        This is equivalent to independent binomial MLEs for each
        (item j, reduced group k) cell.  No numerical optimiser needed.
        """
        new_ip = np.zeros_like(Ng)
        for j in range(len(Lj)):
            kj = Lj[j]
            denom = np.maximum(Ng[j, :kj], 1e-16)
            new_ip[j, :kj] = np.clip(Rg[j, :kj] / denom, self.lower_p, self.upper_p)
        return new_ip

    # ── private: initialisation ──────────────────────────────

    def _init_item_parm(self, Q: np.ndarray) -> list[np.ndarray]:
        """
        Generate n_starts initial item_parm matrices.

        Start 0 — deterministic, replicates R CDM seed=0:
            delta[0]  = 0.2  (intercept → P(all-0) = 0.2)
            delta[1:] = 0.6 / (n_delta − 1)  (effects equally weighted)
            → P(all-1) = 0.8, intermediate groups linearly interpolated

        Starts 1+ — random, cycling through DINA / DINO / A-CDM archetypes
            with g ~ U(0.05, 0.25) and p ~ U(0.75, 0.95).
        """
        J      = Q.shape[0]
        Lj_max = int(self.parloc.max())
        starts = []

        for s in range(self.n_starts):
            ip = np.zeros((J, Lj_max))

            if s == 0:
                # deterministic R CDM–compatible initialisation
                for j in range(J):
                    Kj  = int(Q[j].sum())
                    M   = self._design_matrix(Kj)
                    n   = M.shape[1]
                    d   = np.full(n, 0.6 / max(n - 1, 1))
                    d[0] = 0.2
                    ip[j, : 2 ** Kj] = np.clip(M @ d, self.lower_p, self.upper_p)

            else:
                archetype = ("dina", "dino", "acdm")[(s - 1) % 3]
                for j in range(J):
                    req  = np.where(Q[j] > 0)[0]
                    Kj   = len(req)
                    patt = _skill_profiles(Kj)
                    Lj_j = len(patt)
                    g    = np.random.uniform(0.05, 0.25)
                    p    = np.random.uniform(0.75, 0.95)
                    if archetype == "dina":
                        pj = np.where(patt.all(axis=1), p, g)
                    elif archetype == "dino":
                        pj = np.where(patt.any(axis=1), p, g)
                    else:  # acdm
                        pj = g + (p - g) * patt.sum(axis=1) / max(Kj, 1)
                    ip[j, :Lj_j] = pj

            starts.append(ip)
        return starts

    # ── public API ───────────────────────────────────────────

    def fit(self, dat: np.ndarray, Q: np.ndarray) -> "GDINA":
        """
        Fit the GDINA model via MMLE/EM.

        Parameters
        ----------
        dat : array-like (N, J)
            Binary item response matrix (0/1). Missing values not supported.
        Q : array-like (J, K)
            Binary Q-matrix: Q[j, k] = 1 if item j requires attribute k.

        Returns
        -------
        self
        """
        dat = np.asarray(dat, dtype=float)
        Q   = np.asarray(Q,   dtype=float)
        J, K = Q.shape

        # ── structural setup ──
        self.att_pattern = _skill_profiles(K)
        L                = len(self.att_pattern)
        self.parloc      = self._build_parloc(Q)
        Lj               = np.array([int(self.parloc[j].max()) for j in range(J)])

        # ── compress data ──
        Y_uniq, freq, raw2unique = self._compress(dat)
        N = freq.sum()

        # ── initial log-prior ──
        if self.att_dist == "saturated":
            log_prior = np.full(L, -np.log(L))          # uniform
        else:
            log_prior = np.full(L, -K * np.log(2.0))    # independent, all p=0.5

        # ── select best starting value by initial obs-LL ──
        np.random.seed(self.random_seed)
        starts = self._init_item_parm(Q)

        if self.n_starts > 1:
            init_ll = [
                float(np.sum(freq * logsumexp(
                    Y_uniq @ np.log(np.clip(self._mP(ip), 1e-16, 1 - 1e-16))
                    + (1 - Y_uniq) @ np.log(np.clip(1 - self._mP(ip), 1e-16, 1 - 1e-16))
                    + log_prior[None, :], axis=1)))
                for ip in starts
            ]
            best = int(np.argmax(init_ll))
            if self.verbose:
                names = ["R-default"] + ["DINA", "DINO", "A-CDM"] * 10
                for i, ll in enumerate(init_ll):
                    print(f"  Start {i + 1} ({names[i]}): deviance = {-2 * ll:.2f}")
                print(f"  → selected Start {best + 1}\n")
        else:
            best = 0

        item_parm = starts[best].copy()

        # ── EM loop with best-deviance save ──
        # At each iteration we store the state with the lowest deviance seen
        # so far.  If EM overshoots near convergence, we roll back — identical
        # in spirit to R CDM's save.devmin=TRUE.
        LL_prev   = -np.inf
        dev_min   = np.inf
        ip_min    = item_parm.copy()
        lp_min    = log_prior.copy()
        lpost_min : np.ndarray | None = None
        converged = False

        for itr in range(1, self.max_iter + 1):
            LL, log_post_u, Ng, Rg = self._estep(item_parm, log_prior, Y_uniq, freq)

            # M-step
            wpost_sum = np.exp(log_post_u) * freq[:, None]   # (n_uniq, L)
            wpost_cls = wpost_sum.sum(axis=0)                 # (L,)
            log_prior = self._mstep_prior(wpost_cls, N)
            item_parm = self._mstep_items(Ng, Rg, Lj)

            dev = -2 * LL
            if dev < dev_min:
                dev_min   = dev
                ip_min    = item_parm.copy()
                lp_min    = log_prior.copy()
                lpost_min = log_post_u

            delta = abs(LL - LL_prev)
            if self.verbose:
                print(f"\rIter={itr:4d}  |ΔLL|={delta:.2e}  deviance={dev:.4f}",
                      end="", flush=True)

            if delta < self.conv_crit and itr > 1:
                converged = True
                break
            LL_prev = LL

        if self.verbose:
            tag = "converged" if converged else "WARNING: not converged"
            print(f"\n{tag}  deviance={dev_min:.4f}  iter={itr}")

        # roll back to minimum-deviance solution
        self.item_parm  = ip_min
        self.log_prior  = lp_min
        self.log_post   = lpost_min[raw2unique]   # (N, L)
        self.deviance   = dev_min
        return self

    def person_parm(self, what: str = "mp") -> np.ndarray:
        """
        Compute person-level attribute estimates from the fitted posterior.

        Parameters
        ----------
        what : {'mp', 'eap', 'map'}, default 'mp'
            'mp'  — marginal mastery probability P(α_k=1 | X) for each
                    attribute k; shape (N, K), continuous ∈ (0, 1).
            'eap' — EAP classification: binarise mp > 0.5; shape (N, K).
            'map' — MAP classification: attribute profile with highest
                    posterior; shape (N, K).

        Returns
        -------
        ndarray, shape (N, K)
        """
        if self.log_post is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        post = np.exp(self.log_post)  # (N, L)
        what = what.lower()

        if what == "mp":
            return post @ self.att_pattern
        if what == "eap":
            return (post @ self.att_pattern > 0.5).astype(int)
        if what == "map":
            return self.att_pattern[np.argmax(self.log_post, axis=1)]
        raise ValueError(f"what must be 'mp', 'eap', or 'map', got {what!r}")

    def item_table(self) -> list[dict]:
        """
        Return item parameters as a list of dicts (suitable for pd.DataFrame).

        Each record contains:
            item    : 1-indexed item number
            group   : 1-indexed reduced group index
            pattern : tuple of 0/1 for the required attributes
            P       : estimated success probability for this group
        """
        if self.item_parm is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        rows = []
        for j in range(self.parloc.shape[0]):
            Lj_j  = int(self.parloc[j].max())
            Kj    = int(np.log2(Lj_j)) if Lj_j > 1 else 0
            pattj = _skill_profiles(Kj)
            for k in range(Lj_j):
                rows.append({
                    "item"   : j + 1,
                    "group"  : k + 1,
                    "pattern": tuple(int(x) for x in pattj[k]),
                    "P"      : float(self.item_parm[j, k]),
                })
        return rows


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(0)
    K  = 2
    ap = _skill_profiles(K)
    Q  = np.array([[1,0],[0,1],[1,1],[1,0],[0,1]], dtype=float)
    N  = 800

    true_prior = np.array([0.2, 0.3, 0.2, 0.3])
    cls        = np.random.choice(4, size=N, p=true_prior)
    persons    = ap[cls]

    true_g = [0.10, 0.10, 0.05, 0.10, 0.15]
    true_s = [0.10, 0.10, 0.05, 0.10, 0.10]
    Y = np.zeros((N, 5))
    for j in range(5):
        req = np.where(Q[j] > 0)[0]
        eta = np.all(persons[:, req] == 1, axis=1).astype(float)
        Y[:, j] = np.random.binomial(1, (1 - true_s[j]) * eta + true_g[j] * (1 - eta))

    print("=" * 55)
    print("GDINA quick test  (N=800, K=2, J=5, DINA-true)")
    print("=" * 55)
    mod = GDINA(att_dist="saturated", max_iter=500, conv_crit=1e-6,
                n_starts=3, verbose=True)
    mod.fit(Y, Q)

    true_P = {
        0: [true_g[0], 1 - true_s[0]],
        1: [true_g[1], 1 - true_s[1]],
        2: [true_g[2], true_g[2], true_g[2], 1 - true_s[2]],
        3: [true_g[3], 1 - true_s[3]],
        4: [true_g[4], 1 - true_s[4]],
    }
    print("\n── item parameters ──────────────────────")
    print(f"{'item':>4}  {'group':>5}  {'pattern':<10}  {'P_hat':>7}  {'P_true':>7}")
    for row in mod.item_table():
        j = row["item"] - 1; k = row["group"] - 1
        tp = true_P[j][k] if k < len(true_P[j]) else float("nan")
        print(f"  {row['item']:2d}     {row['group']:2d}   {str(row['pattern']):<10}"
              f"  {row['P']:7.4f}  {tp:7.4f}")

    print("\n── attribute prior ──────────────────────")
    print(f"  {'class':>5}  {'profile':<12}  {'π_hat':>7}  {'π_true':>7}")
    for l, (profile, lp) in enumerate(zip(mod.att_pattern, mod.log_prior)):
        print(f"  {l+1:5d}  {str(profile):<12}  {np.exp(lp):7.4f}  {true_prior[l]:7.4f}")

    eap = mod.person_parm("eap")
    mp  = mod.person_parm("mp")
    print("\n── first 8 persons: true | EAP | mp ────")
    for i in range(8):
        print(f"  {persons[i]}  |  {eap[i]}  |  {mp[i].round(3)}")


# ─────────────────────────────────────────────────────────────
# Check If the Q Matrix is Identifiable.
# ─────────────────────────────────────────────────────────────

import numpy as np

def check_q_identifiability(Q):
    """
    Q-matrix 可辨識性與完備性檢查 (Identifiability & Completeness Check)
    
    參考文獻:
    1. Chiu, C. Y., Douglas, J. A., & Li, X. (2009). Completeness Condition. 
    2. de la Torre, J. (2011). Identifiability in GDINA.
    """
    J, K = Q.shape
    # 生成 2^K 個潛在屬性剖面 (Attribute Profiles)
    from pygdina import _skill_profiles
    att_patterns = _skill_profiles(K) 
    L = len(att_patterns)
    
    # --- 1. 檢查 強可辨識性 (Strong Identifiability) ---
    # 依據 Chiu et al. (2009)，檢查 Q 是否包含 KxK 單位矩陣
    missing_pure_skills = []
    for k in range(K):
        target = np.zeros(K)
        target[k] = 1
        if not any(np.array_equal(row, target) for row in Q):
            missing_pure_skills.append(f"A{k+1}")
            
    is_strongly_identifiable = (len(missing_pure_skills) == 0)

    # --- 2. 檢查 弱可辨識性 (Weak Identifiability) ---
    # 檢查不同剖面產生的理想反應模式 (Ideal Response Patterns) 是否唯一
    ideal_responses = []
    for l in range(L):
        alpha = att_patterns[l]
        # eta_jl = 1 代表受測者擁有的能力足以應付題目 j 的要求
        eta = np.all(alpha >= Q, axis=1).astype(int)
        ideal_responses.append(tuple(eta))
    
    unique_responses = set(ideal_responses)
    is_weakly_identifiable = (len(unique_responses) == L)

    # --- 3. 診斷情況說明與後果分析 ---
    report = {"is_identifiable": is_weakly_identifiable}

    if not is_weakly_identifiable:
        report["level"] = "不可辨識 (Unidentifiable)"
        report["details"] = "【結構缺失】不同的屬性剖面產生完全相同的作答模式。"
        report["consequence"] = ("後果：模型參數無唯一解，EM 演算法將無法收斂或隨機收斂，"
                                 "分類結果 (EMR) 將失去心理計量學意義。")
        
        # 找出混淆組別 (使用純 dict 搭配 setdefault 代替 defaultdict)
        groups = {}
        for idx, res in enumerate(ideal_responses):
            groups.setdefault(res, []).append(att_patterns[idx].tolist())
        report["aliased_profiles"] = [v for v in groups.values() if len(v) > 1]
        
    elif not is_strongly_identifiable:
        report["level"] = "弱可辨識 (Weakly Identifiable)"
        report["details"] = f"【非完備】滿足區分要求但缺失屬性 {missing_pure_skills} 的純測項。"
        report["consequence"] = ("後果：依據 de la Torre (2011)，模型雖可解，但迭代次數會顯著增加，"
                                 "且題參數與能力分類的標準誤 (SE) 會較大，易受隨機誤差干擾。")
        report["aliased_profiles"] = []
        
    else:
        report["level"] = "強可辨識 (Strongly Identifiable)"
        report["details"] = "【最優結構】滿足 Chiu et al. (2009) 完備性條件 (Completeness Condition)。"
        report["consequence"] = "後果：具備最強的統計檢定力，EM 演算法收斂速度最快，且分類準確度最為穩定。"
        report["aliased_profiles"] = []

    return report