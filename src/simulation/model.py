import numpy as np
import pymc as pm


class NBFrameModelHierarchical:
    """
    Hierarchical Negative-Binomial model at nucleotide resolution (one observation per nt).
      - N = number of nucleotides (e.g. CDS length in nt, 3 * n_codons)
      - codon_ids: integer array, length N; codon index for nucleotide i (nt_index // 3)
                    values in 0..60 (61 sense codons)
      - frames: integer array length N; p-site frame for nucleotide i (nt_index % 3)
                values in {0,1,2} or experimentally inferred frame per nucleotide
      - positions: numeric array length N; codon position for nucleotide i (nt_index // 3)
      - counts: integer array length N; observed P-site counts at nucleotide positions
      - log_lambda: scalar or array length N (library-size / normalization offset on log scale)
    """

    def __init__(self, codon_ids, frames, positions, counts=None, log_lambda=0.0, priors=None):
        self.codon_ids = np.asarray(codon_ids, dtype=int)
        self.frames = np.asarray(frames, dtype=int)
        self.positions = np.asarray(positions, dtype=float)
        self.counts = None if counts is None else np.asarray(counts, dtype=int)
        self.N = len(self.codon_ids)

        if np.isscalar(log_lambda):
            self.log_lambda = np.full(self.N, float(log_lambda))
        else:
            self.log_lambda = np.asarray(log_lambda, dtype=float)

        assert len(self.frames) == self.N and len(self.positions) == self.N, \
            "codon_ids, frames and positions must have the same length (nucleotide resolution)."

        # Default prior callables (return PyMC RVs when called inside model context)
        default_priors = {
            "beta0": lambda: pm.Normal("beta0", mu=0.0, sigma=2.0),
            "beta_codon": lambda: pm.Normal("beta_codon", mu=0.0, sigma=0.5, shape=61),
            "alpha_ramp": lambda: pm.HalfNormal("alpha_ramp", mu=0.0, sigma=1.0),
            "tau_ramp": lambda: pm.Gamma("tau_ramp", alpha=2.0, beta=0.5),
            "phi": lambda: pm.Gamma("phi", alpha=2.0, beta=0.5),
            "kappa": lambda: pm.HalfNormal("kappa", sigma=1.0),
            "frame_base_prior": lambda: pm.Normal("frame_base_raw", mu=0.0, sigma=1.0, shape=3),
        }
        self.priors = {**default_priors, **(priors or {})}

        self.model = self._build_model()
        self.trace = None

    def __repr__(self):
        return (f"<NBFrameModelHierarchical N={self.N}, priors={list(self.priors.keys())}, "
                f"model={self.model}>")

    def _build_model(self):
        with pm.Model() as model:
            # Mutable data: created once and referenced later
            codon_ids_md = pm.Data("codon_ids", self.codon_ids)
            frames_md = pm.Data("frames", self.frames)
            positions_md = pm.Data("positions", self.positions)
            log_lambda_md = pm.Data("log_lambda", self.log_lambda)

            # Priors (callables)
            beta0 = self.priors["beta0"]()
            beta_codon = self.priors["beta_codon"]()
            alpha_ramp = self.priors["alpha_ramp"]()
            tau_ramp = self.priors["tau_ramp"]()
            phi = self.priors["phi"]()
            kappa = self.priors["kappa"]()
            frame_base_raw = self.priors["frame_base_prior"]()  # shape (3,)

            # Center frame_base_raw to sum-to-zero
            frame_base = pm.Deterministic("frame_base", frame_base_raw - pm.math.mean(frame_base_raw))

            # frame-specific deltas (sum-to-zero) and per-nucleotide delta via indexing
            delta_frame = pm.Deterministic("delta_frame", kappa * frame_base)   # shape (3,)
            delta_pos = pm.Deterministic("delta_pos", delta_frame[frames_md])   # shape (N,)

            # Expected log-mean (nucleotide-wise). codon-specific effect indexed by codon_ids_md
            log_mu = (
                log_lambda_md
                + beta0
                + beta_codon[codon_ids_md]
                + alpha_ramp * pm.math.exp(-positions_md / tau_ramp)
                + delta_pos
            )
            mu = pm.math.exp(log_mu)

            pm.NegativeBinomial("y", mu=mu, alpha=phi, observed=self.counts)

        return model

    def fit(self, draws=1000, tune=1000, chains=4, **sample_kwargs):
        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, **sample_kwargs)
        return self.trace

    @staticmethod
    def _flatten_chain_draws(arr):
        if arr.ndim == 1:
            return arr.reshape(-1)
        newshape = (arr.shape[0] * arr.shape[1],) + arr.shape[2:]
        return arr.reshape(newshape)

    def generate_data(self, t=None, posterior=None, n_samples=1, random_seed=None):
        if posterior is None:
            if self.trace is None:
                raise ValueError("No posterior available: fit the model or provide 'posterior'.")
            posterior = self.trace

        rng = np.random.default_rng(random_seed)
        post = posterior.posterior

        beta0_all = self._flatten_chain_draws(post.beta0.values)
        beta_codon_all = self._flatten_chain_draws(post.beta_codon.values)
        alpha_ramp_all = self._flatten_chain_draws(post.alpha_ramp.values)
        tau_ramp_all = self._flatten_chain_draws(post.tau_ramp.values)
        phi_all = self._flatten_chain_draws(post.phi.values)
        kappa_all = self._flatten_chain_draws(post.kappa.values)

        use_learned_delta = (t is None)
        if use_learned_delta:
            delta_frame_all = self._flatten_chain_draws(post.delta_frame.values)
        else:
            if t not in (0, 1, 2):
                raise ValueError("t must be 0, 1, or 2 when provided.")
            pattern = np.array([1.0, -0.5, -0.5], dtype=float)

        M = beta0_all.shape[0]
        replace = n_samples > M
        idxs = rng.choice(M, size=n_samples, replace=replace)

        simulated_list = []
        for idx in idxs:
            beta0 = np.atleast_1d(beta0_all[idx]).item()
            beta_codon = beta_codon_all[idx]
            alpha_ramp = np.atleast_1d(alpha_ramp_all[idx]).item()
            tau_ramp = np.atleast_1d(tau_ramp_all[idx]).item()
            phi = np.atleast_1d(phi_all[idx]).item()
            kappa = np.atleast_1d(kappa_all[idx]).item()

            if use_learned_delta:
                delta_frame = delta_frame_all[idx]
            else:
                delta_frame = kappa * pattern

            delta_pos = delta_frame[self.frames]  # nucleotide-wise mapping
            log_mu = (
                self.log_lambda
                + beta0
                + beta_codon[self.codon_ids]
                + alpha_ramp * np.exp(-self.positions / tau_ramp)
                + delta_pos
            )
            mu = np.exp(log_mu)

            # Gamma-Poisson sampling to match NB(mu, phi)
            lambda_draws = rng.gamma(shape=phi, scale=(mu / phi))
            y_sim = rng.poisson(lam=lambda_draws)
            simulated_list.append(y_sim)

        return np.vstack(simulated_list)  # shape (n_samples, N)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    import arviz as az
    import numpy as np
    import pymc as pm
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    sns.set(style="whitegrid", context="talk")

    # --- Synthetic data generation settings ---
    rng = np.random.default_rng(42)
    n_codons = 61  # 61 sense codons in your model
    n_codons_used = 30  # number of codons actually present in the test transcript
    codon_order = rng.choice(np.arange(n_codons), size=n_codons_used, replace=False)

    # build a transcript of M codons -> nucleotide resolution N = 3*M
    M = 100  # number of codons (change as desired)
    N = 3 * M

    # per-nucleotide arrays expected by the class
    codon_indices = np.repeat(np.arange(M), 3)  # 0..M-1 repeated 3 times
    # map codon index to the codon id (0..60)
    codon_ids = np.array([codon_order[i % n_codons_used] for i in codon_indices], dtype=int)
    frames = np.array([i % 3 for i in range(N)], dtype=int)
    positions = codon_indices.astype(float)  # position measured in codons

    # --- True generative parameters (ground truth) ---
    true = {}
    true["beta0"] = -1.0
    # codon effects: draw from Normal(0, 0.5) but only for the 61 codons
    true["beta_codon"] = rng.normal(loc=0.0, scale=0.5, size=n_codons)
    true["alpha_ramp"] = 1.2
    true["tau_ramp"] = 25.0
    true["phi"] = 5.0   # overdispersion (Gamma-Poisson)
    true["kappa"] = 0.8
    # frame base pattern (will be centered to sum-to-zero by model)
    frame_base_pattern = np.array([1.0, -0.5, -0.5])
    # center like model does: frame_base_raw - mean(frame_base_raw)
    raw = frame_base_pattern.copy()
    raw = raw + rng.normal(scale=0.01, size=3)  # small jitter
    raw = raw - raw.mean()
    true["frame_base_raw"] = raw
    true["delta_frame"] = true["kappa"] * (true["frame_base_raw"] - true["frame_base_raw"].mean())

    # offsets (log_lambda): constant library size offset
    log_lambda = np.log(1.0)  # zero offset for simplicity

    # --- Simulate counts using Gamma-Poisson (Negative-binomial) ---
    delta_pos = true["delta_frame"][frames]  # nucleotide-wise delta from frame
    log_mu = (
        log_lambda
        + true["beta0"]
        + true["beta_codon"][codon_ids]
        + true["alpha_ramp"] * np.exp(-positions / true["tau_ramp"])
        + delta_pos
    )
    mu = np.exp(log_mu)
    # Gamma-Poisson parameterization: lambda ~ Gamma(phi, scale=mu/phi)
    lambda_draws = rng.gamma(shape=true["phi"], scale=(mu / true["phi"]))
    y_obs = rng.poisson(lam=lambda_draws).astype(int)

    # --- Instantiate model with synthetic data ---
    model_obj = NBFrameModelHierarchical(
        codon_ids=codon_ids,
        frames=frames,
        positions=positions,
        counts=y_obs,
        log_lambda=log_lambda,
    )

    print("Model summary:", model_obj)

    # --- Fit the model ---
    # Keep draws/tune small for example speed; increase for real use
    draws = 800
    tune = 800
    chains = 2

    print("Running sampling... (this may take a minute)")
    trace = model_obj.fit(draws=draws, tune=tune, chains=chains, random_seed=42, cores=1)

    # Convert to arviz InferenceData (pm.sample already returns an ArviZ-compatible structure)
    idata = az.from_pymc3(trace) if False else trace  # PyMC v4/5 returns InferenceData already
    # if trace is an xarray InferenceData, use that directly
    idata = trace

    # --- Posterior predictive sampling ---
    with model_obj.model:
        # In PyMC v4/5, pm.sample_posterior_predictive accepts `trace` or `idata`
        ppc = pm.sample_posterior_predictive(trace, var_names=["y"], random_seed=123, progressbar=False)

    y_ppc = ppc["y"]  # shape (n_draws_total, N)

    # --- Diagnostic plots: trace & divergences ---
    fig1 = az.plot_trace(idata, var_names=["beta0", "alpha_ramp", "tau_ramp", "phi", "kappa"], compact=True)
    fig1[0].suptitle("Trace + posterior for selected parameters", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Energy / R-hat summary
    print("Summary (selected variables):")
    display_vars = ["beta0", "alpha_ramp", "tau_ramp", "phi", "kappa", "delta_frame"]
    try:
        summary = az.summary(idata, var_names=display_vars, round_to=3)
        print(summary)
    except Exception:
        # fallback to az.summary without var restriction
        print(az.summary(idata, round_to=3))

    # --- Compare posterior means to true parameter values ---
    # helper to get posterior mean
    def post_mean(name):
        try:
            return idata.posterior[name].mean(dim=("chain", "draw")).values
        except Exception:
            return idata.posterior[name].mean(dim=("chain", "draw")).values

    pm_beta0 = float(post_mean("beta0"))
    pm_alpha_ramp = float(post_mean("alpha_ramp"))
    pm_tau_ramp = float(post_mean("tau_ramp"))
    pm_phi = float(post_mean("phi"))
    pm_kappa = float(post_mean("kappa"))

    print("\nTrue vs posterior mean (scalar params):")
    for nm, tval, pval in [
        ("beta0", true["beta0"], pm_beta0),
        ("alpha_ramp", true["alpha_ramp"], pm_alpha_ramp),
        ("tau_ramp", true["tau_ramp"], pm_tau_ramp),
        ("phi", true["phi"], pm_phi),
        ("kappa", true["kappa"], pm_kappa),
    ]:
        print(f"  {nm:12s}  true = {tval:8.3f}   posterior_mean = {pval:8.3f}")

    # Frame delta: posterior deterministic delta_frame
    posterior_delta_frame = idata.posterior["delta_frame"].mean(dim=("chain", "draw")).values
    print("\nTrue delta_frame:", true["delta_frame"])
    print("Posterior mean delta_frame:", posterior_delta_frame)

    # --- Codon effect recovery: compare true beta_codon to posterior means (for codons present in data) ---
    # Posterior beta_codon shape: (chain, draw, 61)
    beta_codon_post_mean = idata.posterior["beta_codon"].mean(dim=("chain", "draw")).values  # shape (61,)
    codons_present = np.unique(codon_ids)
    # scatter true vs posterior for codons present
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true["beta_codon"][codons_present], beta_codon_post_mean[codons_present], s=40, alpha=0.8)
    mn = min(true["beta_codon"][codons_present].min(), beta_codon_post_mean[codons_present].min())
    mx = max(true["beta_codon"][codons_present].max(), beta_codon_post_mean[codons_present].max())
    ax.plot([mn, mx], [mn, mx], color="k", linestyle="--")
    ax.set_xlabel("True beta_codon")
    ax.set_ylabel("Posterior mean beta_codon")
    ax.set_title("Codon effects: true vs posterior mean (codons present in transcript)")
    plt.tight_layout()
    plt.show()

    # --- Posterior predictive checks: observed counts vs predictive distribution ---
    # compute posterior predictive mean per nucleotide
    y_ppc_mean = y_ppc.mean(axis=0)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # plot observed vs predictive mean along transcript (nucleotide index)
    ax[0].plot(np.arange(N), y_obs, label="observed", color="C0", alpha=0.7)
    ax[0].plot(np.arange(N), y_ppc_mean, label="ppc mean", color="C1", alpha=0.8)
    ax[0].legend()
    ax[0].set_ylabel("counts")
    ax[0].set_title("Observed counts vs posterior predictive mean along transcript (nucleotide resolution)")

    # distributional check: overlay histograms
    ax[1].hist(y_ppc.flatten(), bins=np.arange(0, np.max(y_ppc)+2)-0.5, alpha=0.3, density=True, label="ppc (all draws)")
    ax[1].hist(y_obs, bins=np.arange(0, np.max(y_obs)+2)-0.5, alpha=0.7, density=True, label="observed")
    ax[1].set_xlabel("counts")
    ax[1].set_ylabel("density")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # --- Simple "how much the model learned" metric ---
    # MSE between observed and posterior predictive mean
    mse = np.mean((y_obs - y_ppc_mean) ** 2)
    var_obs = np.var(y_obs)
    explained = 1.0 - mse / var_obs if var_obs > 0 else np.nan
    print(f"\nPosterior predictive MSE: {mse:.3f}")
    print(f"Observed counts variance: {var_obs:.3f}")
    print(f"Fraction of variance explained by posterior predictive mean (1 - MSE/Var): {explained:.3f}")

    # Also show interval coverage: fraction of observed points inside 50% and 90% predictive intervals
    lower50, upper50 = np.percentile(y_ppc, [25, 75], axis=0)
    lower90, upper90 = np.percentile(y_ppc, [5, 95], axis=0)
    in50 = np.mean((y_obs >= lower50) & (y_obs <= upper50))
    in90 = np.mean((y_obs >= lower90) & (y_obs <= upper90))
    print(f"Fraction of nucleotides within 50% PPC interval: {in50:.3f}")
    print(f"Fraction of nucleotides within 90% PPC interval: {in90:.3f}")

    # --- Optional: show a small table of true vs posterior for first few codons/positions ---
    print("\nSample parameter recovery (first 8 codons present):")
    for i, cod in enumerate(codons_present[:8]):
        print(f" codon {cod:2d}   true beta = {true['beta_codon'][cod]: .3f}   posterior mean = {beta_codon_post_mean[cod]: .3f}")

    print("\nDone.")

