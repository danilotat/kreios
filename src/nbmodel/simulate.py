import numpy as np

class RiboseqSimulator:
    def __init__(self, n_codons=300, variant_pos=150, seed=42):
        self.M = n_codons
        self.k = variant_pos
        self.rng = np.random.default_rng(seed)
        self.positions = np.arange(self.M)
        self.mask = (self.positions >= self.k).astype(float)

    def simulate(
            self, 
            # --- Abundance Params ---
            beta0=None,         # Log-baseline. If None, calculated from avg_reads_per_codon
            avg_reads=5.0,      # EASIER TO TUNE: Target average reads per codon
            alpha_ramp=1.0,     # Ramp height
            tau_ramp=20.0,      # Ramp decay
            delta_abund=-2.0,   # Effect size
            phi=0.5,            # DISPERSION: Lower phi (e.g. 0.1-0.5) = More zeros/noise
            
            # --- Frame Params ---
            alpha_base=np.array([10.0, 0.5, 0.5]), 
            delta_shape=np.array([0.0, 0.0, 0.0])
            ):
        
        # 1. Set Baseline (beta0) based on desired depth
        if beta0 is None:
            # simple heuristic to set intercept to match desired mean roughly
            beta0 = np.log(avg_reads)
            
        # 2. Calculate Expected Means (The "True" Biology)
        ramp = alpha_ramp * np.exp(-self.positions / tau_ramp)
        step = delta_abund * self.mask
        log_mu = beta0 + ramp + step
        mu = np.exp(log_mu)
        
        # 3. Sample Total Counts (Negative Binomial)
        # Low phi = High dispersion = Many zeros (Gamma distribution becomes very skewed)
        # We use Gamma-Poisson formulation
        
        # Sample lambda for each codon from Gamma
        lam = self.rng.gamma(shape=phi, scale=mu/phi)
        
        # Sample observed count from Poisson
        total_counts = self.rng.poisson(lam)
        
        # 4. Sample Triplets (Dirichlet-Multinomial)
        alphas = np.tile(alpha_base, (self.M, 1))
        alpha_down = alpha_base * np.exp(delta_shape)
        alphas[self.mask == 1] = alpha_down
        
        triplet_counts = np.zeros((self.M, 3), dtype=int)
        
        for i in range(self.M):
            if total_counts[i] > 0:
                p = self.rng.dirichlet(alphas[i])
                triplet_counts[i] = self.rng.multinomial(total_counts[i], p)
            else:
                # Explicitly 0 if total is 0 (already init with zeros, but for clarity)
                triplet_counts[i] = [0, 0, 0]
        
        return {
            "positions": self.positions,
            "total_counts": total_counts,
            "triplet_counts": triplet_counts,
            "true_mu": mu,
            "variant_pos": self.k
        }

if __name__ == '__main__':
    pass
