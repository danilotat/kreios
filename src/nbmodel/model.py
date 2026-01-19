# # Rewrite of the original model for computing alterations in riboseq profiles.
# import torch
# import pyro
# import pyro.distributions as dist
# from pyro.infer import SVI, Trace_ELBO
# from pyro.infer.autoguide import AutoNormal
# from torch.distributions import constraints


# class KreiosModel:
#     def __init__(self, frames, positions, counts, variant_idx: int):
#         self.frames = torch.tensor(frames, dtype=torch.long)       # (batch_size, seq_len)
#         self.positions = torch.tensor(positions, dtype=torch.float) # (batch_size, seq_len)
#         self.counts = torch.tensor(counts, dtype=torch.float)       # (batch_size, seq_len, 3)
#         self.N, self.M = self.positions.shape  # batch_size, seq_len
#         self.total_counts = self.counts.sum(dim=-1)  # (batch_size, seq_len)
#         self.mask = (self.positions >= variant_idx).float()

#         # self.frames = torch.tensor(frames, dtype=torch.long) #  (batch_size, seq_len/3)
#         # self.positions = torch.tensor(positions, dtype=torch.float) # (batch_size, seq_len/3, )
#         # self.counts = torch.tensor(counts, dtype=torch.float) # (batch_size, seq_len/3, 3)
#         # self.M = len(frames)
#         # # counts for each codon 
#         # self.total_counts = self.counts.sum(dim=2) # (batch_size, seq_len/3)
#         # # we should split then the input tensor to be first/after variant. 
#         # # we will use this as a binary step function to define where we want to see the effect.
#         # self.mask = (self.positions >= variant_idx).float() 

#     def model(self, positions, total_counts, counts_triplet, mask):
#         # positions: (batch_size, seq_len)
#         # total_counts: (batch_size, seq_len)
#         # counts_triplet: (batch_size, seq_len, 3)
#         # mask: (batch_size, seq_len)
        
#         batch_size, seq_len = positions.shape
        
#         # Global parameters
#         beta0 = pyro.sample("beta0", dist.Normal(0., 2.))
#         alpha_ramp = pyro.sample("alpha_ramp", dist.LogNormal(0., 1.))
#         tau_ramp = pyro.sample("tau_ramp", dist.LogNormal(3., 1.))
#         delta_abund = pyro.sample("delta_abund", dist.Normal(0., 2.))
#         phi = pyro.sample("phi", dist.Exponential(1.))
        
#         ramp = alpha_ramp * torch.exp(-positions / tau_ramp)
#         log_mu = beta0 + ramp + (delta_abund * mask)
#         mu = torch.exp(log_mu)
#         logits_nb = torch.log(phi) - torch.log(mu)
        
#         # Nested plates: batch is dim=-2, codons is dim=-1
#         with pyro.plate("batch", batch_size, dim=-2):
#             with pyro.plate("codons", seq_len, dim=-1):
#                 # Mask out padded positions from likelihood
#                 with pyro.poutine.mask(mask=padding_mask.bool()):
#                     pyro.sample("obs_abundance", 
#                                 dist.NegativeBinomial(total_count=phi, logits=logits_nb), 
#                                 obs=total_counts)
        
#         # Frame shift parameters
#         base_alpha = pyro.sample("base_alpha_vec", dist.LogNormal(0., 1.).expand([3]).to_event(1))
#         delta_shape = pyro.sample("delta_shape_vec", dist.Normal(0., 1.).expand([3]).to_event(1))
#         downstream_alpha = torch.exp(torch.log(base_alpha) + delta_shape)
#         current_alpha = base_alpha + (downstream_alpha - base_alpha) * mask.unsqueeze(-1)
        
#         with pyro.plate("batch_shape", batch_size, dim=-2):
#             with pyro.plate("codons_shape", seq_len, dim=-1):
#                 pyro.sample("obs_frame", 
#                             dist.DirichletMultinomial(total_count=total_counts, concentration=current_alpha), 
#                             obs=counts_triplet)



#         # batch_size, seq_len = positions.shape
#         # # Ramp parameters to reflect the decrease while moving from the 5' to the end.
#         # # the idea is to model the decay as parametrized from a baseline level with an exponential decay
#         # # along the transcript. 
#         # # log(λ(x_i)) = β_0 + α_ramp * exp(-(x_i)/τ_ramp)
#         # beta0 = pyro.sample("beta0", dist.Normal(0., 2.)) # baseline expression
#         # alpha_ramp = pyro.sample("alpha_ramp", dist.LogNormal(0., 1.)) # amplitude of 5' accumulation (α_ramp > 0)
#         # tau_ramp = pyro.sample("tau_ramp", dist.LogNormal(3., 1.)) # that's the true decay parameter
#         # # Counts are modeled as the results of NB distribution.
#         # # N_i ~ NB(μ_i, ϕ)
#         # # following the reparametrization, we get
#         # # log(μ_i) = log(λ(x_i)) + δ_abund 
#         # # δ_abund is a the LFC downstream of the variant. this could be either positive or negative
#         # delta_abund = pyro.sample("delta_abund", dist.Normal(0., 2.))
#         # phi = pyro.sample("phi", dist.Exponential(1.)) # simple dispersion for the NB
#         # ramp = alpha_ramp * torch.exp(-positions / tau_ramp)
#         # log_mu = beta0 + ramp + (delta_abund * mask)
#         # mu = torch.exp(log_mu)
#         # # First likelihood to compute using the NB, incorporating the ramp decay
#         # # Using parametrization (total_count=phi, logits=log(phi)-log(mu))
#         # logits_nb = torch.log(phi) - torch.log(mu)
#         # with pyro.plate("codons", self.M):
#         #     # NOTE: this is working with the sum of codons.
#         #     pyro.sample("obs_abundance", 
#         #                 dist.NegativeBinomial(total_count=phi, logits=logits_nb), 
#         #                 obs=total_counts)
#         # # To detect frameshift, the idea is that we would observe a strong deviation from the conventional
#         # # translation frame. In ideal conditions, the concentration will show [High, Mid/Low, Low] for frames [0,1,2].
#         # # The more conventional way to model this is with a Multinomial Dirichlet, where we impose that they should 
#         # # adhere to a dirichlet concentration vector alpha to define the preference.

#         # # Base concentration, relative to the region before variant occurrence. Constrained to be positive.
#         # base_alpha = pyro.sample("base_alpha_vec", dist.LogNormal(0., 1.).expand([3]).to_event(1))        
#         # # Shift towards other frames. 
#         # delta_shape = pyro.sample("delta_shape_vec", dist.Normal(0., 1.).expand([3]).to_event(1))        
#         # # Calculate Downstream Alpha
#         # # downstream: exp(log(base_alpha) + delta)
#         # downstream_alpha = torch.exp(torch.log(base_alpha) + delta_shape)
#         # # Combine using mask
#         # current_alpha = base_alpha + (downstream_alpha - base_alpha) * mask.unsqueeze(-1) # output is (batch_size, seq_len/3, 3)

#         # with pyro.plate("codons_shape", self.M):
#         #     # distribution of counts from the codon is y_i 
#         #     # y_i ~ DirichletMultinomial(N_i, α_i)
#         #     pyro.sample("obs_frame", 
#         #                 dist.DirichletMultinomial(total_count=total_counts, concentration=current_alpha), 
#         #                 obs=counts_triplet)

import torch
import pyro
import pyro.distributions as dist
from pyro import poutine


class KreiosModel:
    def __init__(self, variant_idx: int = None):
        """
        Initialize KreiosModel.
        
        Args:
            variant_idx: Optional fixed variant index. If None, mask must be 
                        provided to model() call (for variable variant positions).
        """
        self.variant_idx = variant_idx

    def model(self, positions, total_counts, counts_triplet, mask, padding_mask=None):
        """
        Pyro model for ribosome profiling data.
        
        Args:
            positions: (batch_size, seq_len) codon positions
            total_counts: (batch_size, seq_len) total reads per codon
            counts_triplet: (batch_size, seq_len, 3) frame counts
            mask: (batch_size, seq_len) downstream-of-variant mask (1 = downstream)
            padding_mask: (batch_size, seq_len) valid position mask (1 = valid, 0 = padding)
                         If None, assumes no padding (all positions valid).
        """
        batch_size, seq_len = positions.shape
        
        # Default padding mask: all valid
        if padding_mask is None:
            padding_mask = torch.ones_like(positions)
        
        # Convert to boolean for pyro.poutine.mask
        valid_mask = padding_mask.bool()
        
        # =====================
        # Global parameters
        # =====================
        
        # Abundance model parameters
        beta0 = pyro.sample("beta0", dist.Normal(0., 2.))
        alpha_ramp = pyro.sample("alpha_ramp", dist.LogNormal(0., 1.))
        tau_ramp = pyro.sample("tau_ramp", dist.LogNormal(2.7, 0.5))
        delta_abund = pyro.sample("delta_abund", dist.Normal(0., 2.))
        phi = pyro.sample("phi", dist.Exponential(1.))
        
        # Frame model parameters
        base_alpha = pyro.sample(
            "base_alpha_vec", 
            dist.LogNormal(0., 1.).expand([3]).to_event(1)
        )
        delta_shape = pyro.sample(
            "delta_shape_vec", 
            dist.Normal(0., 1.).expand([3]).to_event(1)
        )
        
        # =====================
        # Abundance likelihood
        # =====================
        
        # Compute expected log-counts
        ramp = alpha_ramp * torch.exp(-positions / tau_ramp)
        log_mu = beta0 + ramp + (delta_abund * mask)
        mu = torch.exp(log_mu)
        
        # NegBin parameterization: logits = log(phi/mu)
        logits_nb = torch.log(phi) - torch.log(mu)
        
        with pyro.plate("batch", batch_size, dim=-2):
            with pyro.plate("codons", seq_len, dim=-1):
                with poutine.mask(mask=valid_mask):
                    pyro.sample(
                        "obs_abundance", 
                        dist.NegativeBinomial(total_count=phi, logits=logits_nb), 
                        obs=total_counts
                    )
        
        # =====================
        # Frame likelihood
        # =====================
        
        # Downstream alpha: base_alpha * exp(delta_shape)
        downstream_alpha = base_alpha * torch.exp(delta_shape)
        
        # Interpolate: upstream uses base_alpha, downstream uses downstream_alpha
        # mask: (batch_size, seq_len) -> unsqueeze for broadcasting with (3,)
        mask_expanded = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        current_alpha = base_alpha + (downstream_alpha - base_alpha) * mask_expanded
        # current_alpha: (batch_size, seq_len, 3)
        
        with pyro.plate("batch_frame", batch_size, dim=-2):
            with pyro.plate("codons_frame", seq_len, dim=-1):
                with poutine.mask(mask=valid_mask):
                    pyro.sample(
                        "obs_frame", 
                        dist.DirichletMultinomial(
                            total_count=total_counts, 
                            concentration=current_alpha
                        ), 
                        obs=counts_triplet
                    )
    
    @staticmethod
    def compute_mask(positions, variant_idx):
        """
        Compute downstream-of-variant mask.
        
        Args:
            positions: (batch_size, seq_len) or (seq_len,)
            variant_idx: int or (batch_size,) array of variant positions
            
        Returns:
            mask: same shape as positions, 1.0 where position >= variant_idx
        """
        if isinstance(variant_idx, int):
            return (positions >= variant_idx).float()
        else:
            # Per-sample variant positions
            variant_idx = torch.as_tensor(variant_idx).unsqueeze(-1)  # (batch_size, 1)
            return (positions >= variant_idx).float()