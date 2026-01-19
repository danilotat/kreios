import torch
import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal, AutoDiagonalNormal
from pyro.optim import ClippedAdam
import logging
from typing import Optional, Literal
import matplotlib.pyplot as plt
from model import KreiosModel
from dataloader import create_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KreiosTrainer:
    def __init__(
        self,
        model: KreiosModel,
        guide_type: Literal["normal", "diagonal"] = "normal",
        lr: float = 0.005,
        clip_norm: float = 10.0,
    ):
        self.model = model
        self.losses = []
        self.epoch_losses = []
        
        # Clear param store for fresh training
        pyro.clear_param_store()
        
        # Set up guide
        if guide_type == "normal":
            self.guide = AutoNormal(model.model)
        elif guide_type == "diagonal":
            self.guide = AutoDiagonalNormal(model.model)
        else:
            raise ValueError(f"Unknown guide type: {guide_type}")
        
        # Set up optimizer
        optimizer = ClippedAdam({"lr": lr, "clip_norm": clip_norm})
        
        # Set up SVI
        self.svi = SVI(
            model=model.model,
            guide=self.guide,
            optim=optimizer,
            loss=Trace_ELBO()
        )
    
    def train_step(self, batch: dict) -> float:
        """Single training step on a batch."""
        loss = self.svi.step(
            batch['positions'],
            batch['total_counts'],
            batch['counts'],
            batch['mask'],
            batch['padding_mask']
        )
        return loss
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch over the dataloader."""
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            loss = self.train_step(batch)
            self.losses.append(loss)
            epoch_loss += loss
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        self.epoch_losses.append(avg_loss)
        return avg_loss
    
    def train(
        self,
        dataloader,
        num_epochs: int = 100,
        log_every: int = 10,
    ) -> list:
        """
        Train model using dataloader.
        
        Args:
            dataloader: PyTorch DataLoader yielding batches
            num_epochs: Number of epochs to train
            log_every: Log every N epochs
            
        Returns:
            List of per-step losses
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader)
            
            if (epoch + 1) % log_every == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")
        
        logger.info("Training complete")
        return self.losses
    
    def train_full_batch(
        self,
        positions,
        total_counts,
        counts,
        mask,
        padding_mask=None,
        num_steps: int = 1000,
        log_every: int = 100,
    ) -> list:
        """
        Train on full batch (no dataloader). 
        Useful for small datasets or debugging.
        """
        logger.info(f"Starting full-batch training for {num_steps} steps")
        
        for step in range(num_steps):
            loss = self.svi.step(
                positions, total_counts, counts, mask, padding_mask
            )
            self.losses.append(loss)
            
            if (step + 1) % log_every == 0:
                avg_loss = sum(self.losses[-log_every:]) / log_every
                logger.info(f"Step {step + 1}/{num_steps} | Loss: {avg_loss:.4f}")
        
        logger.info("Training complete")
        return self.losses
    
    def get_posterior_samples(self, batch: dict, num_samples: int = 1000) -> dict:
        """Draw samples from the learned posterior."""
        predictive = Predictive(
            self.model.model,
            guide=self.guide,
            num_samples=num_samples,
            return_sites=[
                "beta0", "alpha_ramp", "tau_ramp", 
                "delta_abund", "phi",
                "base_alpha_vec", "delta_shape_vec"
            ]
        )
        samples = predictive(
            batch['positions'],
            batch['total_counts'],
            batch['counts'],
            batch['mask'],
            batch['padding_mask']
        )
        return {k: v.detach().cpu() for k, v in samples.items()}
    
    def get_posterior_summary(self, batch: dict, num_samples: int = 1000) -> dict:
        """Get mean and std of posterior samples."""
        samples = self.get_posterior_samples(batch, num_samples)
        summary = {}
        for name, values in samples.items():
            summary[name] = {
                "mean": values.mean(dim=0),
                "std": values.std(dim=0),
                "q05": values.quantile(0.05, dim=0),
                "q95": values.quantile(0.95, dim=0),
            }
        return summary
    
    def plot_loss(self, savepath: Optional[str] = None, plot_epochs: bool = False):
        """Plot training loss curve."""
        fig, ax = plt.subplots(figsize=(10, 4))
        
        if plot_epochs and self.epoch_losses:
            ax.plot(self.epoch_losses)
            ax.set_xlabel("Epoch")
        else:
            ax.plot(self.losses)
            ax.set_xlabel("Step")
        
        ax.set_ylabel("ELBO Loss")
        ax.set_title("Training Loss")
        
        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
        
        return fig, ax


def train_kreios(
    transcripts: list[dict],
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 0.005,
    guide_type: Literal["normal", "diagonal"] = "normal",
    seed: int = 42,
) -> tuple[KreiosModel, KreiosTrainer, dict]:
    """
    Convenience function to train KreiosModel with a dataloader.
    
    Args:
        transcripts: List of transcript dicts with 'positions', 'counts', 'variant_idx'
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        lr: Learning rate
        guide_type: Type of variational guide
        seed: Random seed
    
    Returns:
        model: The KreiosModel instance
        trainer: The KreiosTrainer instance
        summary: Posterior summary statistics
    """
    pyro.set_rng_seed(seed)
    
    # Create dataloader
    dataloader = create_dataloader(transcripts, batch_size=batch_size, shuffle=True)
    
    # Initialize model and trainer
    model = KreiosModel()
    trainer = KreiosTrainer(model, guide_type=guide_type, lr=lr)
    
    # Train
    trainer.train(dataloader, num_epochs=num_epochs)
    
    # Get summary using first batch
    first_batch = next(iter(dataloader))
    summary = trainer.get_posterior_summary(first_batch)
    
    return model, trainer, summary


if __name__ == "__main__":
    from dataloader import create_dataloader
    from pipeline import DatasetBuilder
    from gtf import TranscriptCollector
    from vcf import VariantCollector
    from reader import RibotishReader, RibotishORFReader

    # =====================
    # Data paths
    # =====================
    GTF_PATH = "/Users/danilo/Research/Tools/kreios/examples/ref/gencode.v48.annotation.gtf"
    VCF_PATH = "/Users/danilo/Research/Tools/kreios/examples/vcf/SRR20649716_GSM6395076.deepvariant.phased.vep.vcf.gz"
    RIBOTISH_PROFILE = "/Users/danilo/Research/Tools/kreios/examples/ribotish/SRR15513184_GSM5527709_Fibroblast_40_RiboSeq_Homo_sapiens_RNA-Seq_transprofile.py"
    RIBOTISH_ORF = "/Users/danilo/Research/Tools/kreios/examples/ribotish/SRR15513184_GSM5527709_Fibroblast_40_RiboSeq_Homo_sapiens_RNA-Seq_pred.txt"

    # =====================
    # Training parameters
    # =====================
    batch_size = 16
    num_epochs = 200
    lr = 0.005

    # =====================
    # Load data sources
    # =====================
    print("Loading data sources...")
    tc = TranscriptCollector(GTF_PATH)
    print(f"  Loaded GTF: {len(tc.transcripts)} transcripts")

    vc = VariantCollector(VCF_PATH)
    print(f"  Loaded VCF: {vc}")

    rr = RibotishReader(RIBOTISH_PROFILE)
    print(f"  Loaded ribotish profiles")

    orfs = RibotishORFReader(RIBOTISH_ORF)
    print(f"  Loaded ORFs: {orfs}")

    # =====================
    # Build dataset
    # =====================
    print("\nBuilding dataset...")
    builder = DatasetBuilder(
        variant_collector=vc,
        transcript_collector=tc,
        ribotish_reader=rr,
        orf_reader=orfs,
    )
    print(f"  {builder}")

    transcripts = builder.to_dataset_format()
    print(f"  Generated {len(transcripts)} dataset entries")

    if len(transcripts) == 0:
        print("No data to train on. Exiting.")
        exit(1)

    # =====================
    # Train model
    # =====================
    print("\nTraining model...")
    pyro.set_rng_seed(42)

    dataloader = create_dataloader(transcripts, batch_size=batch_size, shuffle=True)

    model = KreiosModel()
    trainer = KreiosTrainer(model, guide_type="normal", lr=lr)
    trainer.train(dataloader, num_epochs=num_epochs, log_every=20)

    # =====================
    # Evaluate
    # =====================
    eval_batch = next(iter(dataloader))
    summary = trainer.get_posterior_summary(eval_batch, num_samples=1000)

    print("\n" + "=" * 70)
    print("Posterior Summary:")
    print("=" * 70)
    print(f"{'Parameter':<20} {'Mean':>12} {'Std':>10} {'Q05':>10} {'Q95':>10}")
    print("-" * 70)

    scalar_params = ["beta0", "alpha_ramp", "tau_ramp", "delta_abund", "phi"]
    for param in scalar_params:
        if param in summary:
            est_mean = summary[param]["mean"].item()
            est_std = summary[param]["std"].item()
            q05 = summary[param]["q05"].item()
            q95 = summary[param]["q95"].item()
            print(f"{param:<20} {est_mean:>12.3f} {est_std:>10.3f} {q05:>10.3f} {q95:>10.3f}")

    print("-" * 70)
    print(f"\nbase_alpha_vec:")
    print(f"  Mean: {summary['base_alpha_vec']['mean'].squeeze().numpy()}")
    print(f"  Std:  {summary['base_alpha_vec']['std'].squeeze().numpy()}")

    print(f"\ndelta_shape_vec:")
    print(f"  Mean: {summary['delta_shape_vec']['mean'].squeeze().numpy()}")
    print(f"  Std:  {summary['delta_shape_vec']['std'].squeeze().numpy()}")

    # =====================
    # Plot
    # =====================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(trainer.losses)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("ELBO Loss")
    axes[0].set_title("Training Loss (per step)")

    axes[1].plot(trainer.epoch_losses)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Avg ELBO Loss")
    axes[1].set_title("Training Loss (per epoch)")

    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150, bbox_inches="tight")
    print("\nSaved loss plot to training_loss.png")
    plt.show()