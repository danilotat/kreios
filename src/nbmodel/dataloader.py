import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RiboseqDataset(Dataset):
    """Dataset for ribosome profiling data."""
    
    def __init__(self, transcripts: list[dict]):
        """
        Args:
            transcripts: List of dicts, each containing:
                - 'positions': array of codon positions (seq_len,)
                - 'counts': array of shape (seq_len, 3)
                - 'variant_idx': int, position of variant
        """
        self.transcripts = transcripts
    
    def __len__(self):
        return len(self.transcripts)
    
    def __getitem__(self, idx):
        t = self.transcripts[idx]
        return {
            'positions': torch.tensor(t['positions'], dtype=torch.float),
            'counts': torch.tensor(t['counts'], dtype=torch.float),
            'variant_idx': t['variant_idx'],
            'length': len(t['positions'])
        }


def collate_riboseq(batch):
    """
    Collate variable-length sequences with padding.
    
    Returns dict with:
        - positions: (batch_size, max_len)
        - counts: (batch_size, max_len, 3)
        - total_counts: (batch_size, max_len)
        - mask: (batch_size, max_len) - downstream of variant
        - padding_mask: (batch_size, max_len) - 1 = valid, 0 = padding
        - lengths: (batch_size,)
    """
    batch_size = len(batch)
    max_len = max(item['length'] for item in batch)
    
    # Pre-allocate padded tensors
    positions = torch.zeros(batch_size, max_len)
    counts = torch.zeros(batch_size, max_len, 3)
    mask = torch.zeros(batch_size, max_len)
    padding_mask = torch.zeros(batch_size, max_len)
    
    for i, item in enumerate(batch):
        length = item['length']
        variant_idx = item['variant_idx']
        
        positions[i, :length] = item['positions']
        counts[i, :length] = item['counts']
        
        # Variant mask (downstream of variant)
        pos_array = item['positions']
        mask[i, :length] = (pos_array >= variant_idx).float()
        
        # Padding mask (1 = valid position)
        padding_mask[i, :length] = 1.0
    
    return {
        'positions': positions,
        'counts': counts,
        'total_counts': counts.sum(dim=-1),
        'mask': mask,
        'padding_mask': padding_mask,
        'lengths': torch.tensor([item['length'] for item in batch])
    }


def create_dataloader(transcripts, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader for ribosome profiling data.
    
    Args:
        transcripts: List of transcript dicts
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        
    Returns:
        DataLoader instance
    """
    dataset = RiboseqDataset(transcripts)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_riboseq,
        num_workers=num_workers
    )


# =====================
# Convenience functions
# =====================

def transcripts_from_simulator(simulator_outputs: list[dict], variant_idx: int) -> list[dict]:
    """
    Convert simulator outputs to transcript format for dataloader.
    
    Args:
        simulator_outputs: List of dicts from RiboseqSimulator.simulate()
        variant_idx: Variant position (shared across all)
        
    Returns:
        List of transcript dicts ready for RiboseqDataset
    """
    transcripts = []
    for data in simulator_outputs:
        transcripts.append({
            'positions': data['positions'],
            'counts': data['triplet_counts'],
            'variant_idx': variant_idx
        })
    return transcripts


if __name__ == "__main__":
    # Test with simulated data
    from simulator import RiboseqSimulator
    
    n_codons = 100
    variant_pos = 50
    n_transcripts = 50
    
    # Generate variable-length transcripts
    sim_outputs = []
    for i in range(n_transcripts):
        # Variable length between 80 and 120
        length = np.random.randint(80, 121)
        var_pos = length // 2  # Variant at middle
        
        sim = RiboseqSimulator(n_codons=length, variant_pos=var_pos, seed=i)
        data = sim.simulate()
        data['variant_idx'] = var_pos
        sim_outputs.append({
            'positions': data['positions'],
            'counts': data['triplet_counts'],
            'variant_idx': var_pos
        })
    
    # Create dataloader
    loader = create_dataloader(sim_outputs, batch_size=8, shuffle=True)
    
    # Test iteration
    for batch in loader:
        print("Batch shapes:")
        print(f"  positions:    {batch['positions'].shape}")
        print(f"  counts:       {batch['counts'].shape}")
        print(f"  total_counts: {batch['total_counts'].shape}")
        print(f"  mask:         {batch['mask'].shape}")
        print(f"  padding_mask: {batch['padding_mask'].shape}")
        print(f"  lengths:      {batch['lengths']}")
        break