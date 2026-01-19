import scipy.stats as stats
import numpy as np
from typing import Optional


def decompose_into_frames(profile: np.ndarray, start_pos: int, start_codon_pos: int) -> np.ndarray:
    """
    Decompose a 1D profile into frame counts based on start_codon position.

    Args:
        profile: 1D array of read counts at nucleotide resolution
        start_pos: The transcript-relative position where this profile starts
        start_codon_pos: The transcript-relative position of the start codon

    Returns:
        Array of shape (3,) with total counts per frame (frame 0, 1, 2)
    """
    frame_counts = np.zeros(3, dtype=int)
    for i, count in enumerate(profile):
        # Absolute position in transcript coordinates
        abs_pos = start_pos + i
        # Frame relative to start codon
        frame = (abs_pos - start_codon_pos) % 3
        frame_counts[frame] += count
    return frame_counts


def get_start_codon_from_features(features: list) -> Optional[int]:
    """
    Extract start_codon position from features list.

    Args:
        features: List of (feature_type, rel_start, rel_end) tuples

    Returns:
        Relative start position of start_codon, or None if not found
    """
    for feat_type, rel_start, rel_end in features:
        if feat_type == 'start_codon':
            return rel_start
    return None





def chi2_test(before_counts: np.ndarray, after_counts: np.ndarray):
    """
    Perform chi-squared test on frame distribution before vs after.

    Args:
        before_counts: Array of shape (3,) with frame counts before variant
        after_counts: Array of shape (3,) with frame counts after variant

    Returns:
        Tuple of (chi2_statistic, p_value)
    """
    contingency_table = np.array([before_counts, after_counts])

    # Check for sufficient counts to run chi2
    if contingency_table.sum() == 0:
        return np.nan, np.nan

    # Check if any row/column is all zeros (would cause issues)
    if (contingency_table.sum(axis=0) == 0).any() or (contingency_table.sum(axis=1) == 0).any():
        return np.nan, np.nan

    stat, pval, dof, expected = stats.chi2_contingency(contingency_table)
    return stat, pval


def analyze_entry(entry: dict) -> dict:
    """
    Analyze a single entry from DatasetBuilder.

    Decomposes profiles into frames based on start_codon position and
    performs chi-squared test comparing frame distribution before/after variant.

    Args:
        entry: Dict with keys:
            - 'pre_profile': 1D array of counts before variant
            - 'after_profile': 1D array of counts after variant
            - 'variant_rel_pos': variant position relative to transcript 5' end
            - 'strand': '+' or '-'
            - 'features': list of (feature_type, rel_start, rel_end) tuples

    Returns:
        Dict with:
            - 'before_frames': array of shape (3,) with frame counts before variant
            - 'after_frames': array of shape (3,) with frame counts after variant
            - 'chi2_stat': chi-squared statistic
            - 'p_value': p-value from chi-squared test
            - 'start_codon_pos': position of start codon used
            - 'valid': whether analysis was successful
    """
    # Extract start codon position from features
    start_codon_pos = get_start_codon_from_features(entry['features'])

    if start_codon_pos is None:
        return {
            'before_frames': np.zeros(3, dtype=int),
            'after_frames': np.zeros(3, dtype=int),
            'chi2_stat': np.nan,
            'p_value': np.nan,
            'start_codon_pos': None,
            'valid': False
        }

    variant_rel_pos = entry['variant_rel_pos']
    pre_profile = entry['pre_profile']
    after_profile = entry['after_profile']

    # Decompose pre_profile: starts at position 0
    before_frames = decompose_into_frames(pre_profile, start_pos=0, start_codon_pos=start_codon_pos)

    # Decompose after_profile: starts at variant_rel_pos
    after_frames = decompose_into_frames(after_profile, start_pos=variant_rel_pos, start_codon_pos=start_codon_pos)

    # Perform chi-squared test
    chi2_stat, p_value = chi2_test(before_frames, after_frames)

    return {
        'before_frames': before_frames,
        'after_frames': after_frames,
        'chi2_stat': chi2_stat,
        'p_value': p_value,
        'start_codon_pos': start_codon_pos,
        'valid': not np.isnan(chi2_stat)
    }


def analyze_dataset(profiles: dict) -> list[dict]:
    """
    Analyze all entries from DatasetBuilder.profiles.

    Args:
        profiles: Dict mapping tid -> list of entry dicts

    Returns:
        List of result dicts with tid and variant_id included
    """
    results = []
    for tid, entries in profiles.items():
        for entry in entries:
            result = analyze_entry(entry)
            result['tid'] = tid
            result['variant_id'] = entry.get('variant_id')
            result['variant_pos'] = entry.get('variant_pos')
            result['variant_rel_pos'] = entry.get('variant_rel_pos')
            result['strand'] = entry.get('strand')
            results.append(result)
    return results


if __name__ == '__main__':
    # Test with mock data
    mock_entry = {
        'variant_id': 'chr1:1000:A>T',
        'pre_profile': np.array([0, 0, 5, 10, 2, 8, 15, 3, 6, 12, 1, 9]),
        'after_profile': np.array([3, 8, 2, 4, 7, 1, 5, 9, 3, 2, 6, 4]),
        'variant_rel_pos': 12,
        'strand': '+',
        'features': [('start_codon', 0, 3), ('CDS', 0, 24)]
    }

    result = analyze_entry(mock_entry)
    print("Mock entry analysis:")
    print(f"  Variant: {mock_entry['variant_id']}")
    print(f"  Before frames (f0, f1, f2): {result['before_frames']}")
    print(f"  After frames (f0, f1, f2): {result['after_frames']}")
    print(f"  Chi2 statistic: {result['chi2_stat']:.4f}")
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  Valid: {result['valid']}")
