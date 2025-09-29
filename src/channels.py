import torch
from abc import ABC, abstractmethod
from .region import Region, VCFRegion
import numpy as np


class ChannelEncoder(ABC):
    """
    Abstract base class for creating a specific tensor channel from a list of reads.
    """
    def __init__(self, name: str, dtype: torch.dtype, fill_value):
        self.name = name
        self.dtype = dtype
        self.fill_value = fill_value

    def create_empty_tensor(self, shape: tuple, device: str) -> torch.Tensor:
        """
        Instantiate the empty tensor
        """
        return torch.full(shape, fill_value=self.fill_value, dtype=self.dtype, device=device)

    @abstractmethod
    def _get_read_data(self, masked_read) -> np.ndarray:
        """
        Abstract method to get the specific data array from a MaskedRead.
        This is replaced within each subclass according to the channel's logic
        """
        pass

    def __call__(self, reads_to_process: list, start_row: int, tensor_shape: tuple, region_start: int, device: str, **kwargs) -> torch.Tensor:
        final_tensor = self.create_empty_tensor(tensor_shape, device)
        region_length = tensor_shape[1]

        for i, m_read in enumerate(reads_to_process):
            # use a stable row index if provided by the fetcher
            row_idx = getattr(m_read, 'row_index', i)
            dest_row = start_row + row_idx

            # ensure we don't write out of bounds
            if dest_row < 0 or dest_row >= tensor_shape[0]:
                continue

            tensor_start = max(0, m_read.start - region_start)
            tensor_end = min(region_length, m_read.end - region_start)
            if tensor_start >= tensor_end:
                # *still* write fill (optional) OR continue while keeping row mapping stable.
                # we'll keep the row (already filled with fill_value) and just continue.
                continue

            read_slice_start = max(0, region_start - m_read.start)
            read_slice_end = read_slice_start + (tensor_end - tensor_start)

            padded_array = self._get_read_data(m_read)
            data_to_place = padded_array[read_slice_start:read_slice_end]

            final_tensor[dest_row, tensor_start:tensor_end] = torch.from_numpy(data_to_place).to(self.dtype)

        return final_tensor


class SequenceEncoder(ChannelEncoder):
    def __init__(self, base_color_stride=20, offset_t_c=5, offset_a_g=4, default_color=0):
        super().__init__(
            name='sequence_color',
            dtype=torch.uint8,
            fill_value=default_color
        )
        
        # Create a lookup table (an array of 256 elements for all possible ASCII values)
        self.base_to_color_map = np.full(256, default_color, dtype=np.uint8)
        
        # Populate the lookup table for the bases we care about
        color_map = {
            'A': offset_a_g + base_color_stride * 3, 'a': offset_a_g + base_color_stride * 3,
            'G': offset_a_g + base_color_stride * 2, 'g': offset_a_g + base_color_stride * 2,
            'T': offset_t_c + base_color_stride * 1, 't': offset_t_c + base_color_stride * 1,
            'C': offset_t_c + base_color_stride * 0, 'c': offset_t_c + base_color_stride * 0,
            'N': default_color, 'n': default_color,
            '-': default_color
        }
        
        for base, value in color_map.items():
            if not (0 <= value <= 255):
                raise ValueError(f"Color for base '{base}' is {value}, which is outside the [0, 255] range.")
            self.base_to_color_map[ord(base)] = value

    def _get_read_data(self, masked_read) -> np.ndarray:
        """
        Gets the sequence data and uses the pre-computed lookup table for a very
        fast conversion from character to "color".
        """
        sequence_chars = masked_read.get_padded_array('sequence')
        # View the character array as an array of their underlying integer (ASCII/UTF-8) values
        sequence_integers = sequence_chars.view(np.uint32)
        
        # Use the integer values to directly index our lookup table. This is extremely fast.
        return self.base_to_color_map[sequence_integers]

class BaseQualityEncoder(ChannelEncoder):
    """Encodes the base quality score channel."""
    def __init__(self, padding_value=np.nan):
        super().__init__(
            name='base_qualities', 
            dtype=torch.float32, 
            fill_value=padding_value
        )
        # base above this value will be capped 
        self.base_quality_cap = 16

    def _get_read_data(self, masked_read) -> np.ndarray:
        base_qualities = masked_read.get_padded_array('base_qualities')
        # Cap the base qualities to avoid extreme values
        np.clip(base_qualities, a_min=0, a_max=self.base_quality_cap, out=base_qualities)
        return base_qualities

class IntronicEncoder(ChannelEncoder):
    """Encodes whether the read base is in an intronic region."""
    def __init__(self, intron_value=1, non_intron_value=0, padding_value=-1):
        super().__init__(
            name='intronic', 
            dtype=torch.int8, 
            fill_value=padding_value
        )
        self.intron_value = intron_value
        self.non_intron_value = non_intron_value

    def _get_read_data(self, masked_read) -> np.ndarray:
        read_sequence = masked_read.get_padded_array('sequence')
        
        # Start with an array filled with the non-intron value.
        intron_array = np.full(read_sequence.shape, self.non_intron_value, dtype=np.int8)
        
        # Use boolean masks to set values for specific conditions in a vectorized way.
        intron_array[read_sequence == 'N'] = self.intron_value
        intron_array[read_sequence == '-'] = self.fill_value
        
        return intron_array

class ReferenceMatchingEncoder(ChannelEncoder):
    """Encodes whether the read base matches the reference base."""
    def __init__(self, padding_value=-1):
        super().__init__(
            name='ref_match', 
            dtype=torch.int8, 
            fill_value=padding_value
        )
        self.reference_matching = 11
        self.reference_mismatching = 16
        self.reference_sequence = None
        self.region_start = None

    def __call__(self, reads_to_process: list, start_row: int, tensor_shape: tuple, region_start: int, device: str, **kwargs) -> torch.Tensor:
        ref_seq = kwargs.get('reference_sequence')
        if ref_seq is None:
            raise ValueError("ReferenceMatchingEncoder requires 'reference_sequence'.")
        
        # Convert reference to a NumPy array of characters once per call.
        self.reference_sequence = np.fromiter(ref_seq.upper(), dtype='c')
        self.region_start = region_start
        
        return super().__call__(
            reads_to_process, start_row, tensor_shape, region_start, device, **kwargs
        )

    def _get_read_data(self, masked_read) -> np.ndarray:
        read_sequence = masked_read.get_padded_array('sequence').astype('c')
        match_array = np.full(read_sequence.shape, self.fill_value, dtype=np.int8)
        
        # Create a boolean mask for valid (non-padding) bases in the read
        valid_bases_mask = (read_sequence != b'-')
        if not np.any(valid_bases_mask):
            return match_array

        # Generate an array of genomic positions for all bases in the read
        read_positions = masked_read.start + np.arange(len(read_sequence))
        
        # Convert genomic positions to indices into our reference sequence
        ref_indices = read_positions - self.region_start
        
        # Create a mask for positions that are within the bounds of the reference sequence
        in_bounds_mask = (ref_indices >= 0) & (ref_indices < len(self.reference_sequence))
        
        # Combine masks: we only care about bases that are valid AND in bounds
        final_mask = valid_bases_mask & in_bounds_mask
        
        # Get the relevant indices for both the read and the reference
        read_indices_to_compare = np.where(final_mask)[0]
        ref_indices_to_compare = ref_indices[final_mask]
        
        # Extract the bases to compare using our calculated indices
        read_bases = read_sequence[read_indices_to_compare]
        ref_bases = self.reference_sequence[ref_indices_to_compare]
        
        # Perform the comparison in a single vectorized operation
        match_mask = (read_bases == ref_bases)
        
        # Use the boolean match_mask to assign matching and mismatching values
        match_array[read_indices_to_compare[match_mask]] = self.reference_matching
        match_array[read_indices_to_compare[~match_mask]] = self.reference_mismatching
                    
        return match_array

class ReadStrandEncoder(ChannelEncoder):
    """
    Encodes reads based on whether they map to a strand or another.
    REFACTORED to use the base class __call__ method for consistency.
    """
    def __init__(self, padding_value=-1):
        super().__init__(
            name='read_strand',
            dtype=torch.int8, 
            fill_value=padding_value
        )
        self.positive_strand_color = 12
        self.negative_strand_color = 15

    def _get_read_data(self, masked_read) -> np.ndarray:
        """
        Generates a per-base array representing the read's strand.
        This array respects padding for deletions.
        """
        # Determine the single color value for this read
        color = self.positive_strand_color if not masked_read.read.is_reverse else self.negative_strand_color
        
        # Get the padded sequence to know the correct shape and where deletions are
        read_sequence = masked_read.get_padded_array('sequence')
        
        # Create an array of the correct shape, filled with the strand color
        strand_array = np.full(read_sequence.shape, color, dtype=np.int8)
        
        # IMPORTANT: Apply padding where the original sequence has deletions ('-')
        # This ensures structural consistency with other channels.
        strand_array[read_sequence == '-'] = self.fill_value
        
        return strand_array


class ReadsSupportingAlleleEncoder(ChannelEncoder):
    """
    Encodes reads based on whether they support a given alternate allele.
    REFACTORED to use the base class __call__ method for consistency.
    """
    def __init__(self, padding_value=-1):
        super().__init__(
            name='allele_support',
            dtype=torch.int8, 
            fill_value=padding_value
        )
        self.allele_supporting_read = 2
        self.allele_unsupporting_read = 10

    def _get_read_data(self, masked_read, **kwargs) -> np.ndarray:
        """
        Generates a per-base array indicating if the read supports the alt allele.
        This array respects padding for deletions.
        """
        variant = kwargs.get('variant')
        if not isinstance(variant, VCFRegion):
            raise ValueError("ReadsSupportingAlleleEncoder requires a 'variant' of type VCFRegion")

        variant_pos = variant.start
        alt_allele = variant.alt.upper()

        supports_alt = False
        try:
            ref_positions = masked_read.read.get_reference_positions(full_length=True)
            q_indices = [qidx for qidx, rpos in enumerate(ref_positions) if rpos == variant_pos]
            if q_indices:
                query_index = q_indices[0]
                read_seq = masked_read.read.get_forward_sequence()
                if read_seq and query_index < len(read_seq):
                    read_base = read_seq[query_index].upper()
                    if read_base == alt_allele:
                        supports_alt = True
        except Exception:
            supports_alt = False

        color = self.allele_supporting_read if supports_alt else self.allele_unsupporting_read
        
        # Get the padded sequence to know the correct shape and where deletions are
        read_sequence = masked_read.get_padded_array('sequence')
        
        # Create an array of the correct shape, filled with the support color
        support_array = np.full(read_sequence.shape, color, dtype=np.int8)
        
        # Apply padding where the original sequence has deletions ('-')
        support_array[read_sequence == '-'] = self.fill_value
        
        return support_array

    def __call__(self, reads_to_process: list, start_row: int, tensor_shape: tuple, region_start: int, device: str, **kwargs) -> torch.Tensor:
        """
        Overwrite of the main __class__
        """
        final_tensor = self.create_empty_tensor(tensor_shape, device)
        region_length = tensor_shape[1]

        for i, m_read in enumerate(reads_to_process):
            row_idx = getattr(m_read, 'row_index', i)
            dest_row = start_row + row_idx
            if dest_row < 0 or dest_row >= tensor_shape[0]:
                continue
            tensor_start = max(0, m_read.start - region_start)
            tensor_end = min(region_length, m_read.end - region_start)
            if tensor_start >= tensor_end:
                continue
            read_slice_start = max(0, region_start - m_read.start)
            read_slice_end = read_slice_start + (tensor_end - tensor_start)
            padded_array = self._get_read_data(m_read, **kwargs)
            data_to_place = padded_array[read_slice_start:read_slice_end]

            final_tensor[dest_row, tensor_start:tensor_end] = torch.from_numpy(data_to_place).to(self.dtype)

        return final_tensor

if __name__ == "__main__":
    from plotter import TensorVisualizer
    from model import SimpleGenomicCNN
    from genome import retrieve_sequence

    region = 'chr5:140561643-140561644'
    variant = VCFRegion(
        region, ref='A', alt='T'
    )
    simplecnn = SimpleGenomicCNN()
    # extend the region by 50 bp 
    region = 'chr5:140561593-140561694'
    region_object = Region(region)
    genome = '/CTGlab/db/GRCh38_GIABv3_no_alt_analysis_set_maskedGRC_decoys_MAP2K3_KMT2C_KCNJ18.fasta'
    reference_sequence = retrieve_sequence(genome, region_object)
    print(f"Length of reference_sequence is {len(reference_sequence)}")
    all_encoders = [
        BaseQualityEncoder(),
        SequenceEncoder(),
        IntronicEncoder(),
        ReferenceMatchingEncoder(),
        ReadsSupportingAlleleEncoder(),
        ReadStrandEncoder()

    ]
    bam = "/CTGlab/projects/ribo/goyal_2023/RNAseq/SRR20649710_GSM6395082.markdup.sorted.bam"
    fetcher = fetch.ReadFetcher(bam, max_reads=256, channel_encoders=all_encoders, genome=genome, device='cpu')
# 5. Call fetch_tensors and pass the variant-specific data.
    chromosome, positions = region.split(':')
    start, end = map(int, positions.split('-'))
    tensors = fetcher.fetch_tensors(chromosome, start, end, variant)
    single_tensor = torch.nan_to_num(
        torch.stack(list(tensors.values()), dim=0))
    print(simplecnn(single_tensor.unsqueeze(0)))
    fig = TensorVisualizer().plot(tensors, chromosome, start, end)
    fig.savefig("test_encoders_1.png")
