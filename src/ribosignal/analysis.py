import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.stats import circmean

class HybridSpectralAnalyzer:
    """
    Hybrid spectral analysis combining single-taper Fourier with multitaper estimation
    for robust frame quantification in ribosome profiling data.
    """

    def __init__(self, 
                 window_size=150, 
                 step_size=15,
                 multitaper_window=300,
                 nw=3, 
                 k=5,
                 phase_threshold=np.pi/6):
        """
        Parameters:
        -----------
        window_size : int
            Size of sliding window for initial Fourier analysis
        step_size : int
            Step size for sliding window
        multitaper_window : int
            Window size for multitaper refinement
        nw : float
            Time-bandwidth product for DPSS tapers
        k : int
            Number of tapers to use
        phase_threshold : float
            Threshold for phase consistency check (radians)
        """
        self.window_size = window_size
        self.step_size = step_size
        self.multitaper_window = multitaper_window
        self.nw = nw
        self.k = k
        self.phase_threshold = phase_threshold
        self.target_freq = 1/3  # 3-nucleotide periodicity
        
    def sliding_window_fourier(self, psite_coverage):
        """
        Initial frame mapping using single-taper Fourier analysis.
        
        Parameters:
        -----------
        psite_coverage : array
            P-site coverage vector along genome
            
        Returns:
        --------
        positions : array
            Center positions of windows
        phases : array
            Phase estimates at target frequency
        powers : array
            Power fraction at target frequency
        """
        n_windows = (len(psite_coverage) - self.window_size) // self.step_size + 1
        positions = np.zeros(n_windows)
        phases = np.zeros(n_windows)
        powers = np.zeros(n_windows)
        # Hann taper
        hann = signal.windows.hann(self.window_size)
        for i in range(n_windows):
            start = i * self.step_size
            end = start + self.window_size
            window_data = psite_coverage[start:end]
            # Apply Hann taper
            tapered = window_data * hann
            # Compute FFT
            fft = np.fft.fft(tapered)
            freqs = np.fft.fftfreq(self.window_size)
            # Find closest frequency to 1/3
            freq_idx = np.argmin(np.abs(freqs - self.target_freq))
            # Extract phase and power
            complex_coeff = fft[freq_idx]
            phases[i] = np.angle(complex_coeff)
            # Power fraction (normalized power at target frequency)
            total_power = np.sum(np.abs(fft)**2)
            powers[i] = np.abs(complex_coeff)**2 / total_power if total_power > 0 else 0
            positions[i] = start + self.window_size // 2
            
        return positions, phases, powers
    
    def multitaper_estimation(self, psite_coverage, center_pos):
        """
        Multitaper refinement for a specific genomic region.
        
        Parameters:
        -----------
        psite_coverage : array
            P-site coverage vector
        center_pos : int
            Center position for extraction
            
        Returns:
        --------
        phase : float
            Variance-reduced phase estimate
        power : float
            Variance-reduced power estimate
        variance : float
            Estimated variance across tapers
        """
        # Extract window centered on position
        half_window = self.multitaper_window // 2
        start = max(0, center_pos - half_window)
        end = min(len(psite_coverage), center_pos + half_window)
        
        # Handle edge cases
        if end - start < self.multitaper_window:
            # Pad if necessary
            window_data = np.zeros(self.multitaper_window)
            offset = (self.multitaper_window - (end - start)) // 2
            window_data[offset:offset + (end - start)] = psite_coverage[start:end]
        else:
            window_data = psite_coverage[start:end]
        
        # Detrend by mean subtraction
        window_data = window_data - np.mean(window_data)
        # Compute DPSS tapers
        tapers, _ = signal.windows.dpss(self.multitaper_window, self.nw, self.k, 
                                        return_ratios=True)
        # Compute Fourier coefficients for each taper
        complex_coeffs = []
        for taper in tapers:
            tapered = window_data * taper
            fft = np.fft.fft(tapered)
            freqs = np.fft.fftfreq(self.multitaper_window)
            freq_idx = np.argmin(np.abs(freqs - self.target_freq))
            complex_coeffs.append(fft[freq_idx])
        complex_coeffs = np.array(complex_coeffs)
        # Average complex coefficients across tapers
        mean_coeff = np.mean(complex_coeffs)
        phase = np.angle(mean_coeff)
        power = np.abs(mean_coeff)**2
        # Estimate variance (circular variance for phase)
        phases_per_taper = np.angle(complex_coeffs)
        variance = np.var(phases_per_taper)
        
        return phase, power, variance
    
    def identify_refinement_sites(self, positions, phases, powers, 
                                   min_coverage=10, min_power=0.1):
        """
        Identify sites requiring multitaper refinement.
        
        Parameters:
        -----------
        positions : array
            Window center positions
        phases : array
            Phase estimates from single-taper analysis
        powers : array
            Power estimates from single-taper analysis
        min_coverage : float
            Minimum coverage threshold
        min_power : float
            Minimum power fraction threshold
            
        Returns:
        --------
        refinement_sites : list
            List of positions requiring refinement
        """
        refinement_sites = []
        
        # Find abrupt phase transitions
        phase_diffs = np.abs(np.diff(phases))
        # Handle circular wrapping
        phase_diffs = np.minimum(phase_diffs, 2*np.pi - phase_diffs)
        
        for i in range(len(phase_diffs)):
            if phase_diffs[i] > np.pi/3 and powers[i] > min_power:
                refinement_sites.append(positions[i])
        
        return refinement_sites
    
    def hybrid_analysis(self, psite_coverage):
        """
        Complete hybrid spectral analysis pipeline.
        
        Parameters:
        -----------
        psite_coverage : array
            P-site coverage vector along genome
            
        Returns:
        --------
        result : dict
            Dictionary containing:
            - positions: window center positions
            - phases: hybrid phase estimates
            - powers: hybrid power estimates
            - confidence: confidence flags
            - variances: estimated variances
        """
        # Stage 1: Initial frame mapping
        positions, st_phases, st_powers = self.sliding_window_fourier(psite_coverage)
        
        # Identify sites for refinement
        refinement_sites = self.identify_refinement_sites(positions, st_phases, st_powers)
        
        # Stage 2: Multitaper refinement
        mt_phases = {}
        mt_powers = {}
        mt_variances = {}
        
        for site in refinement_sites:
            phase, power, variance = self.multitaper_estimation(psite_coverage, site)
            mt_phases[site] = phase
            mt_powers[site] = power
            mt_variances[site] = variance
        
        # Interpolate multitaper estimates back to original grid
        if len(mt_phases) > 1:
            mt_sites = np.array(list(mt_phases.keys()))
            mt_phase_vals = np.array(list(mt_phases.values()))
            mt_power_vals = np.array(list(mt_powers.values()))
            
            # Interpolation for phases (handle circular nature)
            phase_interp = interp1d(mt_sites, mt_phase_vals, 
                                   kind='linear', fill_value='extrapolate')
            power_interp = interp1d(mt_sites, mt_power_vals,
                                   kind='linear', fill_value='extrapolate')
            
            mt_phases_grid = phase_interp(positions)
            mt_powers_grid = power_interp(positions)
        else:
            mt_phases_grid = np.zeros_like(positions)
            mt_powers_grid = np.zeros_like(positions)
        
        # Stage 3: Integration of estimates
        hybrid_phases = np.zeros_like(st_phases)
        hybrid_powers = np.zeros_like(st_powers)
        confidence = np.ones_like(st_phases, dtype=bool)
        
        for i in range(len(positions)):
            if positions[i] in mt_phases:
                # Check consistency
                phase_diff = np.abs(st_phases[i] - mt_phases[positions[i]])
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                
                if phase_diff < self.phase_threshold:
                    # Consistent - compute circular mean
                    hybrid_phases[i] = circmean([st_phases[i], mt_phases[positions[i]]])
                    # Weighted average for power (inverse variance weights)
                    var = mt_variances[positions[i]]
                    if var > 0:
                        w_mt = 1 / var
                        w_st = 1.0  # Assume unit variance for single-taper
                        hybrid_powers[i] = (w_st * st_powers[i] + w_mt * mt_powers[positions[i]]) / (w_st + w_mt)
                    else:
                        hybrid_powers[i] = (st_powers[i] + mt_powers[positions[i]]) / 2
                else:
                    # Inconsistent - flag low confidence
                    confidence[i] = False
                    hybrid_phases[i] = st_phases[i]
                    hybrid_powers[i] = st_powers[i]
            else:
                # No multitaper estimate available
                hybrid_phases[i] = st_phases[i]
                hybrid_powers[i] = st_powers[i]
        
        return {
            'positions': positions,
            'phases': hybrid_phases,
            'powers': hybrid_powers,
            'confidence': confidence,
            'st_phases': st_phases,
            'mt_phases': mt_phases
        }
    
    def permutation_test(self, psite_coverage_1, psite_coverage_2, n_permutations=1000):
        """
        Statistical inference using permutation-based circular distance test.
        
        Parameters:
        -----------
        psite_coverage_1, psite_coverage_2 : array
            P-site coverage vectors for two conditions
        n_permutations : int
            Number of permutations for null distribution
            
        Returns:
        --------
        p_values : array
            P-values at each position
        """
        # Analyze both conditions
        result_1 = self.hybrid_analysis(psite_coverage_1)
        result_2 = self.hybrid_analysis(psite_coverage_2)
        
        # Compute observed circular distance
        phase_diff = np.abs(result_1['phases'] - result_2['phases'])
        phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)
        
        # Generate null distribution
        combined = np.vstack([psite_coverage_1, psite_coverage_2])
        null_distances = []
        
        for _ in range(n_permutations):
            # Permute condition labels
            np.random.shuffle(combined)
            perm_1, perm_2 = combined[0], combined[1]
            
            res_1 = self.hybrid_analysis(perm_1)
            res_2 = self.hybrid_analysis(perm_2)
            
            perm_diff = np.abs(res_1['phases'] - res_2['phases'])
            perm_diff = np.minimum(perm_diff, 2*np.pi - perm_diff)
            null_distances.append(perm_diff)
        
        null_distances = np.array(null_distances)
        
        # Compute p-values
        p_values = np.mean(null_distances >= phase_diff[np.newaxis, :], axis=0)
        
        return p_values