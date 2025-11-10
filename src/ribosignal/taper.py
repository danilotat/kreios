import numpy as np 
import scipy.signal as signal
import logging
import scipy.stats as stats

# DESIGN
# ------
# Goal: quantify 3-nt translational periodicity along coverage using a hybrid approach:
# 1) Slide a Hann-tapered Fourier window to get quick per-position phase/power at f = 1/3 cycles/nt.
# 2) Refine via multitaper (DPSS) around positions of interest and use an F-test (Thomson) for significance.
# 3) Integrate estimates by phase-consistency and variance-weighted power when both agree.
# Why this design: single-taper is fast but noisy; multitaper improves bias/variance and offers a test, at higher cost.
# Keeping the fast path enables scanning long transcripts; refinement controls false positives where phase is unstable.
# CHECKLIST
# ---------
# - If you change target frequency (e.g., non-3nt periodicity), update teacher notes and bin selection.
# - If you change window function or normalization, verify power comparability across windows.
# - If you change the F-test threshold or degrees of freedom logic, adjust the integration gate accordingly.
# - If you alter k or nw, ensure k >= 2 for F-test validity and revisit variance weighting.



class HybridSpectralAnalyzer:
    """
    Class to run a hybrid spectral pipeline for frame detection in ribosome profiling.
    The instance stores configuration (window sizes, DPSS params, thresholds) and exposes analysis methods.

    It quantifies 3-nt translational periodicity via a hybrid approach:
    - Slide a Hann-tapered Fourier window to get quick per-position phase/power at \(f=1/3\) cycles/nt.
    - Refine via multitaper (DPSS) around positions of interest and use a Thomson F-test for significance.
    - Integrate estimates by phase-consistency and variance-weighted power when both agree.
    
    Rationale: Single-taper is fast but noisy; multitaper improves bias/variance and provides a test at higher cost.
    The fast path enables scanning long transcripts; refinement curbs false positives where phase is unstable. [web:1]

    Parameters
    ----------
    window_size : int, default 150
        Sliding window length for the single-taper Fourier pass.
    step_size : int, default 15
        Stride between adjacent sliding windows.
    multitaper_window : int, default 300
        Window length for DPSS multitaper refinement.
    nw : float, default 3
        DPSS time-bandwidth product.
    k : int, default 5
        Number of DPSS tapers.
    phase_threshold : float, default np.pi/6
        Maximum circular phase difference allowed when merging single-taper and multitaper estimates.

    Attributes
    ----------
    target_freq : float
        Target frequency in cycles/nt for frame periodicity; defaults to \(1/3\).

    """
    
    def __init__(self, 
                 window_size=150, 
                 step_size=15,
                 multitaper_window=300,
                 nw=3, 
                 k=5,
                 phase_threshold=np.pi/6):
        self.window_size = window_size
        self.step_size = step_size
        self.multitaper_window = multitaper_window
        self.nw = nw
        self.k = k
        self.phase_threshold = phase_threshold
        self.target_freq = 1/3  # 3-nucleotide periodicity

    def _pad_or_skip(self, arr):
        """
        Return a slice/padded vector of length window_size or None if signal is too sparse for meaningful FFT.

        Parameters
        ----------
        arr : array-like
            Input coverage slice.

        Returns
        -------
        ndarray or None
            A vector of length `window_size` (padded or truncated) or None if there are fewer than 6 nonzeros.

        Notes
        -----
        A minimum of 6 nonzero samples avoids spuriously stable phases from near-zero windows. The output length
        is fixed to keep FFT binning consistent across windows.
        """
        # A minimum number of nonzeros avoids spuriously stable phases from near-zero windows; 6 is a pragmatic floor.
        nonzero = np.count_nonzero(arr)
        if nonzero < 6:
            return None  # insufficient data
        # pad short windows to fixed FFT size for consistent binning.
        if len(arr) < self.window_size:
            logging.info(f"Window size {len(arr)} is lower than {self.window_size}. Padding it")
            padded = np.zeros(self.window_size)
            padded[:len(arr)] = arr
            return padded
        # truncate longer slices to the analysis size.
        return arr[:self.window_size]


    def sliding_window_fourier(self, psite_coverage):
        """
        Fast per-window phase/power estimation at the 3-nt frequency using a Hann-tapered FFT.

        Parameters
        ----------
        psite_coverage : array-like
            P-site coverage along the transcript (one sample per nucleotide).

        Returns
        -------
        positions : ndarray
            Window-center positions (ints).
        phases : ndarray
            Phase at the target frequency for each window (radians).
        powers : ndarray
            Power at the target frequency, normalized by total FFT power.

        Notes
        -----
        `np.fft.fftfreq(N)` uses unit sample spacing; with nucleotide indexing, the 3-nt periodicity maps to \(1/3\).
        A Hann taper reduces spectral leakage; normalization by total power aids comparability across windows.
        """
        # ensure at least one full window via right padding so FFT size is stable.
        if len(psite_coverage) < self.window_size:
            psite_coverage = np.pad(psite_coverage, 
                                    (0, self.window_size - len(psite_coverage)))
        # precompute windows and storage.
        n_windows = max(1, (len(psite_coverage) - self.window_size) // self.step_size + 1)
        positions = np.zeros(n_windows)
        phases = np.zeros(n_windows)
        powers = np.zeros(n_windows)
        hann = signal.windows.hann(self.window_size)
        # iterate sliding windows and estimate f=1/3 bin.
        for i in range(n_windows):
            start = i * self.step_size
            end = start + self.window_size
            window_data = psite_coverage[start:end]
            window_data = self._pad_or_skip(window_data)
            if window_data is None:
                # record position but mark unusable estimates; downstream code will ignore or refine as needed.
                phases[i], powers[i] = np.nan, np.nan
                positions[i] = start + self.window_size // 2
                continue


            # lightweight taper to reduce spectral leakage.
            tapered = window_data * hann
            # FFT and pick the nearest bin to the target frequency.
            fft = np.fft.fft(tapered)
            freqs = np.fft.fftfreq(self.window_size)
            freq_idx = np.argmin(np.abs(freqs - self.target_freq))
            # extract complex coefficient, then phase and normalized power.
            complex_coeff = fft[freq_idx]
            total_power = np.sum(np.abs(fft)**2)
            # normalize by total power to make scores comparable across windows with different coverage scales.
            phases[i] = np.angle(complex_coeff)
            powers[i] = np.abs(complex_coeff)**2 / total_power if total_power > 0 else np.nan
            # report geometric center of the window to align with local context for refinement/integration.
            positions[i] = start + self.window_size // 2


        return positions, phases, powers


    def multitaper_estimation(self, psite_coverage, center_pos):
        """
        Multitaper refinement at a given center position with DPSS tapers and an F-test for line component.

        Parameters
        ----------
        psite_coverage : array-like
            P-site coverage along the transcript.
        center_pos : int
            Center position for the refinement window.

        Returns
        -------
        phase : float
            Multitaper phase estimate at \(f=1/3\) (radians).
        power : float
            Squared magnitude of the mean complex coefficient at \(f=1/3\).
        variance : float
            Variance of per-taper phases (radians squared).
        F_value : float
            Thomson multitaper F-statistic.
        p_value : float
            P-value for the F-statistic with DOF \((2, 2k-2)\).
        """
        half_window = self.multitaper_window // 2
        start = max(0, center_pos - half_window)
        end = min(len(psite_coverage), center_pos + half_window)
        window_data_slice = psite_coverage[start:end]
        # skip windows with too few nonzeros; multitaper needs enough energy for stable estimates.
        if np.count_nonzero(window_data_slice) < 6:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        # embed slice into a fixed-size zero-padded buffer to keep DPSS and FFT shapes aligned.
        window_data = np.zeros(self.multitaper_window)
        pad_start = max(0, half_window - center_pos)
        pad_end = pad_start + len(window_data_slice)
        window_data[pad_start:pad_end] = window_data_slice
        # remove DC to reduce leakage into low-frequency bins; improves narrowband estimation.
        window_data = window_data - np.mean(window_data)
        # DPSS (Slepian) tapers maximize energy concentration in a bandwidth ~ nw/N. Using k orthogonal tapers
        # reduces variance of spectral estimates while controlling leakage and bias.
        tapers, _ = signal.windows.dpss(self.multitaper_window, self.nw, self.k, 
                                        return_ratios=True)
        # project each tapered sequence, pick the f = 1/3 bin, collect complex coefficients.
        complex_coeffs = []
        for taper in tapers:
            tapered = window_data * taper
            fft = np.fft.fft(tapered)
            freqs = np.fft.fftfreq(self.multitaper_window)
            freq_idx = np.argmin(np.abs(freqs - self.target_freq))
            complex_coeffs.append(fft[freq_idx])
        complex_coeffs = np.array(complex_coeffs)
        mean_coeff = np.mean(complex_coeffs)
        phase = np.angle(mean_coeff)
        power = np.abs(mean_coeff)**2
        # Thomson’s multitaper F-test assesses a sinusoid (line) embedded in noise using coherently-summed tapers.
        # For k tapers, numerator DOF = 2, denominator DOF = 2k - 2; larger F supports a significant line component.
        # compute F-statistic and p-value with stability checks.
        num = (2 * self.k - 2)
        numer = (2 * np.abs(np.sum(complex_coeffs))**2) / self.k
        denom = np.sum(np.abs(complex_coeffs)**2) - numer / 2
        if denom <= 0:
            F_value, p_value = np.nan, np.nan
        else:
            F_value = (numer / 2) / (denom / num)
            p_value = 1 - stats.f.cdf(F_value, 2, num)
        # phase variability across tapers informs confidence/weighting during integration.
        variance = np.var(np.angle(complex_coeffs))
        # TODO:
        # - consider small-sample corrections or jackknife variance for more robust p-values when coverage is low.
        # - Ensure k >= 2 so denominator DOF (= 2k-2) is positive for a valid F-test.


        return phase, power, variance, F_value, p_value


    def identify_refinement_sites(self, positions, phases, powers, 
                                   min_power=0.1):
        """
        Heuristic to flag windows where single-taper phase shifts abruptly and power is adequate.
        Intended to trigger local multitaper refinement when global refinement is off.

        Parameters
        ----------
        positions : array-like
            Window-center positions.
        phases : array-like
            Single-taper phases (radians).
        powers : array-like
            Single-taper normalized powers.
        min_power : float, default 0.1
            Minimum power required to consider a phase discontinuity.

        Returns
        -------
        list
            Positions where circular phase jumps exceed \(\pi/3\) and power exceeds `min_power`.

        Notes
        -----
        Circular phase difference is computed as \(\min(|\Delta|, 2\pi - |\Delta|)\) to respect phase wrapping.
        """
        # Phase lives on the circle; use the minimal circular difference Δφ = min(|Δ|, 2π - |Δ|) to detect jumps.
        refinement_sites = []
        phase_diffs = np.abs(np.diff(phases))
        phase_diffs = np.minimum(phase_diffs, 2*np.pi - phase_diffs)
        # require both a sizable phase discontinuity and sufficient power to avoid chasing noise ridges.
        for i in range(len(phase_diffs)):
            if phase_diffs[i] > np.pi/3 and powers[i] > min_power:
                refinement_sites.append(positions[i])
        return refinement_sites


    def hybrid_analysis(self, psite_coverage, global_multitaper=True):
        """
        Orchestrate the full pipeline:
        1) Single-taper scan to get initial phase/power across windows.
        2) Multitaper refinement globally or at flagged sites.
        3) Integrate estimates if phases agree and F-test passes; otherwise mark as low-confidence.
        
        Parameters
        ------------
        psite_coverage: array like
            Array of psite_coverage 
        global_multitaper: bool
            Compute if globally or just on significant sites


        Returns
        ------------
        out: dict
            Dictionary containing positions, phases, powers, confidence (bool mask), and mt_results per position.

        Notes
        -----
        Circular mean combines consistent phases, and inverse-variance weighting emphasizes stable multitaper power.
        The default F-test gate uses \(\alpha=0.05\); adjust to tune sensitivity/specificity.
        """
        positions, st_phases, st_powers = self.sliding_window_fourier(psite_coverage)
        # decide refinement strategy.
        if global_multitaper:
            refinement_sites = positions
        else:
            refinement_sites = self.identify_refinement_sites(positions, st_phases, st_powers)
        # run refinement as needed and store per-position statistics.
        mt_results = {}
        for pos in positions:  # iterate all windows for completeness
            if (global_multitaper) or (pos in refinement_sites):
                phase, power, variance, F_value, p_value = self.multitaper_estimation(psite_coverage, int(pos))
                mt_results[pos] = {
                    'phase': phase,
                    'power': power,
                    'variance': variance,
                    'F_value': F_value,
                    'p_value': p_value
                }
        # initialize hybrid outputs with single-taper estimates.
        hybrid_phases = np.copy(st_phases)
        hybrid_powers = np.copy(st_powers)
        confidence = np.ones_like(st_phases, dtype=bool)
        # When phases agree within a small circular threshold, combine:
        # - phase via circular mean; power via variance-weighted average favoring more stable multitaper estimates.
        for i, pos in enumerate(positions):
            if pos not in mt_results:
                continue
            mt = mt_results[pos]
            # require a significant F-test (default α = 0.05) to trust a line at f = 1/3.
            if np.isnan(mt['p_value']) or mt['p_value'] > 0.05:
                confidence[i] = False
                continue
            # circular phase difference for consistency gating.
            phase_diff = np.abs(st_phases[i] - mt['phase'])
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)
            if phase_diff < self.phase_threshold:
                # circular mean for phase; inverse-variance weight for power.
                hybrid_phases[i] = stats.circmean([st_phases[i], mt['phase']])
                var = mt['variance']
                w_mt = 1 / var if var > 0 else 1
                hybrid_powers[i] = (st_powers[i] + w_mt * mt['power']) / (1 + w_mt)
            else:
                # disagreeing phase indicates local nonstationarity or noise; keep ST estimate but flag low confidence.
                confidence[i] = False
        out = {
            'positions': positions,
            'phases': hybrid_phases,
            'powers': hybrid_powers,
            'confidence': confidence,
            'mt_results': mt_results
        }
        return out