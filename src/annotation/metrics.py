"""
This module provides classes and data structures for calculating and storing
variant coverage metrics and statistical power.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Union
from scipy.stats import binom
import numpy as np


    
DEFAULT_FPR = 5e-7
DEFAULT_ERROR_RATE = 1e-3


@dataclass
class CoverageMetrics:
    """Holds coverage and allele metrics for a variant."""
    ac: Dict[str, int]
    dp: int
    bqs: Dict[str, Union[int, None]]
    mqs: Dict[str, Union[int, None]]
    all_bqs: Dict[str, List[int]] = field(default_factory=dict)
    all_mqs: Dict[str, List[int]] = field(default_factory=dict)


class PowerCalculator:
    """
    Calculates the statistical power to detect a variant, simplified for healthy,
    diploid data where tumor purity and ploidy are not relevant.
    """

    def __init__(
        self,
        expected_vaf: float,
        fpr: float = DEFAULT_FPR,
        error_rate: float = DEFAULT_ERROR_RATE,
    ):
        """
        Initializes the calculator with fixed parameters.

        Args:
            expected_vaf: The expected variant allele frequency for a heterozygous variant.
                          Defaults to 0.5.
            fpr: The acceptable false positive rate.
            error_rate: The assumed sequencing error rate per base.
        """
        self.expected_vaf = expected_vaf
        self.fpr = fpr
        self.error_rate = error_rate

    def _calculate_p(self, m: int, n: int) -> float:
        """
        Calculates the probability of observing m or more identical non-reference
        reads due to sequencing error. Based on Carter, 2012.

        Args:
            m: Number of observed reads supporting a mutation.
            n: Total coverage (dp).
        """
        if m < 1:
            return 1.0
        # The probability of a specific error is error_rate / 3 (for the 3 other bases)
        prob_specific_error = self.error_rate / 3
        return 1.0 - binom.cdf(k=m - 1, n=n, p=prob_specific_error)

    def _calculate_k(self, dp: int) -> int:
        """
        Finds the minimum number of supporting reads (k) required to meet the
        specified false positive rate (FPR).

        Args:
            dp: Total coverage.
        """
        k = 1
        while self._calculate_p(m=k, n=dp) > self.fpr:
            k += 1
        return k

    def _calculate_d(self, k: int, n: int) -> float:
        """
        Calculates the 'd' factor from Carter, 2012, used for power adjustment.

        Args:
            k: The minimum number of supporting reads.
            n: Total coverage (dp).
        """
        p_k = self._calculate_p(m=k, n=n)
        p_k_minus_1 = self._calculate_p(m=k - 1, n=n)

        denominator = p_k_minus_1 - p_k
        if denominator == 0:
            return 0.0  # Avoid division by zero

        return (self.fpr - p_k) / denominator

    def calculate_absolute_power(self, dp: int) -> tuple[float, int]:
        """
        Calculates the statistical power as defined in Carter, 2012.

        This is the probability of observing at least the minimum required
        number of reads (k) for a variant to be considered real, given
        the total coverage and expected VAF.

        Args:
            dp: Total coverage at the site.

        Returns:
            A tuple containing:
                - The calculated power (float).
                - The minimum required allele count (k) (int).
        """
        if dp == 0:
            return 0.0, 0
        k = self._calculate_k(dp=dp)
        f = self.expected_vaf
        # Power is P(X >= k)
        power = 1.0 - binom.cdf(k=k - 1, n=dp, p=f)
        # Carter, 2012 includes an adjustment term
        d_factor = self._calculate_d(k=k, n=dp)
        power += d_factor * binom.pmf(k=k, n=dp, p=f)
        return round(power, 5), k

    def calculate_power(self, dp: int, ac: int) -> float:
        """
        Calculates the binomial probability of observing 'ac' or MORE supporting reads,
        given a total coverage 'dp' and the expected VAF.

        This represents the power to detect the variant at the given allele count.

        Args:
            dp: Total depth of coverage at the site.
            ac: Allele count (number of reads supporting the variant).

        Returns:
            The rounded binomial probability (detection power).
        """
        if dp == 0:
            return 0.0
        # Power = P(X >= ac) = 1 - P(X <= ac - 1)
        power = 1.0 - binom.cdf(k=ac - 1, n=dp, p=self.expected_vaf)
        return round(power, 5)
