"""
Stimulus protocols for the ring attractor network.

This module contains dataclasses and functions for defining and computing
spatially and temporally localized stimuli on the ring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .connectivity import angular_distance


@dataclass(frozen=True)
class RingStimulus:
    """
    Configuration for a stimulus on the ring.

    The stimulus is a spatially localized current injection to PYR neurons,
    with a Gaussian spatial profile centered at a specific angular location.

    Attributes:
        center_deg: Stimulus center (degrees, 0-360)
        amplitude: Peak current amplitude
        sigma_deg: Spatial width (degrees)
        onset_ms: When stimulus turns on (ms)
        duration_ms: How long stimulus lasts (ms)
    """

    # Location
    center_deg: float  # Stimulus center (degrees, 0-360)

    # Spatial profile
    amplitude: float  # Peak current amplitude
    sigma_deg: float = 20.0  # Spatial width (degrees)

    # Temporal profile
    onset_ms: float = 500.0  # When stimulus turns on
    duration_ms: float = 250.0  # How long stimulus lasts

    @property
    def offset_ms(self) -> float:
        """Time when stimulus ends."""
        return self.onset_ms + self.duration_ms

    @property
    def center_rad(self) -> float:
        """Stimulus center in radians."""
        return self.center_deg * np.pi / 180.0

    @property
    def sigma_rad(self) -> float:
        """Stimulus width in radians."""
        return self.sigma_deg * np.pi / 180.0


def compute_stimulus_current(
    stimulus: RingStimulus,
    node_angles_rad: np.ndarray,
    t_ms: float,
) -> np.ndarray:
    """
    Compute stimulus current at each node for a given time.

    Parameters:
        stimulus: RingStimulus configuration
        node_angles_rad: Angular positions of nodes (radians)
        t_ms: Current time (ms)

    Returns:
        I_stim: Stimulus current at each node, shape (n_nodes,)
    """
    n_nodes = len(node_angles_rad)

    # Check if within temporal window
    if t_ms < stimulus.onset_ms or t_ms >= stimulus.offset_ms:
        return np.zeros(n_nodes)

    # Compute spatial profile (Gaussian centered at stimulus location)
    dist = angular_distance(node_angles_rad, stimulus.center_rad)
    spatial = np.exp(-dist**2 / (2 * stimulus.sigma_rad**2))

    return stimulus.amplitude * spatial


@dataclass(frozen=True)
class WorkingMemoryProtocol:
    """
    Protocol for working memory task with cue, delay, and optional distractor.

    The task consists of:
    1. Pre-cue baseline period
    2. Cue presentation (brief stimulus at target location)
    3. Delay period (memory retention without stimulus)
    4. Post-delay period (for analysis)

    Optionally, a distractor can be presented during the delay period.

    Attributes:
        cue_location_deg: Where to present cue (0-360 degrees)
        cue_amplitude: Cue stimulus current amplitude
        cue_duration_ms: Duration of cue stimulus
        cue_sigma_deg: Spatial width of cue
        pre_cue_ms: Baseline period before cue
        delay_ms: Delay period (memory retention)
        post_delay_ms: Period after delay for analysis
        distractor_location_deg: Optional distractor location (None = no distractor)
        distractor_amplitude: Distractor stimulus amplitude
        distractor_onset_ms: When distractor appears (relative to simulation start)
        distractor_duration_ms: Duration of distractor
    """

    # Cue stimulus
    cue_location_deg: float  # Where to present cue (0-360)
    cue_amplitude: float = 5.0
    cue_duration_ms: float = 250.0
    cue_sigma_deg: float = 20.0

    # Timing
    pre_cue_ms: float = 500.0  # Baseline before cue
    delay_ms: float = 3000.0  # Delay period (memory retention)
    post_delay_ms: float = 500.0  # After delay (for analysis)

    # Optional distractor
    distractor_location_deg: Optional[float] = None
    distractor_amplitude: float = 3.0
    distractor_onset_ms: float = 1500.0  # During delay (relative to sim start)
    distractor_duration_ms: float = 200.0

    @property
    def total_duration_ms(self) -> float:
        """Total simulation duration."""
        return self.pre_cue_ms + self.cue_duration_ms + self.delay_ms + self.post_delay_ms

    @property
    def cue_onset_ms(self) -> float:
        """Cue stimulus onset time."""
        return self.pre_cue_ms

    @property
    def cue_offset_ms(self) -> float:
        """Cue stimulus offset time."""
        return self.pre_cue_ms + self.cue_duration_ms

    @property
    def delay_onset_ms(self) -> float:
        """When delay period starts."""
        return self.pre_cue_ms + self.cue_duration_ms

    @property
    def delay_offset_ms(self) -> float:
        """When delay period ends."""
        return self.delay_onset_ms + self.delay_ms

    def get_stimuli(self) -> list[RingStimulus]:
        """Generate list of stimuli for this protocol."""
        stimuli = [
            RingStimulus(
                center_deg=self.cue_location_deg,
                amplitude=self.cue_amplitude,
                sigma_deg=self.cue_sigma_deg,
                onset_ms=self.cue_onset_ms,
                duration_ms=self.cue_duration_ms,
            )
        ]

        if self.distractor_location_deg is not None:
            stimuli.append(
                RingStimulus(
                    center_deg=self.distractor_location_deg,
                    amplitude=self.distractor_amplitude,
                    sigma_deg=self.cue_sigma_deg,
                    onset_ms=self.distractor_onset_ms,
                    duration_ms=self.distractor_duration_ms,
                )
            )

        return stimuli
