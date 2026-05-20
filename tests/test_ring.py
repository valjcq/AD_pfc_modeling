"""
Tests for the circuit_model.ring subpackage.

Run with: pytest tests/test_ring.py -v
"""

import numpy as np
import pytest

from circuit_model import CircuitParams
from circuit_model.ring import (
    RingParams,
    RingConnectivity,
    RingStimulus,
    WorkingMemoryProtocol,
    build_pyr_pyr_weights,
    build_pv_pyr_weights,
    build_som_pyr_weights,
    compute_stimulus_current,
    simulate_ring,
    population_vector_decode,
    decode_bump_center,
    estimate_bump_width,
    compute_bump_metrics,
    angular_distance,
)


# Shared fixtures
_LOCAL = CircuitParams()


class TestRingParams:
    """Test RingParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = RingParams()
        assert params.n_nodes == 64
        assert params.sigma_pyr_deg == 15.0
        assert params.sigma_som_deg == 15.0

    def test_angular_spacing(self):
        """Test angular spacing calculation."""
        params = RingParams(n_nodes=64)
        assert params.angular_spacing_deg == pytest.approx(360 / 64)
        assert params.angular_spacing_rad == pytest.approx(2 * np.pi / 64)

    def test_node_angles(self):
        """Test node angle arrays."""
        params = RingParams(n_nodes=8)
        expected_deg = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        assert np.allclose(params.node_angles_deg, expected_deg)

    def test_angle_to_node_conversion(self):
        """Test angle to node index conversion."""
        params = RingParams(n_nodes=64)
        assert params.angle_to_node(0) == 0
        assert params.angle_to_node(180) == 32
        assert params.angle_to_node(360) == 0

    def test_sigma_som_rad(self):
        """sigma_som_rad should convert sigma_som_deg correctly."""
        params = RingParams(sigma_som_deg=30.0)
        assert params.sigma_som_rad == pytest.approx(30.0 * np.pi / 180.0)


class TestConnectivity:
    """Test connectivity matrix construction."""

    def test_pyr_weights_includes_diagonal(self):
        """Diagonal should be non-zero (unified kernel includes self-weight)."""
        params = RingParams(n_nodes=8)
        W = build_pyr_pyr_weights(params, _LOCAL)
        assert np.all(np.diag(W) > 0), "W_pyr diagonal must be > 0"

    def test_pyr_weights_row_sum(self):
        """Each row of W_pyr should sum to J_NMDA."""
        params = RingParams(n_nodes=16)
        W = build_pyr_pyr_weights(params, _LOCAL)
        assert np.allclose(W.sum(axis=1), _LOCAL.J_NMDA), (
            f"Row sums={W.sum(axis=1)}, expected J_NMDA={_LOCAL.J_NMDA}"
        )

    def test_pyr_weights_symmetric(self):
        """Weight matrix should be symmetric for isotropic connectivity."""
        params = RingParams(n_nodes=32)
        W = build_pyr_pyr_weights(params, _LOCAL)
        assert np.allclose(W, W.T)

    def test_pyr_weights_peak_at_self(self):
        """Diagonal (self-weight) should be the maximum in each row."""
        params = RingParams(n_nodes=64, sigma_pyr_deg=15.0)
        W = build_pyr_pyr_weights(params, _LOCAL)
        for i in range(64):
            assert W[i, i] == pytest.approx(W[i].max(), rel=1e-6), (
                f"Row {i}: diagonal {W[i,i]:.6f} not max (max={W[i].max():.6f})"
            )

    def test_pv_weights_uniform_including_diagonal(self):
        """PV weights should be equal for ALL entries including diagonal."""
        params = RingParams(n_nodes=8)
        W = build_pv_pyr_weights(params, _LOCAL)
        expected = _LOCAL.w_pe / 8
        assert np.allclose(W, expected), f"Expected uniform {expected}, got {W}"

    def test_pv_weights_row_sum(self):
        """Each row of W_pv should sum to w_pe."""
        params = RingParams(n_nodes=16)
        W = build_pv_pyr_weights(params, _LOCAL)
        assert np.allclose(W.sum(axis=1), _LOCAL.w_pe)

    def test_som_weights_zero_diagonal(self):
        """SOM kernel should have zero diagonal (purely lateral, no self-inhibition)."""
        params = RingParams(n_nodes=8)
        W = build_som_pyr_weights(params, _LOCAL)
        assert np.allclose(np.diag(W), 0.0)

    def test_som_weights_row_sum(self):
        """Each row of W_som (excluding diagonal) should sum to w_se."""
        params = RingParams(n_nodes=16)
        W = build_som_pyr_weights(params, _LOCAL)
        assert np.allclose(W.sum(axis=1), _LOCAL.w_se)

    def test_som_weights_symmetric(self):
        """SOM lateral kernel should be symmetric."""
        params = RingParams(n_nodes=32)
        W = build_som_pyr_weights(params, _LOCAL)
        assert np.allclose(W, W.T)

    def test_connectivity_matrix_products(self):
        """Direct matrix products should give correct shapes and signs."""
        params = RingParams(n_nodes=8)
        conn = RingConnectivity.from_params(params, _LOCAL)

        S_pyr = np.ones(8) * 0.5   # NMDA gating variable
        r_pv  = np.ones(8) * 5.0   # PV firing rates
        r_som = np.ones(8) * 2.0   # SOM firing rates

        I_pyr_nmda = conn.W_pyr_pyr @ S_pyr   # should be positive
        I_pv_denom = conn.W_pv_pyr  @ r_pv    # should be positive
        I_som_lat  = conn.W_som_pyr @ r_som   # should be positive

        assert np.all(I_pyr_nmda > 0)
        assert np.all(I_pv_denom > 0)
        assert np.all(I_som_lat > 0)

        # At homogeneous fixed point the row-sum property gives exact scalars
        assert np.allclose(I_pyr_nmda, _LOCAL.J_NMDA * 0.5)
        assert np.allclose(I_pv_denom, _LOCAL.w_pe * 5.0)
        assert np.allclose(I_som_lat,  _LOCAL.w_se * 2.0)


class TestStimulus:
    """Test stimulus generation."""

    def test_stimulus_zero_outside_window(self):
        """Stimulus should be zero before onset and after offset."""
        stim = RingStimulus(
            center_deg=180, amplitude=5.0, onset_ms=500, duration_ms=200
        )
        params = RingParams(n_nodes=32)
        angles = params.node_angles_rad

        assert np.all(compute_stimulus_current(stim, angles, 0) == 0)
        assert np.all(compute_stimulus_current(stim, angles, 499) == 0)
        assert np.all(compute_stimulus_current(stim, angles, 700) == 0)

    def test_stimulus_nonzero_during_window(self):
        """Stimulus should be non-zero during the window."""
        stim = RingStimulus(
            center_deg=180, amplitude=5.0, onset_ms=500, duration_ms=200
        )
        params = RingParams(n_nodes=32)
        angles = params.node_angles_rad
        I = compute_stimulus_current(stim, angles, 600)
        assert np.any(I > 0)

    def test_stimulus_peaked_at_center(self):
        """Stimulus should be maximal at specified location."""
        stim = RingStimulus(
            center_deg=90, amplitude=5.0, sigma_deg=20.0, onset_ms=0, duration_ms=100
        )
        params = RingParams(n_nodes=64)
        angles = params.node_angles_rad
        I = compute_stimulus_current(stim, angles, 50)
        node_90deg = 64 // 4
        assert np.argmax(I) == node_90deg

    def test_working_memory_protocol(self):
        """Test WorkingMemoryProtocol creates correct stimuli."""
        protocol = WorkingMemoryProtocol(
            cue_location_deg=180.0,
            cue_amplitude=5.0,
            cue_duration_ms=250.0,
            pre_cue_ms=500.0,
            delay_ms=3000.0,
        )

        assert protocol.total_duration_ms == 4250.0
        assert protocol.cue_onset_ms == 500.0
        assert protocol.delay_onset_ms == 750.0

        stimuli = protocol.get_stimuli()
        assert len(stimuli) == 1
        assert stimuli[0].center_deg == 180.0


class TestSimulation:
    """Test ring simulation."""

    def test_simulation_runs(self):
        """Basic smoke test: simulation should run without errors."""
        ring = RingParams(n_nodes=16)
        stim = RingStimulus(center_deg=180, amplitude=5.0, onset_ms=100, duration_ms=100)
        result = simulate_ring(_LOCAL, ring, T_ms=500, dt_ms=1.0, stimuli=[stim])
        assert result.r.shape == (501, 16, 4)
        assert np.all(result.r >= 0)

    def test_simulation_without_stimulus(self):
        """Simulation should run without stimulus."""
        ring = RingParams(n_nodes=8)
        result = simulate_ring(_LOCAL, ring, T_ms=100, dt_ms=1.0)
        assert result.r.shape == (101, 8, 4)
        assert result.stim_angle_deg == 0.0

    def test_simulation_result_properties(self):
        """Test RingSimulationResult properties."""
        ring = RingParams(n_nodes=32)
        stim = RingStimulus(center_deg=90, amplitude=5.0, onset_ms=50, duration_ms=50)
        result = simulate_ring(_LOCAL, ring, T_ms=200, dt_ms=1.0, stimuli=[stim])
        assert result.n_nodes == 32
        assert result.n_steps == 201
        assert result.stim_node == 8  # 90 degrees = node 8 for n=32
        assert result.get_pyr_activity().shape == (201, 32)

    def test_interneuron_ceiling(self):
        """Interneuron rates should never exceed their soft ceiling."""
        from circuit_model.constants import R_MAX_PV, R_MAX_SOM, R_MAX_VIP
        ring = RingParams(n_nodes=8)
        result = simulate_ring(_LOCAL, ring, T_ms=200, dt_ms=0.1, noise_type="none")
        r = result.r
        assert r[:, :, 1].max() < R_MAX_SOM * 1.01, "SOM exceeded ceiling"
        assert r[:, :, 2].max() < R_MAX_PV  * 1.01, "PV exceeded ceiling"
        assert r[:, :, 3].max() < R_MAX_VIP * 1.01, "VIP exceeded ceiling"


class TestAnalysis:
    """Test analysis functions."""

    def test_decode_uniform_activity(self):
        """Uniform activity should have low decoding confidence."""
        params = RingParams(n_nodes=64)
        uniform = np.ones(64)
        center, amp = population_vector_decode(uniform, params.node_angles_rad)
        assert amp < 0.1

    def test_decode_peaked_activity(self):
        """Peaked activity should decode correctly."""
        params = RingParams(n_nodes=64)
        angles = params.node_angles_rad
        target_rad = np.pi / 2
        activity = np.exp(-angular_distance(angles, target_rad) ** 2 / (2 * 0.5**2))
        center, amp = population_vector_decode(activity, angles)
        decoded_deg = center * 180 / np.pi
        assert abs(decoded_deg - 90) < 5.0
        assert amp > 0.5

    def test_bump_width_estimation(self):
        """Test bump width estimation."""
        params = RingParams(n_nodes=64)
        angles = params.node_angles_rad
        target_rad = np.pi
        sigma = 0.3
        activity = np.exp(-angular_distance(angles, target_rad) ** 2 / (2 * sigma**2))
        width = estimate_bump_width(activity, angles, target_rad)
        assert width > 0
        assert width < 90


class TestIntegration:
    """Integration tests."""

    def test_bump_forms_with_stimulus(self):
        """Activity should increase at stimulus location."""
        local = CircuitParams(I0_pyr=0.3)
        ring = RingParams(n_nodes=32, sigma_pyr_deg=15.0)
        stim = RingStimulus(center_deg=180, amplitude=5.0, onset_ms=100, duration_ms=200)

        result = simulate_ring(local, ring, T_ms=500, dt_ms=0.5, stimuli=[stim], seed=42)

        stim_node = result.stim_node
        opposite_node = (stim_node + 16) % 32

        t_check = 150
        idx = int(np.argmin(np.abs(result.t_ms - t_check)))
        pyr_stim = result.r[idx, stim_node, 0]
        pyr_opposite = result.r[idx, opposite_node, 0]
        assert pyr_stim > pyr_opposite

    def test_working_memory_protocol_integration(self):
        """Test full working memory protocol."""
        ring = RingParams(n_nodes=32)
        protocol = WorkingMemoryProtocol(
            cue_location_deg=90.0,
            cue_amplitude=8.0,
            pre_cue_ms=100.0,
            cue_duration_ms=100.0,
            delay_ms=200.0,
            post_delay_ms=100.0,
        )

        stimuli = protocol.get_stimuli()
        result = simulate_ring(
            _LOCAL,
            ring,
            T_ms=protocol.total_duration_ms,
            dt_ms=1.0,
            stimuli=stimuli,
        )

        assert result.t_ms[-1] >= protocol.total_duration_ms - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
