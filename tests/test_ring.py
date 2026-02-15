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
    build_pyr_pyr_weights_compte,
    build_pv_pyr_weights,
    compute_stimulus_current,
    simulate_ring,
    population_vector_decode,
    decode_bump_center,
    estimate_bump_width,
    compute_bump_metrics,
    angular_distance,
)


class TestRingParams:
    """Test RingParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = RingParams()
        assert params.n_nodes == 64
        assert params.w_pyr_pyr_inter == 18.55
        assert params.sigma_pyr_deg == 30.0

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
        # 0 degrees -> node 0
        assert params.angle_to_node(0) == 0
        # 180 degrees -> node 32
        assert params.angle_to_node(180) == 32
        # 360 degrees wraps to node 0
        assert params.angle_to_node(360) == 0


class TestConnectivity:
    """Test connectivity matrix construction."""

    def test_pyr_weights_diagonal_zero(self):
        """Self-connections should be zero (handled by local w_ee)."""
        params = RingParams(n_nodes=8)
        W = build_pyr_pyr_weights(params)
        assert np.allclose(np.diag(W), 0.0)

    def test_pyr_weights_symmetric(self):
        """Weight matrix should be symmetric for isotropic connectivity."""
        params = RingParams(n_nodes=32)
        W = build_pyr_pyr_weights(params)
        assert np.allclose(W, W.T)

    def test_pyr_weights_peak_at_neighbors(self):
        """Weights should peak at adjacent nodes and decay with distance."""
        params = RingParams(n_nodes=64, sigma_pyr_deg=30.0)
        W = build_pyr_pyr_weights(params)
        # Check node 0: neighbors (1 and 63) should have highest weights
        row = W[0]
        assert row[1] > row[16]  # Adjacent > distant
        assert row[63] > row[32]  # Adjacent > opposite

    def test_pv_global_uniform(self):
        """Uniform PV weights should be equal for all off-diagonal elements."""
        params = RingParams(n_nodes=8, pv_global_type="uniform", w_pv_global=1.0)
        W = build_pv_pyr_weights(params)
        # All off-diagonal should be equal
        expected = 1.0 / 7  # 1/(n-1)
        for i in range(8):
            for j in range(8):
                if i != j:
                    assert W[i, j] == pytest.approx(expected, rel=1e-5)
                else:
                    assert W[i, j] == 0.0

    def test_connectivity_compute_inputs(self):
        """Test inter-node input computation."""
        params = RingParams(n_nodes=8, w_pyr_pyr_inter=1.0, w_pv_global=0.5)
        conn = RingConnectivity.from_params(params)

        # All PYR firing at rate 1
        r_pyr = np.ones(8)
        r_pv = np.ones(8)

        I_pyr_inter, I_pv_inter = conn.compute_inter_node_inputs(r_pyr, r_pv)

        # Should have non-zero inter-node inputs
        assert np.all(I_pyr_inter > 0)
        assert np.all(I_pv_inter > 0)


class TestStimulus:
    """Test stimulus generation."""

    def test_stimulus_zero_outside_window(self):
        """Stimulus should be zero before onset and after offset."""
        stim = RingStimulus(
            center_deg=180, amplitude=5.0, onset_ms=500, duration_ms=200
        )
        params = RingParams(n_nodes=32)
        angles = params.node_angles_rad

        # Before onset
        assert np.all(compute_stimulus_current(stim, angles, 0) == 0)
        assert np.all(compute_stimulus_current(stim, angles, 499) == 0)
        # After offset
        assert np.all(compute_stimulus_current(stim, angles, 700) == 0)

    def test_stimulus_nonzero_during_window(self):
        """Stimulus should be non-zero during the window."""
        stim = RingStimulus(
            center_deg=180, amplitude=5.0, onset_ms=500, duration_ms=200
        )
        params = RingParams(n_nodes=32)
        angles = params.node_angles_rad

        # During window
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
        node_90deg = 64 // 4  # 90 degrees = node 16
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
        local = CircuitParams()
        ring = RingParams(n_nodes=16)
        stim = RingStimulus(
            center_deg=180, amplitude=5.0, onset_ms=100, duration_ms=100
        )

        result = simulate_ring(local, ring, T_ms=500, dt_ms=1.0, stimuli=[stim])

        assert result.r.shape == (501, 16, 4)
        assert np.all(result.r >= 0)  # Firing rates non-negative

    def test_simulation_without_stimulus(self):
        """Simulation should run without stimulus."""
        local = CircuitParams()
        ring = RingParams(n_nodes=8)

        result = simulate_ring(local, ring, T_ms=100, dt_ms=1.0)

        assert result.r.shape == (101, 8, 4)
        assert result.stim_angle_deg == 0.0

    def test_simulation_result_properties(self):
        """Test RingSimulationResult properties."""
        local = CircuitParams()
        ring = RingParams(n_nodes=32)
        stim = RingStimulus(center_deg=90, amplitude=5.0, onset_ms=50, duration_ms=50)

        result = simulate_ring(local, ring, T_ms=200, dt_ms=1.0, stimuli=[stim])

        assert result.n_nodes == 32
        assert result.n_steps == 201
        assert result.stim_node == 8  # 90 degrees = node 8 for n=32
        assert result.get_pyr_activity().shape == (201, 32)


class TestAnalysis:
    """Test analysis functions."""

    def test_decode_uniform_activity(self):
        """Uniform activity should have low decoding confidence."""
        params = RingParams(n_nodes=64)
        uniform = np.ones(64)
        center, amp = population_vector_decode(uniform, params.node_angles_rad)

        assert amp < 0.1  # Low confidence for uniform

    def test_decode_peaked_activity(self):
        """Peaked activity should decode correctly."""
        params = RingParams(n_nodes=64)
        angles = params.node_angles_rad

        # Create Gaussian bump at 90 degrees
        target_rad = np.pi / 2
        activity = np.exp(-angular_distance(angles, target_rad) ** 2 / (2 * 0.5**2))

        center, amp = population_vector_decode(activity, angles)

        decoded_deg = center * 180 / np.pi
        assert abs(decoded_deg - 90) < 5.0  # Within 5 degrees
        assert amp > 0.5  # High confidence

    def test_bump_width_estimation(self):
        """Test bump width estimation."""
        params = RingParams(n_nodes=64)
        angles = params.node_angles_rad

        # Create narrow bump
        target_rad = np.pi
        sigma = 0.3  # radians
        activity = np.exp(-angular_distance(angles, target_rad) ** 2 / (2 * sigma**2))

        width = estimate_bump_width(activity, angles, target_rad)

        # Width should be roughly proportional to sigma
        expected_width_deg = sigma * 180 / np.pi
        assert width > 0
        assert width < 90  # Should be reasonably narrow


class TestIntegration:
    """Integration tests."""

    def test_bump_forms_with_stimulus(self):
        """Activity should increase at stimulus location."""
        local = CircuitParams()
        ring = RingParams(n_nodes=32, w_pyr_pyr_inter=2.0, sigma_pyr_deg=45.0)
        stim = RingStimulus(
            center_deg=180, amplitude=10.0, onset_ms=100, duration_ms=200
        )

        result = simulate_ring(
            local, ring, T_ms=500, dt_ms=0.5, stimuli=[stim], seed=42
        )

        # Activity at stim node should be higher than opposite node at end of stimulus
        stim_node = result.stim_node
        opposite_node = (stim_node + 16) % 32

        # Get activity at end of stimulus (use recorded time array)
        t_end_stim = 300  # ms
        idx = int(np.argmin(np.abs(result.t_ms - t_end_stim)))

        pyr_stim = result.r[idx, stim_node, 0]
        pyr_opposite = result.r[idx, opposite_node, 0]

        assert pyr_stim > pyr_opposite

    def test_working_memory_protocol_integration(self):
        """Test full working memory protocol."""
        local = CircuitParams()
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
            local,
            ring,
            T_ms=protocol.total_duration_ms,
            dt_ms=1.0,
            stimuli=stimuli,
        )

        # Should complete without error
        assert result.t_ms[-1] >= protocol.total_duration_ms - 1


class TestCompteConnectivity:
    """Test Compte et al. (2000) connectivity profile."""

    def test_diagonal_zero(self):
        """Self-connections should be zero."""
        params = RingParams(n_nodes=64, pyr_profile_type="compte",
                            J_plus=1.5, sigma_pyr_deg=30.0)
        W = build_pyr_pyr_weights(params)
        assert np.allclose(np.diag(W), 0.0)

    def test_symmetric(self):
        """Weight matrix should be symmetric."""
        params = RingParams(n_nodes=64, pyr_profile_type="compte",
                            J_plus=1.5, sigma_pyr_deg=30.0)
        W = build_pyr_pyr_weights(params)
        assert np.allclose(W, W.T)

    def test_row_sum(self):
        """Off-diagonal row sums should equal 1/N."""
        params = RingParams(n_nodes=64, pyr_profile_type="compte",
                            J_plus=1.5, sigma_pyr_deg=30.0)
        W = build_pyr_pyr_weights(params)
        n = params.n_nodes
        row_sums = W.sum(axis=1)
        assert np.allclose(row_sums, 1.0 / n, rtol=1e-10)

    def test_local_excitation_peak(self):
        """Nearby weights should be larger than distant weights."""
        params = RingParams(n_nodes=64, pyr_profile_type="compte",
                            J_plus=1.8, sigma_pyr_deg=30.0)
        W = build_pyr_pyr_weights(params)
        assert W[0, 1] > W[0, 32]

    def test_surround_inhibition(self):
        """With large J+, distant weights should be negative."""
        params = RingParams(n_nodes=64, pyr_profile_type="compte",
                            J_plus=2.5, sigma_pyr_deg=20.0)
        W = build_pyr_pyr_weights(params)
        assert W[0, 32] < 0

    def test_scaling_with_n(self):
        """Total input W@r should be 1/N for uniform r=1 at any N."""
        for n in [32, 64, 128]:
            params = RingParams(n_nodes=n, pyr_profile_type="compte",
                                J_plus=1.5, sigma_pyr_deg=30.0)
            W = build_pyr_pyr_weights(params)
            total_input = W @ np.ones(n)
            assert total_input[0] == pytest.approx(1.0 / n, rel=1e-10)

    def test_dispatch_gaussian_default(self):
        """Default pyr_profile_type='gaussian' should produce all non-negative weights."""
        params = RingParams(n_nodes=32, pyr_profile_type="gaussian")
        W = build_pyr_pyr_weights(params)
        assert np.all(W >= 0)

    def test_compte_direct_call(self):
        """Direct call to build_pyr_pyr_weights_compte should work."""
        params = RingParams(n_nodes=32, pyr_profile_type="compte",
                            J_plus=1.5, sigma_pyr_deg=30.0)
        W = build_pyr_pyr_weights_compte(params)
        assert W.shape == (32, 32)
        assert np.allclose(W.sum(axis=1), 1.0 / 32, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
