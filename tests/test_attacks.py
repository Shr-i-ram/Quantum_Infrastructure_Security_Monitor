"""Unit tests for V0.3 attack models, lifecycle, profiles, and engine."""

import numpy as np
import pytest

from qism.attacks.base import (
    AttackEngine,
    AttackPhase,
    AttackScenario,
    AttackState,
    AttackType,
    OnsetProfile,
    evaluate_profile,
)
from qism.attacks.intercept_resend import (
    ChannelLossAttack,
    DetectorManipulationAttack,
    InterceptResendAttack,
    PNSAttack,
    SourceManipulationAttack,
    TimingAttack,
)
from qism.random import derive_attack_seed, make_attack_rng, make_rng


def test_onset_profiles():
    """Verify mathematical properties of onset ramp profiles."""
    # At t=0, all should be 0.0
    for prof in (OnsetProfile.SMOOTHSTEP, OnsetProfile.LINEAR, OnsetProfile.INSTANT, OnsetProfile.EXPONENTIAL):
        val0 = evaluate_profile(0.0, prof)
        assert val0 == 0.0

    # For instant, u > 0 jumps to 1.0 immediately
    assert evaluate_profile(0.001, OnsetProfile.INSTANT) == 1.0

    # At t=1, all should be 1.0
    for prof in OnsetProfile:
        val1 = evaluate_profile(1.0, prof)
        assert pytest.approx(val1, abs=1e-5) == 1.0

    # Monotonicity between 0 and 1
    xs = np.linspace(0.0, 1.0, 50)
    for prof in (OnsetProfile.SMOOTHSTEP, OnsetProfile.LINEAR, OnsetProfile.EXPONENTIAL):
        ys = [evaluate_profile(x, prof) for x in xs]
        for y1, y2 in zip(ys[:-1], ys[1:]):
            assert y2 >= y1 - 1e-9


def test_attack_lifecycle_and_phases():
    """Verify 5-phase ground truth lifecycle."""
    sc = AttackScenario(
        scenario_id="test_sc",
        attack_type=AttackType.INTERCEPT_RESEND,
        severity=1.0,
        start_pulse=1000,
        end_pulse=5000,
        onset_pulses=500,
        termination_pulses=500,
        onset_profile=OnsetProfile.SMOOTHSTEP,
        recovery_profile=OnsetProfile.SMOOTHSTEP,
    )
    atk = InterceptResendAttack(sc)

    # 1. Normal (before start)
    atk.update(500)
    assert atk.state == AttackState.INACTIVE
    assert atk.phase == AttackPhase.NORMAL
    assert atk.activation_level == 0.0

    # 2. Onset phase
    atk.update(1250)
    assert atk.state == AttackState.ACTIVE
    assert atk.phase == AttackPhase.ONSET
    assert 0.0 < atk.activation_level < 1.0

    # 3. Active plateau
    atk.update(3000)
    assert atk.state == AttackState.ACTIVE
    assert atk.phase == AttackPhase.ACTIVE
    assert atk.activation_level == 1.0

    # 4. Termination phase (pulse between 5000 and 5500)
    atk.update(5250)
    assert atk.state == AttackState.TERMINATING
    assert atk.phase == AttackPhase.TERMINATION
    assert 0.0 < atk.activation_level < 1.0

    # 5. Recovery settling phase
    atk.update(5600)
    assert atk.state == AttackState.COMPLETED
    assert atk.phase == AttackPhase.RECOVERY
    assert atk.activation_level == 0.0

    # 6. Settled back to Normal
    atk.update(6500)
    assert atk.state == AttackState.COMPLETED
    assert atk.phase == AttackPhase.NORMAL
    assert atk.activation_level == 0.0


def test_attack_constructors():
    """Verify that all 6 attacks can be constructed and take parameters."""
    sc_ir = AttackScenario("ir", AttackType.INTERCEPT_RESEND, 0.8, 100, 200, parameters={"fraction": 0.5})
    atk_ir = InterceptResendAttack(sc_ir)
    assert atk_ir.fraction == 0.5

    sc_loss = AttackScenario("loss", AttackType.CHANNEL_LOSS, 0.5, 100, 200, parameters={"delta_L_max_db": 4.5})
    atk_loss = ChannelLossAttack(sc_loss)
    assert atk_loss.delta_L_max_db == 4.5

    sc_src = AttackScenario("src", AttackType.SOURCE_MANIPULATION, 0.6, 100, 200, parameters={"delta_mu": 0.2, "direction": "decrease"})
    atk_src = SourceManipulationAttack(sc_src)
    assert atk_src.delta_mu == 0.2
    assert atk_src.direction == "decrease"

    sc_det = AttackScenario("det", AttackType.DETECTOR_MANIPULATION, 0.7, 100, 200, parameters={"mode": "dark_count", "delta_dark_hz": 2000.0})
    atk_det = DetectorManipulationAttack(sc_det)
    assert atk_det.mode == "dark_count"
    assert atk_det.delta_dark_hz == 2000.0

    sc_time = AttackScenario("time", AttackType.TIMING, 0.4, 100, 200, parameters={"delta_t_max_s": 150e-12})
    atk_time = TimingAttack(sc_time)
    assert atk_time.delta_t_max_s == 150e-12

    sc_pns = AttackScenario("pns", AttackType.PNS, 0.9, 100, 200, parameters={"single_photon_suppression": 0.7})
    atk_pns = PNSAttack(sc_pns)
    assert atk_pns.single_photon_suppression == 0.7


def test_attack_engine_multi_attack():
    """Verify multi-attack episode management in AttackEngine."""
    class MockConfig:
        enabled = True
        type = "none"
        strength = 0.0
        start_pulse = 0
        end_pulse = None
        params = {}
        scenarios = [
            AttackScenario("ep1", AttackType.INTERCEPT_RESEND, 0.8, 1000, 3000, onset_pulses=200, termination_pulses=200),
            AttackScenario("ep2", AttackType.CHANNEL_LOSS, 0.5, 4000, 6000, onset_pulses=200, termination_pulses=200),
        ]

    engine = AttackEngine(MockConfig())
    assert len(engine.attacks) == 2

    # Pulse 500: Normal
    engine.update(500)
    assert len(engine.active_attacks) == 0
    assert engine.get_phase() == AttackPhase.NORMAL
    assert engine.get_dominant_attack() is None

    # Pulse 2000: Episode 1 active
    engine.update(2000)
    assert len(engine.active_attacks) == 1
    assert engine.get_phase() == AttackPhase.ACTIVE
    dom = engine.get_dominant_attack()
    assert dom is not None
    assert dom.scenario.scenario_id == "ep1"

    # Pulse 3500: Between episodes -> Normal
    engine.update(3500)
    assert len(engine.active_attacks) == 0
    assert engine.get_phase() == AttackPhase.NORMAL

    # Pulse 5000: Episode 2 active
    engine.update(5000)
    assert len(engine.active_attacks) == 1
    dom = engine.get_dominant_attack()
    assert dom is not None
    assert dom.scenario.scenario_id == "ep2"


def test_deterministic_attack_rng_isolation():
    """Verify that attack RNG is independent from simulation chunk seeds."""
    seed = 42
    chunk = 5

    # Derive seeds
    attack_seed = derive_attack_seed(seed, chunk)
    atk_rng1 = make_attack_rng(seed, chunk)
    atk_rng2 = make_attack_rng(seed, chunk)

    # Identical draws
    draws1 = atk_rng1.random(10)
    draws2 = atk_rng2.random(10)
    np.testing.assert_array_equal(draws1, draws2)

    # Different chunks produce different seeds
    assert derive_attack_seed(seed, 5) != derive_attack_seed(seed, 6)
