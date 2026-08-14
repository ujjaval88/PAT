from pat_sim.plant.transport_delay import TransportDelay

DELAY_S = 0.5e-3
DT_S = 50e-6


def test_delay_sample_count():
    delay = TransportDelay(delay_s=DELAY_S, dt_s=DT_S)
    assert delay.n_samples == 10


def test_impulse_emerges_after_n_samples():
    delay = TransportDelay(delay_s=DELAY_S, dt_s=DT_S, initial_value=0.0)
    outputs = [delay.step(1.0 if k == 0 else 0.0) for k in range(15)]
    assert outputs[:10] == [0.0] * 10
    assert outputs[10] == 1.0
    assert outputs[11:] == [0.0] * 4


def test_zero_delay_is_passthrough():
    delay = TransportDelay(delay_s=0.0, dt_s=DT_S)
    assert delay.step(3.14) == 3.14
