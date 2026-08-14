from pat_sim.simulation.scheduler import ArrivalQueue, PeriodicSchedule


def test_periodic_schedule_fires_at_expected_count():
    schedule = PeriodicSchedule(rate_hz=60.0)
    dt = 50e-6
    n_steps = int(1.0 / dt)
    fires = 0
    for k in range(n_steps):
        t = k * dt
        if schedule.is_due(t):
            fires += 1
            schedule.advance()
    assert abs(fires - 60) <= 1


def test_arrival_queue_delivers_in_arrival_time_order_even_if_pushed_out_of_order():
    queue: ArrivalQueue[str] = ArrivalQueue()
    queue.push(0.010, "captured_first_arrives_late")
    queue.push(0.005, "captured_second_arrives_early")
    ready_at_6ms = queue.pop_ready(0.006)
    assert ready_at_6ms == ["captured_second_arrives_early"]
    ready_at_11ms = queue.pop_ready(0.011)
    assert ready_at_11ms == ["captured_first_arrives_late"]


def test_arrival_queue_pop_ready_empty_when_nothing_due():
    queue: ArrivalQueue[int] = ArrivalQueue()
    queue.push(1.0, 42)
    assert queue.pop_ready(0.5) == []
