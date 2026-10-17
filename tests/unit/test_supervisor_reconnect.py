"""Supervisor reconnect policy + desired-state (no Ray)."""
from ray_hive.supervisor.desired_state import DesiredState, StickyModel
from ray_hive.supervisor.reconnect import ReconnectPolicy, is_thread_exhaustion


def test_reconnect_backoff():
    p = ReconnectPolicy(base_s=30, max_s=300)
    assert p.should_attempt(0)
    d = p.note_failure(0.0)
    assert d == 30
    assert not p.should_attempt(10)
    assert p.should_attempt(30)
    p.note_success()
    assert p.failures == 0


def test_serve_cooldown():
    p = ReconnectPolicy(serve_cooldown_s=600)
    assert p.should_attempt_serve(0)
    p.note_serve_attempt(100.0)
    assert not p.should_attempt_serve(200.0)
    assert p.should_attempt_serve(700.0)


def test_thread_wedge():
    p = ReconnectPolicy(thread_fail_exit_after=2)
    p.note_failure(0, thread_exhausted=True)
    assert not p.should_exit_wedged()
    p.note_failure(1, thread_exhausted=True)
    assert p.should_exit_wedged()
    assert is_thread_exhaustion("can't start new thread")


def test_desired_state_roundtrip(tmp_path):
    db = tmp_path / "state.db"
    s = DesiredState(db)
    s.pin(
        StickyModel(
            model_id="m1",
            model_name="Qwen/Qwen3-0.6B",
            max_input_prompt_length=256,
            max_output_prompt_length=64,
            replicas=1,
            vllm_kwargs={"max_num_seqs": 1},
        )
    )
    rows = s.list()
    assert len(rows) == 1 and rows[0].model_id == "m1"
    s.unpin("m1")
    assert s.list() == []
