"""ClusterRegistryLogic claims, collisions, GC."""
from ray_hive.cluster.registry import ClaimKind, ClusterRegistryLogic, EntityState


def test_claim_collision():
    reg = ClusterRegistryLogic()
    reg.claim(ClaimKind.SERVE_ROUTE.value, "/markets", "lexis-markets")
    try:
        reg.claim(ClaimKind.SERVE_ROUTE.value, "/markets", "other")
        assert False, "expected collision"
    except RuntimeError as e:
        assert "collision" in str(e)


def test_same_project_reclaim_ok():
    reg = ClusterRegistryLogic()
    reg.claim("model_id", "m1", "p1")
    c = reg.claim("model_id", "m1", "p1")
    assert c.project_id == "p1"


def test_gc_dead_project_releases_claims():
    reg = ClusterRegistryLogic(heartbeat_miss_s=1.0, grace_s=1.0)
    reg.register_project(
        {"project_id": "p1", "namespace": "ns1", "serve_route_prefix": "/x"}
    )
    reg.claim("named_actor", "gate", "p1")
    reg._projects["p1"].last_heartbeat_at = 0.0
    out = reg.gc(now=100.0)
    assert out["purged"] >= 1
    assert reg.list_claims() == []


def test_heartbeat_keeps_alive():
    reg = ClusterRegistryLogic(heartbeat_miss_s=10.0, grace_s=10.0)
    reg.register_project({"project_id": "p1", "namespace": "ns"})
    assert reg.heartbeat_project("p1")
    assert reg.get_project("p1")["state"] == EntityState.ALIVE.value
