"""Hive Ray namespace default and env override."""
from unittest.mock import patch

from ray_hive.core.ray_utils.naming import DEFAULT_RAY_NAMESPACE, NAMESPACE_ENV, ray_namespace
from ray_hive.core.ray_utils.session import init_ray


def test_ray_namespace_default(monkeypatch):
    monkeypatch.delenv(NAMESPACE_ENV, raising=False)
    assert ray_namespace() == DEFAULT_RAY_NAMESPACE == "ray_hive"


def test_ray_namespace_env(monkeypatch):
    monkeypatch.setenv(NAMESPACE_ENV, "other-hive")
    assert ray_namespace() == "other-hive"


def test_ray_namespace_blank_falls_back(monkeypatch):
    monkeypatch.setenv(NAMESPACE_ENV, "   ")
    assert ray_namespace() == DEFAULT_RAY_NAMESPACE


def test_init_ray_uses_hive_namespace(monkeypatch):
    monkeypatch.delenv(NAMESPACE_ENV, raising=False)
    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)

    with patch("ray_hive.core.ray_utils.session.ray.init", side_effect=fake_init):
        init_ray("ray://head:10001", suppress_logging=True)

    assert captured["namespace"] == "ray_hive"
    assert captured["runtime_env"]["env_vars"][NAMESPACE_ENV] == "ray_hive"
    assert ray_namespace() == "ray_hive"


def test_init_ray_respects_explicit_namespace(monkeypatch):
    monkeypatch.delenv(NAMESPACE_ENV, raising=False)
    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)

    with patch("ray_hive.core.ray_utils.session.ray.init", side_effect=fake_init):
        init_ray("ray://head:10001", suppress_logging=True, namespace="custom-ns")

    assert captured["namespace"] == "custom-ns"
    assert captured["runtime_env"]["env_vars"][NAMESPACE_ENV] == "custom-ns"
    assert ray_namespace() == "custom-ns"
