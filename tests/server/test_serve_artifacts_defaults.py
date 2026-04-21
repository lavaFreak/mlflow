import mlflow.server.handlers as server_handlers
from mlflow.server.constants import (
    ARTIFACT_ROOT_ENV_VAR,
    BACKEND_STORE_URI_ENV_VAR,
    SERVE_ARTIFACTS_ENV_VAR,
)
from mlflow.store.tracking import DEFAULT_ARTIFACTS_URI


def test_is_serving_proxied_artifacts_defaults_true_when_env_unset(monkeypatch):
    monkeypatch.delenv(SERVE_ARTIFACTS_ENV_VAR, raising=False)
    assert server_handlers._is_serving_proxied_artifacts() is True


def test_is_serving_proxied_artifacts_respects_falsey_values(monkeypatch):
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "false")
    assert server_handlers._is_serving_proxied_artifacts() is False
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "0")
    assert server_handlers._is_serving_proxied_artifacts() is False


def test_tracking_store_defaults_artifact_root_when_env_unset(monkeypatch):
    monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, "file:///tmp/mlruns")
    monkeypatch.delenv(ARTIFACT_ROOT_ENV_VAR, raising=False)
    monkeypatch.delenv(SERVE_ARTIFACTS_ENV_VAR, raising=False)

    captured = {}

    def fake_get_store(store_uri, artifact_root):
        captured["store_uri"] = store_uri
        captured["artifact_root"] = artifact_root
        return object()

    monkeypatch.setattr(server_handlers, "_tracking_store", None)
    monkeypatch.setattr(server_handlers.utils, "set_tracking_uri", lambda *_: None)
    monkeypatch.setattr(server_handlers._tracking_store_registry, "get_store", fake_get_store)

    server_handlers._get_tracking_store()

    assert captured["store_uri"] == "file:///tmp/mlruns"
    assert captured["artifact_root"] == DEFAULT_ARTIFACTS_URI

