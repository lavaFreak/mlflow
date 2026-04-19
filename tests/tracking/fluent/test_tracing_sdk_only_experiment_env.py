import uuid

import mlflow


def test_get_experiment_id_from_env_does_not_require_mlflowclient_symbol(monkeypatch, tmp_path):
    import mlflow.tracking.fluent as fluent
    from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_EXPERIMENT_NAME

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())

    experiment_name = f"exp-{uuid.uuid4().hex}"
    monkeypatch.delenv(MLFLOW_EXPERIMENT_ID.name, raising=False)
    monkeypatch.setenv(MLFLOW_EXPERIMENT_NAME.name, experiment_name)

    had_mlflow_client = "MlflowClient" in fluent.__dict__
    mlflow_client = fluent.__dict__.get("MlflowClient")
    fluent.__dict__.pop("MlflowClient", None)
    try:
        assert fluent._get_experiment_id_from_env() is not None
    finally:
        if had_mlflow_client:
            fluent.__dict__["MlflowClient"] = mlflow_client


def test_get_experiment_id_tracing_sdk_only_does_not_require_default_experiment_registry(
    monkeypatch, tmp_path
):
    import mlflow.tracking.fluent as fluent
    from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_EXPERIMENT_NAME

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())

    monkeypatch.delenv(MLFLOW_EXPERIMENT_NAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_EXPERIMENT_ID.name, raising=False)
    monkeypatch.setattr(fluent, "IS_TRACING_SDK_ONLY", True)

    had_default_registry = "default_experiment_registry" in fluent.__dict__
    default_registry = fluent.__dict__.get("default_experiment_registry")
    fluent.__dict__.pop("default_experiment_registry", None)

    old_active_experiment_id = fluent._active_experiment_id
    fluent._active_experiment_id = None
    try:
        assert fluent._get_experiment_id() == "0"
    finally:
        fluent._active_experiment_id = old_active_experiment_id
        if had_default_registry:
            fluent.__dict__["default_experiment_registry"] = default_registry

