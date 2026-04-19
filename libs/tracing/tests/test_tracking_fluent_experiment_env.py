import uuid


def test_get_experiment_id_from_env_does_not_require_mlflowclient(monkeypatch, tmp_path):
    """
    The tracing-only distribution (mlflow-tracing) sets `IS_TRACING_SDK_ONLY=True`, which means
    `MlflowClient` is not imported into `mlflow.tracking.fluent`.

    Historically, `_get_experiment_id_from_env()` unconditionally referenced `MlflowClient`,
    raising `NameError` when users configured tracing via `MLFLOW_EXPERIMENT_NAME`.
    """
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tmp_path.as_uri())
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", f"exp-{uuid.uuid4()}")

    import mlflow.tracking.fluent as fluent

    # Simulate tracing-SDK-only mode, where `MlflowClient` is not imported.
    monkeypatch.delattr(fluent, "MlflowClient", raising=False)

    experiment_id = fluent._get_experiment_id_from_env()
    assert experiment_id is not None
