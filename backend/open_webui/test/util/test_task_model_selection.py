from open_webui.utils.task import get_task_model_id


def test_get_task_model_id_prefers_external_when_valid():
    models = {
        "default-model": {"connection_type": "external"},
        "task-external-model": {"connection_type": "external"},
    }

    task_model_id = get_task_model_id(
        default_model_id="default-model",
        task_model="unused-local-task-model",
        task_model_external="task-external-model",
        models=models,
    )

    assert task_model_id == "task-external-model"


def test_get_task_model_id_falls_back_to_default_when_external_missing():
    models = {
        "default-model": {"connection_type": "external"},
    }

    task_model_id = get_task_model_id(
        default_model_id="default-model",
        task_model="unused-local-task-model",
        task_model_external="missing-external-model",
        models=models,
    )

    assert task_model_id == "default-model"


def test_get_task_model_id_falls_back_to_default_when_external_not_set():
    models = {
        "default-model": {"connection_type": "external"},
        "legacy-local-model": {"connection_type": "local"},
    }

    task_model_id = get_task_model_id(
        default_model_id="default-model",
        task_model="legacy-local-model",
        task_model_external="",
        models=models,
    )

    assert task_model_id == "default-model"
