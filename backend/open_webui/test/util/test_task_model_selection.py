from open_webui.utils.task import get_task_model_id


def test_get_task_model_id_prefers_task_model_when_valid():
    models = {
        "default-model": {"connection_type": "external"},
        "task-model": {"connection_type": "external"},
    }

    task_model_id = get_task_model_id(
        default_model_id="default-model",
        task_model="task-model",
        models=models,
    )

    assert task_model_id == "task-model"


def test_get_task_model_id_falls_back_to_default_when_task_model_missing():
    models = {
        "default-model": {"connection_type": "external"},
    }

    task_model_id = get_task_model_id(
        default_model_id="default-model",
        task_model="missing-task-model",
        models=models,
    )

    assert task_model_id == "default-model"


def test_get_task_model_id_falls_back_to_default_when_task_model_not_set():
    models = {
        "default-model": {"connection_type": "external"},
        "legacy-task-model": {"connection_type": "external"},
    }

    task_model_id = get_task_model_id(
        default_model_id="default-model",
        task_model="",
        models=models,
    )

    assert task_model_id == "default-model"
