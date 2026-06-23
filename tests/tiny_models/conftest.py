from shutil import rmtree

import pytest

from tests.tiny_models.model_settings import DIFFUSION_TEST_SETTINGS


@pytest.fixture(scope="session")
def tiny_model_paths(request):
    """Build or download the tiny models for the selected tests.

    NOTE: this is session scoped to avoid churn in tiny model creation,
    but will ensure all the tiny models you need are created for the selected tests
    before it starts to execute them."""
    model_paths = {}
    print("Initializing tiny models...")
    for item in request.session.items:
        if not hasattr(item, "callspec"):
            continue
        model_name = item.callspec.params.get("model_name")
        if model_name is None:
            continue
        if model_name not in model_paths:
            print(f"Calling tiny model builder for: {model_name}")
            model_paths[model_name] = DIFFUSION_TEST_SETTINGS[model_name].builder()

    yield model_paths
    for path in model_paths.values():
        rmtree(path, ignore_errors=True)
