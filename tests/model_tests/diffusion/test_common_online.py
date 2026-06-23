"""
Analogous to test_common_offline, but for server tests. Validates the full
online serving stack (CLI arg parsing, subprocess, API routing, response
encoding) using tiny models.
"""

import pytest

from tests.helpers.runtime import OmniServer, OpenAIClientHandler
from tests.model_tests.diffusion.case_filtering import get_parametrized_options
from tests.model_tests.diffusion.config_types import (
    DiffAccs,
    DiffTasks,
    build_server_args_from_diff_accelerations,
)
from tests.model_tests.diffusion.model_settings import DIFFUSION_TEST_SETTINGS
from tests.model_tests.diffusion.task_runners import (
    run_and_validate_online_image_to_image_request,
    run_and_validate_online_text_to_image_request,
)


@pytest.mark.parametrize(
    "model_name,accelerations,supported_tasks",
    get_parametrized_options(DIFFUSION_TEST_SETTINGS),
)
def test_online_on_supported_tasks(
    model_name: str,
    accelerations: list[DiffAccs] | None,
    supported_tasks: list[DiffTasks],
    tiny_model_paths: dict[str, str],
    subtests,
):
    """Smoke test: start a tiny model server and run each supported task via the API."""
    model_path = tiny_model_paths[model_name]
    server_args = build_server_args_from_diff_accelerations(accelerations)
    server_args.append("--enforce-eager")

    with OmniServer(model_path, server_args) as server:
        client = OpenAIClientHandler(
            host=server.host,
            port=server.port,
            api_key="EMPTY",
            log_stats=server.log_stats,
        )
        for task_type in supported_tasks:
            with subtests.test(msg=task_type):
                if task_type == DiffTasks.TEXT_TO_IMAGE:
                    run_and_validate_online_text_to_image_request(server, client)
                elif task_type == DiffTasks.IMAGE_EDIT:
                    run_and_validate_online_image_to_image_request(server, client)
                else:
                    raise ValueError(f"Task type {task_type} is not yet supported")
