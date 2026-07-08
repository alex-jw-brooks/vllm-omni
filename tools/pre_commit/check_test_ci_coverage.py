#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a pre-commit hook that fails if test files are modified or added
that never run in the CI. For now, this means that every tests file
needs to have a CI level marker (e.g., core_model, advanced_model, full_model,
etc) so that we ensure mutated tests will actually be selected as long as
there are pytest commands pointing at the right paths.
"""

import os
import re
import sys

LEVEL_MARKERS = ("core_model", "advanced_model", "full_model", "slow")

# Match mark, since we could also from pytest import mark.
# \b is used to prevent accidentally matching against potential
# future markers with prefixes that overlap with the level markers
# by accident.
LEVEL_RE = re.compile(r"mark\.(?:" + "|".join(LEVEL_MARKERS) + r")\b")

# Check if a file matches test_<something>.py or
# /tests/some/path/<test> since precommit only passes
# this check ^/tests/ for now.
TEST_FILE_RE = re.compile(r"^tests/(?:.*/)?test_[^/]*\.py$")


def _is_test_file(path: str) -> bool:
    """Determine whether or not a path is pointing at a test file or not."""
    return bool(TEST_FILE_RE.search(path))


def _has_level_marker(path: str) -> bool:
    """Return True if the file path exists and has at least one level marker
    somewhere in the file; this passes for both a decorator per test func and
    module level marks at the moment."""
    if not os.path.isfile(path):
        return True
    with open(path, encoding="utf-8") as f:
        return bool(LEVEL_RE.search(f.read()))


def get_files_missing_markers(staged_files: list[str]) -> list[str]:
    """Given the staged files prefixed with ^tests/, determine which
    added/modified files have level markers."""
    missing_markers = []
    for path in staged_files:
        if _is_test_file(path) and not _has_level_marker(path):
            missing_markers.append(path)
    return missing_markers


if __name__ == "__main__":
    missing_markers = get_files_missing_markers(sys.argv[1:])

    if missing_markers:
        file_list = "\n  ".join(missing_markers)
        print(
            f"\033[91merror:\033[0m the following test file(s) have no CI "
            "level marker and will probably not be collected by Buildkite:\n"
            f"  {file_list}\n\n"
            "You likely need to add a pytestmark, e.g.:\n"
            "  pytestmark = [pytest.mark.core_model, pytest.mark.cpu]\n\n"
            "To skip this check: "
            "SKIP=check-test-ci-coverage git commit ..."
        )
        sys.exit(1)
