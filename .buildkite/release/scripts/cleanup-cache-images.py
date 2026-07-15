#!/usr/bin/env python3
"""Clean up old cache-pr-* tags from ECR Public.

Usage: cleanup-cache-images.py [--max-age-days N] [--repo REPO] [--dry-run]
"""

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone

CACHE_PREFIX = "cache-pr-"

# Max number of images that can be deleted through ecr-public's batch delete in one call.
# Ref: https://docs.aws.amazon.com/cli/latest/reference/ecr-public/batch-delete-image.html#
MAX_DELETE_BATCH_SIZE = 100


def run_aws(*args: str) -> str:
    """Run an aws CLI command as a subprocess and return stdout."""
    return subprocess.run(
        ["aws", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def get_cache_images(repo: str, region: str) -> list[tuple[str, datetime]]:
    """Get the set of cached [tagged] images & the datetime they were pushed at."""
    output = run_aws(
        "ecr-public",
        "describe-images",
        "--repository-name",
        repo,
        "--region",
        region,
        "--no-paginate",
        "--output",
        "json",
    )
    results = []
    for img in json.loads(output)["imageDetails"]:
        if "imageTags" not in img or "imagePushedAt" not in img:
            continue

        # Save the datetime of each tagged image matching the prefix
        pushed_dt = datetime.fromisoformat(img["imagePushedAt"])
        for tag in img["imageTags"]:
            if tag.startswith(CACHE_PREFIX):
                results.append((tag, pushed_dt))
    return results


def delete_tags(repo: str, region: str, tags: list[str]) -> None:
    """Delete the set of tagged images, batching to respect the 100 image limit."""
    for i in range(0, len(tags), MAX_DELETE_BATCH_SIZE):
        batch = tags[i : i + MAX_DELETE_BATCH_SIZE]
        image_ids = [f"imageTag={tag}" for tag in batch]
        run_aws(
            "ecr-public",
            "batch-delete-image",
            "--repository-name",
            repo,
            "--region",
            region,
            "--image-ids",
            *image_ids,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-age-days", type=int, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.max_age_days)
    print(f"Cleaning up {CACHE_PREFIX}* tags older than {args.max_age_days} days in {args.repo}")

    # Get all cached PR image tags that are older than the cutoff
    images = get_cache_images(args.repo, args.region)
    to_delete = [tag for tag, pushed in images if pushed < cutoff]

    kept = len(images) - len(to_delete)
    if to_delete:
        print(f"Found {len(to_delete)} tags to delete, keeping {kept}. Tags to delete: ")
        for tag in to_delete:
            print(f"  - {tag}")
    else:
        print("No PR cache images to delete. Nothing to do!")
        return

    if not args.dry_run:
        delete_tags(args.repo, args.region, to_delete)
        print(f"Cleanup complete: deleted {len(to_delete)} tags")
    else:
        print("No images were deleted [dryrun]")


if __name__ == "__main__":
    main()
