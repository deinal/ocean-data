#!/usr/bin/env python3
"""Upload the staged Baltic zarrs to a Hugging Face dataset repo."""
import argparse

from huggingface_hub import create_repo, upload_large_folder

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", default="deinal/njord-baltic-data")
parser.add_argument("--folder", default="data/hf/njord-baltic")
parser.add_argument("--public", action="store_true", help="Create the repo as public")
parser.add_argument("--num-workers", type=int, default=1)
args = parser.parse_args()

create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
upload_large_folder(
    repo_id=args.repo_id,
    folder_path=args.folder,
    repo_type="dataset",
    num_workers=args.num_workers,
)
