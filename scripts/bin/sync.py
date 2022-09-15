#!/usr/bin/python3

import os
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--source_dir", default="/Users/pauldb/code/ml", help="Source directory")
    parser.add_argument("--target_host", required=True, help="Remote host")
    parser.add_argument("--target_username", default="ubuntu", help="Remote username")
    parser.add_argument("--target_dir", default="myfs/code", help="Remote directory")
    args = parser.parse_args()

    source = args.source_dir
    target = f"{args.target_username}@{args.target_host}:/home/{args.target_username}/{args.target_dir}"

    rsync_command = f"rsync -avz --delete --exclude=.git/ {source} {target}"
    bash_command = f"while ! {rsync_command}; do sleep 5; done"
    command = f"fswatch -o {source} | xargs -n1 sh -c '{bash_command}'"

    print(f"Executing {command}")
    os.system(command)
    print("Script terminated")


if __name__ == "__main__":
    main()
