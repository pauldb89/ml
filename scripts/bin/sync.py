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

    command = (
        f"fswatch -o {args.source_dir} | xargs -n1 -I{{}} rsync -avz {args.source_dir} "
        f"{args.target_username}@{args.target_host}:/home/{args.target_username}/{args.target_dir}"
    )
    print(f"Executing {command}")
    os.system(command)


if __name__ == "__main__":
    main()
