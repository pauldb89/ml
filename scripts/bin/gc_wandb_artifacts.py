from argparse import ArgumentParser

import wandb


def main():
	parser = ArgumentParser()
	parser.add_argument("--project", type=str, required=True, help="Wandb project")
	args = parser.parse_args()

	api = wandb.Api(overrides={"project": args.project, "entity": "pauldb"})

	remaining_model_artifacts = 0
	remaining_model_bytes = 0
	deleted_model_bytes = 0
	deleted_model_artifacts = 0
	for run in api.runs():
		for artifact in run.logged_artifacts():
			if artifact.type != "model" or "latest" in artifact.aliases:
				remaining_model_bytes += artifact.size
				remaining_model_artifacts += 1
				continue

			deleted_model_bytes += artifact.size
			deleted_model_artifacts += 1
			artifact.delete()

	print(f"Deleted {deleted_model_artifacts} artifacts requiring {deleted_model_bytes / (1 << 30):.2f}GB")
	print(f"{remaining_model_artifacts} artifacts remain requiring {remaining_model_bytes / (1 << 30):.2f}GB")


if __name__ == "__main__":
	main()
