"""
Step 2: Similarity Calculation
Calculate similarity between inference results from Step 1.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from lwj_tools.io.reader import FileReader
from utils.tools import get_controversy_output_dir
from utils.calls import InferSimilarityCall


def load_config(config_path: str) -> dict:
    config = FileReader.read(config_path)
    load_dotenv(".env")
    config["calc_similarity"]["base_url"] = os.environ["SIMILARITY_BASE_URL"].split(",")

    controversy_config = FileReader.read(config["controversy_config_path"])
    _config = controversy_config["calc_controversy"]
    _config["models"] = os.environ["CONTROVERSY_API_MODELS"].split(",")

    # step1 output dir as step2 input_dir
    config["controversy_dir"] = get_controversy_output_dir(controversy_config)

    config["output_dir"] = config["controversy_dir"].replace(
        "calc_event_controversy", "calc_similarity"
    )
    os.makedirs(config["output_dir"], exist_ok=True)

    config["splits"] = controversy_config["splits"]
    return config


def main():
    config = load_config("./configs/step2_similarity.yaml")

    controversy_dir = config["controversy_dir"]
    output_dir = config["output_dir"]

    # Find controversy files
    controversy_files = []
    for split in config["splits"]:
        split_file = os.path.join(controversy_dir, f"{split}.jsonl")
        if os.path.exists(split_file):
            controversy_files.append((split, split_file))

    if not controversy_files:
        print(f"[ERROR] No controversy files found in {controversy_dir}")
        return

    print(f"[INFO] Found {len(controversy_files)} split files, as follows:")
    for split, split_file in controversy_files:
        print(f"[INFO]\t{split} -> {split_file}")
    print(f"[INFO] Output directory: {output_dir}")

    similarity_call = InferSimilarityCall(
        base_url=config["calc_similarity"]["base_url"],
        max_workers=config["calc_similarity"]["max_workers"],
    )

    for split, input_file in controversy_files:
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        print(f"[INFO] Processing {split}: {input_file}")
        print(f"[INFO] Output: {output_file}")

        similarity_call(
            file_path=input_file,
            output_path=output_file,
            id_field="index",
            desc=f"calc_similarity {split}",
        )

        print(f"[DONE] {split} -> {output_file}")

    print(f"\n[INFO] All similarity calculations complete")
    print(f"[INFO] Output directory: {output_dir}")


if __name__ == "__main__":
    main()
