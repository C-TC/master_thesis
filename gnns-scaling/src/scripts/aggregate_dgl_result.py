import json
import re
import csv
import argparse
import numpy as np

from pathlib import Path

DATASETS = ("uniform", "kronecker")
DATASET_REGEX = r"n[0-9]+(_[a-z][0-9]+)+_s0\.[0-9]+"


def get_exp_type(filename):
    filename_chunks = filename.split("_")
    if "norm" in filename_chunks:
        return "NORM"
    elif "dgnn" in filename_chunks:
        return "DGNN"
    else:
        return "DIST"


def run(args):
    dataset = args.dataset.lower()
    data_path = Path(args.data_dir)

    dataset_dirs = [
        f
        for f in data_path.iterdir()
        if f.is_dir() and re.match(DATASET_REGEX, f.name) is not None
    ]
    if dataset_dirs:
        print(f"Number of found dataset dirs: '{len(dataset_dirs)}'")
    else:
        raise RuntimeError(
            "No dataset dirs found. Did you provide the correct path?"
        )

    dataset_dirs.sort(key=lambda p: int(p.name.split("_")[0][1:]))

    csv_res = data_path / ("dgl_agg_res_" + dataset + ".csv")
    csv_f = csv_res.open("w")

    csv_writer = csv.writer(csv_f)

    for i, dataset_dir in enumerate(dataset_dirs, 1):
        print(
            "[{}/{}] collecting data from '{}'".format(
                i, len(dataset_dirs), str(dataset_dir)
            )
        )
        name_split = dataset_dir.name.split("_")
        num_nodes = name_split[0][1:]
        sparsity = name_split[-1][1:]

        num_vertices = None
        metadata_file = dataset_dir / (dataset_dir.name + ".json")
        try:
            with metadata_file.open() as f:
                metadata = json.load(f)
                num_vertices = metadata["num_nodes"]
        except Exception as e:
            print(f"Got {e} while parsing '{str(metadata_file)}'")

        res_files = [
            f
            for f in dataset_dir.iterdir()
            if f.is_file() and f.name.startswith("node") and f.suffix != "json"
        ]

        dataset_res = {}
        last_batch_size = None
        for file in res_files:
            model_name = file.stem.split("_")[-1]
            try:
                with file.open() as f:
                    result = json.load(f)
                    exp_type = get_exp_type(file.stem)
                    batch_size = result.get("batch_size")
                    datetime = result["datetimes"][-1]
                    run_times = result["eval_times"]
            except Exception as e:
                print(f"Got {e} while reading '{str(file)}'")
                continue

            exp_case = (model_name, exp_type)
            run_time = [datetime, np.median(run_times), batch_size]

            if exp_case in dataset_res:
                if exp_type == "NORM":
                    run_time[1] += np.median(result["sample_times"])
                dataset_res[exp_case] = max(
                    dataset_res[exp_case], run_time, key=lambda res: res[1]
                )
            else:
                dataset_res[exp_case] = run_time

            if exp_type == "DIST":
                exp_case = (model_name, "DIST_SAMPLE")
                run_time = run_time.copy()
                run_time[1] += np.median(result["sample_times"])
                if exp_case in dataset_res:
                    dataset_res[exp_case] = max(
                        dataset_res[exp_case], run_time, key=lambda res: res[1]
                    )
                else:
                    dataset_res[exp_case] = run_time

            if (
                last_batch_size is not None
                and batch_size is not None
                and batch_size != last_batch_size
            ):
                print("WARNING: different batch sizes in a single dataset")
            last_batch_size = batch_size

        # sort by model names
        dataset_res = {
            k: dataset_res[k]
            for k in sorted(dataset_res.keys(), key=lambda x: x[0])
        }

        for (model, exp_type), (
            datetime,
            eval_time,
            batch_size,
        ) in dataset_res.items():
            print("found {} - {}".format(model, exp_type))
            csv_writer.writerow(
                (
                    datetime,
                    model,
                    num_nodes,
                    num_vertices,
                    128,
                    sparsity,
                    eval_time,
                    exp_type,
                    batch_size,
                )
            )
    csv_f.close()
    print(f"Aggregated resutls save to: '{csv_res}'")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        f"""Aggregate DGL experiment results into a CSV file. Seeks experiment
        data directories that match the regex '${DATASET_REGEX}' in data_dir.
        Then parses result files (the ones whose names start with node) in the
        found directories. The aggregated results are sorted by the number of
        nodes."""
    )
    argparser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="data dir",
    )
    argparser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="kronecker",
        help="dataset name",
    )
    args = argparser.parse_args()

    if args.dataset.lower() not in DATASETS:
        raise ValueError(
            f"Not recognized dataset: f{args.dataset.lower()}. Available are: "
            + ", ".join(DATASETS)
        )

    run(args)
