# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os
import pprint
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.collate_fn import eval_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_eval_dataset
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.evals.eval import MasterConfig, run_env_eval, setup
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config

TokenizerType = PreTrainedTokenizerBase


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Evaluation with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets to evaluate (e.g. math500,aime2024). "
        "Starts the generation server once and runs all datasets sequentially.",
    )

    # Parse known args for the script
    args, remaining = parser.parse_known_args()

    # Convert remaining args to OmegaConf format
    overrides = OmegaConf.from_dotlist(remaining)

    return args, overrides


def setup_data(tokenizer: AutoTokenizer, data_config, env_configs):
    print("Setting up data...")

    # load dataset
    base_dataset = load_eval_dataset(data_config)
    rekeyed_ds = base_dataset.rekeyed_ds

    # Apply dataset-specific verifier type if available
    verifier_type = getattr(base_dataset, "DEFAULT_VERIFIER_TYPE", None)
    math_env_cfg = dict(env_configs["math"])
    if verifier_type is not None:
        math_env_cfg["verifier_type"] = verifier_type

    env = MathEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.math_environment.MathEnvironment"
            )
        }
    ).remote(math_env_cfg)

    dataset = AllTaskProcessedDataset(
        dataset=rekeyed_ds,
        tokenizer=tokenizer,
        default_task_data_spec=base_dataset.task_spec,
        task_data_processors=base_dataset.processor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    return dataset, env, tokenizer


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "evals", "eval.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        override_conf = OmegaConf.from_cli()
        print(f"Overrides: {override_conf}")
        config = OmegaConf.merge(config, override_conf)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Init ray
    init_ray()

    # Setup tokenizer
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(
        config["generation"], tokenizer, is_eval=True
    )

    # Determine datasets to evaluate
    datasets = (
        args.datasets.split(",") if args.datasets else [config["data"]["dataset_name"]]
    )
    multi_dataset = len(datasets) > 1

    # Setup data for the first dataset (needed to initialize the generation server)
    data_config = copy.deepcopy(config["data"])
    data_config["dataset_name"] = datasets[0]
    dataset, env, tokenizer = setup_data(tokenizer, data_config, config["env"])

    # Start generation server once
    generation, dataloader, master_config = setup(config, tokenizer, dataset)

    for dataset_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {dataset_name}")
        print(f"{'=' * 60}\n")

        # Load dataset-specific data and dataloader
        ds_data_config = copy.deepcopy(master_config["data"])
        ds_data_config["dataset_name"] = dataset_name
        dataset, env, tokenizer = setup_data(
            tokenizer, ds_data_config, master_config["env"]
        )
        dataloader = DataLoader(
            dataset,
            batch_size=master_config["generation"]["num_prompts_per_step"],
            shuffle=False,
            collate_fn=eval_collate_fn,
        )

        # Build per-dataset config for save paths
        ds_master_config = copy.deepcopy(master_config)
        ds_master_config["data"]["dataset_name"] = dataset_name
        base_save_path = master_config["eval"].get("save_path")
        if base_save_path and multi_dataset:
            ds_master_config["eval"]["save_path"] = os.path.join(
                base_save_path, dataset_name
            )

        run_env_eval(
            generation,
            dataloader,
            env,
            ds_master_config,
            shutdown_generation=False,
        )

    generation.shutdown()


if __name__ == "__main__":
    main()
