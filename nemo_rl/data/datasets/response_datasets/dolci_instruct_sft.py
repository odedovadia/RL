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

from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class DolciInstructSFTDataset(RawDataset):
    """Wrapper around the allenai/Dolci-Instruct-SFT dataset.

    The dataset contains 2.15M samples across diverse domains (Math, Coding,
    Safety, Multilingual, Science, etc.) with messages already in OpenAI chat
    format. Message dicts may contain extra null fields (function_calls,
    functions) that are stripped during formatting.

    Args:
        split: HuggingFace split to load, default is "train"
        split_validation_size: Fraction of data used for validation, default is 0.05
        seed: Seed for train/validation split, default is 42
        max_samples: Optional maximum number of samples to use from the dataset
    """

    def __init__(
        self,
        split: str = "train",
        split_validation_size: float = 0.05,
        seed: int = 42,
        max_samples: int | None = None,
        **kwargs,
    ) -> None:
        self.task_name = "dolci_instruct_sft"

        self.dataset = load_dataset("allenai/Dolci-Instruct-SFT", split=split)

        if max_samples is not None and max_samples > 0:
            self.dataset = self.dataset.shuffle(seed=seed).select(
                range(min(max_samples, len(self.dataset)))
            )

        # Filter out samples with empty assistant responses or wrong last role
        self.dataset = self.dataset.filter(self._is_valid_sample)

        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=["id", "source_dataset", "domain"],
        )

        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    @staticmethod
    def _is_valid_sample(data: dict[str, Any]) -> bool:
        messages = data.get("messages")
        if not messages:
            return False
        last = messages[-1]
        if last.get("role") != "assistant" or not last.get("content"):
            return False
        # Reject samples where any message has None content (chat templates
        # cannot handle it)
        if any(m.get("content") is None for m in messages):
            return False
        return True

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        # Strip messages to only role+content (dataset may include extra null
        # fields like function_calls/functions); coerce None -> "" as a safety
        # net since Jinja chat templates cannot iterate over None content
        cleaned = [
            {"role": m["role"], "content": m["content"] or ""}
            for m in data["messages"]
        ]
        return {"messages": cleaned, "task_name": self.task_name}
