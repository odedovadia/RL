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

"""MMLU-Pro dataset."""

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class MMLUProDataset:
    DEFAULT_PROMPT = (
        "Answer the following multiple choice question. The last line of your response "
        "should be of the following format: 'Answer: $LETTER' (without quotes) where "
        "LETTER is one of A, B, C, D, E, F, G, H, I, J. Think step by step before answering.\n"
    )
    DEFAULT_VERIFIER_TYPE = "multilingual_multichoice"

    def __init__(self, prompt_file: str, system_prompt_file: Optional[str] = None):
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)

        self.task_spec = TaskDataSpec(
            task_name="MMLU-Pro",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        if self.task_spec.prompt is None:
            self.task_spec.prompt = self.DEFAULT_PROMPT
        self.processor = processors.multichoice_qa_processor

    def _rekey(self, data: dict[str, Any]):
        options = {chr(ord("A") + i): op for i, op in enumerate(data["options"])}
        return {
            "question": data["question"],
            "options": options,
            "answer": data["answer"],
            "subject": data["category"],
        }
