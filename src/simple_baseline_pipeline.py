"""
Simple pipeline: evaluates all criteria "at once" with a single GPT call,
producing a JSON map of {criterion_name: {reasoning, confidence, rating}}.
"""

import os
import json
import logging

from model_handler import ModelHandler

logger = logging.getLogger(__name__)


class SimpleBaselinePipeline:
    """
    A baseline pipeline that asks GPT to evaluate a summary
    against all criteria in one pass.
    """

    def __init__(self, criteria_path, in_context_learning_samples, client, model_id, single_model=True, multi_family=False):
        self.criteria = self._load_criteria(criteria_path)
        self.in_context_learning_samples = self._load_learning_samples(in_context_learning_samples)
        self.client = client
        self.model_id = model_id
        self.single_model = single_model
        # multi_family not used here

    def _load_criteria(self, criteria_path):
        definitions_dict = {}
        for file in os.listdir(criteria_path):
            if file.endswith(".json"):
                with open(os.path.join(criteria_path, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    filename = file.split(".")[0]
                    definitions_dict[filename] = data.get("definition", "")
        return definitions_dict

    def _load_learning_samples(self, learning_samples_path):
        """
        If you have a directory of JSON files with learning samples, load them here.
        For now, we’ll just return {} if no usage.
        """
        if not learning_samples_path or not os.path.isdir(learning_samples_path):
            logger.info("[MESA - SIMPLE BASELINE] No in-context learning samples directory found or not used.")
            return {}
        learning_samples = {}
        for file in os.listdir(learning_samples_path):
            if file.endswith(".json"):
                with open(os.path.join(learning_samples_path, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    filename = file.split("_")[0]
                    learning_samples[filename] = {
                        "likert_score": data[0].get("likert_score"),
                        "reasoning": data[0].get("reasoning"),
                    }
        return learning_samples

    def run(self, sample):
        logger.info("\n\n**************************\n [MESA - SIMPLE BASELINE] Running _ALL_AT_ONCE Pipeline\n**************************\n")

        # This pipeline lumps all criteria together in one call.
        score = self.pipeline(sample, self.criteria)

        # Transform the GPT response (which should be a dict of {criteria_name: {...}}) 
        # into the required "scores" list.
        scores = []
        for crit_name, details in score.items():
            assessment = {
                "criteria": crit_name,
                "selection": "",
                "filter": "",
                "score": {
                    "reasoning": details.get("reasoning", ""),
                    "confidence": int(details.get("confidence", 0)),
                    "rating": int(details.get("rating", 0)),
                },
            }
            scores.append(assessment)

        return scores

    def pipeline(self, data, criteria_dict):
        """
        Single GPT call that sees transcript, summary, and all criteria definitions at once,
        returning a JSON object with a rating per criterion.
        """
        system_prompt = (
            "You are an expert in meeting summarization quality assessment. "
            "You will see multiple criteria definitions and evaluate them in one shot."
        )

        # Combine all criteria definitions in a single text block
        combined_criteria = ""
        for crit, definition in criteria_dict.items():
            combined_criteria += f"{crit}: {definition}\n"

        user_prompt = (
            f"Transcript: <{data['transcript']}>\n"
            f"Summary: <{data['summary']}>\n"
            f"Criteria Definitions:\n{combined_criteria}\n\n"
            "Please return a JSON object mapping each criterion name to:\n"
            '{\n'
            '   "reasoning": "<chain-of-thought>",\n'
            '   "confidence": <0-10>,\n'
            '   "rating": <0-5>\n'
            '}\n'
            "No extra text outside the JSON."
        )

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_raw = ModelHandler.call_model_with_retry(self.client, message, self.model_id, max_tokens=4000)
        response_text = response_raw.choices[0].message.content.strip()
        response_text = response_text.strip("```json").strip("```").strip()
        logger.debug("[MESA - SIMPLE BASELINE] Response: %r", response_text)

        try:
            parsed = json.loads(response_text)
            return parsed
        except json.JSONDecodeError:
            logger.warning("[MESA - SIMPLE BASELINE] Could not parse JSON from SimpleBaselinePipeline. Returning empty.")
            return {}
