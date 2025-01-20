"""
BaselinePipeline: Similar to SimpleBaseline but processes each criterion individually (ONE_BY_ONE).
"""

import os
import json
import logging

from model_handler import ModelHandler

logger = logging.getLogger(__name__)


class BaselinePipeline:
    """
    A baseline pipeline that processes each criterion separately
    in a single GPT call (but repeated for each criterion).
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
        if not learning_samples_path or not os.path.isdir(learning_samples_path):
            logger.info("[MESA - BASELINE] No in-context learning samples directory found or not used.")
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
        logger.info("\n\n**************************\n[MESA - BASELINE] Running _ONE_BY_ONE Baseline\n**************************\n")
        scores = []

        logger.info("Criteria: %s", list(self.criteria.keys()))

        for criteria_name, definition_text in self.criteria.items():
            logger.info("[MESA - BASELINE] Current Criterion: %s", criteria_name)
            score = self.pipeline(sample, criteria_name, definition_text)

            assessment = {
                "criteria": criteria_name,
                "selection": "",
                "filter": "",
                "score": score,
            }
            scores.append(assessment)

        return scores

    def pipeline(self, data, criteria_name, definition_text):
        """
        For each criterion, we do a single GPT call to get rating, confidence, reasoning.
        """
        system_prompt = (
            "You are an expert in evaluating meeting summaries for specific types of errors."
            "You will be given the transcript and summary, plus one criterion definition."
            "Output strictly in JSON with keys: reasoning, confidence, rating."
        )

        user_prompt = (
            f"Transcript: <{data['transcript']}>\n"
            f"Summary: <{data['summary']}>\n"
            f"Criterion: {criteria_name}: {definition_text}\n"
            "Rate the summary from 0 to 5 (0=best, 5=worst). "
            "Also provide reasoning and a confidence from 0 to 10. Return strictly JSON:\n"
            '{\n'
            '  "reasoning": "<your reasoning>",\n'
            '  "confidence": <0-10>,\n'
            '  "rating": <0-5>\n'
            '}'
        )

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_raw = ModelHandler.call_model_with_retry(self.client, message, self.model_id, max_tokens=4000)
        response_text = response_raw.choices[0].message.content.strip()
        response_text = response_text.strip("```json").strip("```").strip()

        logger.debug("[MESA - BASELINE] Response: %r", response_text)

        try:
            parsed = json.loads(response_text)
            return parsed
        except json.JSONDecodeError:
            logger.warning("[MESA - BASELINE] Could not parse JSON from BaselinePipeline. Returning empty.")
            return {"reasoning": "", "confidence": 0, "rating": 0}
