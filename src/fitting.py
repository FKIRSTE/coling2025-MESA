"""
Module for 'Fitting' class that compares model output to human judgments.
"""

import math
import json
import logging
import os
import numpy as np  # if you prefer to handle NaN with numpy, optional
from collections import defaultdict


from model_handler import ModelHandler

logger = logging.getLogger(__name__)


class Fitting:
    """
    Conducts comparisons between GPT-based metrics and human scores,
    providing meta-level feedback on where the model disagrees with humans.
    """

    def __init__(self, client, model_id, criteria_path):
        """
        Args:
            client: The OpenAI-compatible client (e.g., AzureOpenAI).
            model_id: GPT model name.
            criteria_path: Path to directory containing *.json criteria definitions.
        """
        self.client = client
        self.model_id = model_id
        self.criteria = self._load_criteria(criteria_path)

    def _load_criteria(self, criteria_path):
        """
        Load criteria definitions from *.json files in the specified path.
        """
        list_of_criteria = []
        for file in os.listdir(criteria_path):
            if file.endswith(".json"):
                list_of_criteria.append(file.split(".")[0])
        return list_of_criteria

    def likert_score_feedback_numerical(self, score, human_score):
        """
        Example numeric difference measure.
        """
        if math.isnan(human_score):
            logger.warning("[MESA - FITTING] Human score is NaN; defaulting to 0.")
            human_score = 0

        diff = abs((score + 1) - (human_score + 1))
        return diff

    def likert_score_feedback(self, metric_score, human_score):
        """
        Compares a metric score and a human score on a 0-5 scale and categorizes the difference.
        """
        difference = abs(metric_score - human_score)

        # If metric says "no error" but human says "error"
        if metric_score == 0 and human_score >= 1:
            return (
                "Error Detection Discrepancy",
                f"Metric: 0 vs. Human: {human_score} (Human detects an error).",
            )

        # If metric says "error" but human says "no error"
        if metric_score >= 1 and human_score == 0:
            return (
                "Error Detection Discrepancy",
                f"Metric: {metric_score} vs. Human: 0 (Metric detects an error).",
            )

        if difference == 0:
            return (
                "No Difference",
                f"Scores match exactly. (Human: {human_score}; Metric: {metric_score})",
            )
        elif difference == 1:
            return (
                "Minor Difference",
                f"Scores differ by 1 point. (Human: {human_score}; Metric: {metric_score})",
            )
        elif difference == 2:
            return (
                "Moderate Difference",
                f"Scores differ by 2 points. (Human: {human_score}; Metric: {metric_score})",
            )
        else:  # difference >= 3
            return (
                "Major Difference",
                f"Scores differ by >= 3 points. (Human: {human_score}; Metric: {metric_score})",
            )

    def reasoning_feedback(self, model_reasoning, human_reasoning):
        """
        Asks GPT to compare candidate reasoning to human reasoning, returning a quality score.
        """
        system_prompt = (
            "You are a judge tasked with comparing a candidate reasoning to a human reasoning. "
            "You may find differences acceptable unless they are major or inaccurate. "
            "Return a chain-of-thought and a single-value quality score (0 to 100)."
        )

        user_prompt = (
            "Candidate reasoning:\n"
            f"{model_reasoning}\n\n"
            "Human reasoning:\n"
            f"{human_reasoning}\n\n"
            "Please provide your chain-of-thought and a single integer rating from 0 to 100."
        )

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_raw = ModelHandler.call_model_with_retry(
            self.client, message, self.model_id, max_tokens=2000
        )
        response = response_raw.choices[0].message.content.strip()
        return response

    def quality_report(self, list_of_comparisons):
        """
        Summarizes how the metric compares to human judgments across multiple samples.
        """
        system_prompt = (
            "You are a supervisor providing feedback to a junior evaluator on how to improve. "
            "You will be given comparisons of metric vs. human evaluations."
        )
        user_prompt = (
            f"All comparisons:\n{list_of_comparisons}\n\n"
            "Now provide meta-level feedback on how to improve scoring and reasoning. "
            "Return valid JSON in the format:\n"
            '{\n'
            '  "score_similarity": "<feedback>",\n'
            '  "reasoning_quality": "<feedback>"\n'
            '}\n'
        )

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_raw = ModelHandler.call_model_with_retry(
            self.client, message, self.model_id, max_tokens=2000
        )
        text = response_raw.choices[0].message.content.strip()
        try:
            literal = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("[MESA - FITTING] Could not parse JSON from quality report. Returning raw text.")
            return text
        return literal

    def extract_error_type_ratings(self, overall_quality, error_type):
        """
        Extract the ratings for a specific error type from a list of feedback dictionaries.
        """
        error_type_ratings = []
        for sample_feedback in overall_quality:
            for criteria_feedback in sample_feedback:
                if criteria_feedback["criteria"] == error_type:
                    error_type_ratings.append(criteria_feedback["likert_score"])
        return error_type_ratings


    def group_feedback_by_error_type(self, feedback_list):
        """
        Groups the feedback items by the 'criteria' field.
        
        Args:
            feedback_list (list of dict): e.g., [
                {
                    'criteria': 'incoherence',
                    'distance': ...,
                    'quality': ...
                },
                ...
            ]
        
        Returns:
            dict: A dictionary keyed by error type (str) -> list of feedback dicts
                Example:
                    {
                        'incoherence': [ {...}, {...} ],
                        'language': [ {...}, {...} ],
                        ...
                    }
        """
        grouped = defaultdict(list)
        for feedback in feedback_list:
            error_type = feedback["criteria"]
            grouped[error_type].append(feedback)
        return dict(grouped)


    def run(self, scores, human_scores):
        """
        Compare model 'scores' to 'human_scores' across multiple samples
        and produce a final aggregated report.
        """
        
        overall_quality = []
        overall_reports = []

        for index, model_score_dict in scores.items():
            working_sample = {
                "scores": model_score_dict,
                "human_scores": human_scores[index] if human_scores else None,
            }

            feedback_per_criteria = []
            for criteria in working_sample["scores"]:
                error_type = criteria["criteria"]
                
                metric_rating = criteria["score"]["rating"]  # e.g. 0-5
                metric_reasoning = criteria["score"]["reasoning"]

                if working_sample["human_scores"]:
                    human_rating = working_sample["human_scores"][error_type]["impact"]
                    human_reasoning = working_sample["human_scores"][error_type]["reasoning"]
                else:
                    human_rating = 0
                    human_reasoning = ""

                distance = self.likert_score_feedback(metric_rating, human_rating)
                quality = self.reasoning_feedback(metric_reasoning, human_reasoning)

                logger.info("[MESA - FITTING] Distance for %s: %s", error_type, distance)
                logger.debug("[MESA - FITTING] Quality for %s: %s", error_type, quality)

                feedback_per_criteria.append(
                    {
                        "criteria": error_type,
                        "distance": distance,
                        "quality": quality,
                    }
                )


            grouped_feedback_by_error_type = self.group_feedback_by_error_type(feedback_per_criteria)
            
            overall_quality = []
            for group in grouped_feedback_by_error_type:
                quality_per_error_type = self.quality_report(group)
                overall_quality.append(quality_per_error_type)
            

            # -----


            # logging.info("[MESA - FITTING] Feedback per criteria: %s", feedback_per_criteria)

            # quality_per_sample = self.quality_report(feedback_per_criteria)
            
            # logging.info(f"Quality per sample: {quality_per_sample}")
            
            # overall_quality.append(quality_per_sample)

            # overall_reports = []

            # for error_type in self.criteria:
            #     error_type_ratings = self.extract_error_type_ratings(overall_quality, error_type)
            #     report = self.quality_report(error_type_ratings)
            #     overall_reports.append(report)


            # ------


            # Summarize per-sample
            # quality_per_sample = self.quality_report(feedback_per_criteria)
            # overall_quality.append(feedback_per_criteria)
            # overall_quality.append({"sample": working_sample})
            # overall_reports.append(quality_per_sample)

            # logging.info("[MESA - FITTING] Quality report for sample %s: %s", index, quality_per_sample)
            
            # logging.info(overall_quality)


        # (Optional) final aggregated step by error type
        
        # all_report = []
        # for error_type in self.criteria:
        #     error_type_ratings = self.extract_error_type_ratings(overall_quality, error_type)
        #     report = self.quality_report(error_type_ratings)
        #     all_report.append(report) 
        #     # Possibly do another call to quality_report() or stats, etc.

        # logging.info("[MESA - FITTING] Final aggregated report: %s", all_report)
        
        return overall_reports, overall_reports


