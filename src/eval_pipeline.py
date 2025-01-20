"""
Evaluation pipeline that runs a multi-step approach (THREE_STEP) to identify errors,
filter them, and rate summary severity. Now only GPT-based (no LLaMA, Gemini, Phi).
"""

import os
import json
import logging

from panel import Panel
from model_handler import ModelHandler

logger = logging.getLogger(__name__)


class EvalPipeline:
    """
    A pipeline that evaluates a summary against certain criteria in three steps:
    1) Collect potential error instances
    2) Filter which are actual errors
    3) Rate the severity of those errors for the summary
    """

    def __init__(self, criteria_path, in_context_learning_samples, client, model_id, agents=None, single_model=True, multi_family=False):
        """
        Args:
            criteria_path (str): Path to directory containing *.json criteria definitions.
            in_context_learning_samples (str): Path to a CSV/JSON with in-context examples.
            client: OpenAI-compatible client.
            model_id (str): The GPT model to use.
            agents: Optional, if you want to override with a custom Panel or agent list.
            single_model (bool): If True, we do a single GPT call. If False, we use the Panel approach.
            multi_family (bool): If True, it suggests multiple GPT model usage in Panel, but currently not used.
        """
        self.criteria = self._load_criteria(criteria_path)
        self.in_context_learning_samples = self._load_learning_samples_from_csv(in_context_learning_samples)
        self.in_context_learning_samples_path = in_context_learning_samples
        self.client = client
        self.model_id = model_id
        self.agents = agents
        self.single_model = single_model
        self.multi_family = multi_family

        logger.info("### EVAL PIPELINE ###")
        logger.info("[MESA - EVAL PIPELINE] single_model: %s, multi_family: %s", single_model, multi_family)

    def _load_criteria(self, criteria_path):
        """
        Load each JSON file from the criteria_path and store {filename: definition}.
        """
        definitions_dict = {}
        for file in os.listdir(criteria_path):
            if file.endswith(".json"):
                with open(os.path.join(criteria_path, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    filename = file.split(".")[0]
                    definitions_dict[filename] = data.get("definition", "")
        return definitions_dict

    def _load_learning_samples_from_csv(self, csv_file_path):
        """
        Example function that tries to parse the file content as JSON
        (since the user code suggests the CSV might actually be JSON).
        """
        if not csv_file_path or not os.path.exists(csv_file_path):
            logger.info("[MESA - EVAL PIPELINE] In-context learning samples path not provided or invalid.")
            return {}

        try:
            with open(csv_file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
                data = json.loads(file_content)  # The file might be JSON-formatted
                learning_samples = {}
                for error_type, feedback_data in data.items():
                    cleaned = error_type.replace("_scores.csv", "")
                    suggestion = feedback_data.get("suggestions", "No suggestion available")
                    learning_samples[cleaned] = suggestion
                logger.debug("[MESA - EVAL PIPELINE] Learning samples: %s", learning_samples)
                return learning_samples
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("[MESA - EVAL PIPELINE] Could not parse in-context samples as JSON: %s", exc)
            return {}
        

    def compute_quality_score(self, scores, importance_weights=None):
        """
        Compute the quality score based on the provided scores, confidences, and importance weights.

        Args:
            scores (list): A list of dictionaries, where each dictionary has:
                - 'criteria': The error type (e.g., 'OM', 'HAL', 'IRR').
                - 'certainty': The confidence score as a float (0-1).
                - 'rating': The Likert rating score as an integer (0-5).
            importance_weights (dict): A dictionary mapping error types to importance weights.
                                    Default weights will be used if not provided.

        Returns:
            float: The computed quality score (1-10).
        """
        # Default importance weights
        if importance_weights is None:
            importance_weights = {
                "omission": 1.1, "hallucination": 1.1, "irrelevance": 1.1,  # High-priority errors
                "repetition": 0.9, "incoherence": 0.9, "language": 0.9  # Low-priority errors
            }

        numerator = 0
        denominator = 0

        for entry in scores:
            criteria = entry['criteria'].upper()
            certainty = entry['certainty']
            rating = entry['rating']
            importance = importance_weights.get(criteria, 1.0)  # Default to 1.0 if not found

            weighted_value = rating * (certainty * importance)
            numerator += weighted_value
            denominator += (certainty * importance)

        # Calculate impact
        impact = numerator / denominator if denominator != 0 else 0

        # Calculate quality
        quality = 1 + ((5 - impact) / 5) * 9
        return quality



    def run(self, sample):
        """
        Called externally to run the pipeline on a single sample (summary+transcript).
        Returns a list of dictionaries with the pipeline's outcome for each criterion.
        """
        logger.info("\n\n************************** \n [MESA - EVAL PIPELINE] Running _THREE_STEP EVAL pipeline \n**************************\n")

        scores = []
        for criteria in self.criteria:
            score, selection, filter_output, protocol = self.pipeline(sample, criteria)
                        
            assessment = {
                "criteria": criteria,
                "selection": selection,
                "filter": filter_output,
                "score": score,
                "protocol": protocol,
            }
            scores.append(assessment)
        
        quality_score = self.compute_quality_score(scores)
        logging.info("[MESA - EVAL PIPELINE] Quality score: %s", quality_score)
        
        return scores, quality_score

    def pipeline(self, data, criteria):
        """
        3-step approach for each criterion:
        1) Collect potential error instances
        2) Filter them to see which are real
        3) Assign an overall severity rating (0-5)
        """
        logger.info("[MESA - EVAL PIPELINE] Criteria: %s", criteria)

        system_prompt = (
            "You are a Judge tasked to decide if a given summary has specific errors. "
            "You will follow a multi-step process to identify and score these errors."
        )

        # --- Step 1
        format_step_1 = (
            '[{"instance": "<text>", "reasoning": "<reason>", "certainty": "<0-100>"}, ...]'
        )
        user_prompt_step_1 = (
            f"Step 1: Collect potential error instances.\n"
            f"Criteria: *** {criteria}: {self.criteria[criteria]} ***\n"
            f"Summary: ** {data['summary']} **\n"
            f"Transcript: ** {data['transcript']} **\n"
            "List instances where this error *could* occur. Provide short reasoning and a certainty score (0-100). "
            "Return strictly valid JSON with the format:\n"
            f"{format_step_1}"
        )

        list_of_instances, log_instances = self.task_execution(system_prompt, user_prompt_step_1)

        # --- Step 2
        format_step_2 = (
            '[{"instance": "<text>", "reasoning": "<reason>", "certainty": "<0-100>", "error_exists": true}, ...]'
        )
        user_prompt_step_2 = (
            f"Step 2: Filter actual errors.\n"
            f"Criteria: *** {criteria}: {self.criteria[criteria]} ***\n"
            f"Potential instances: {list_of_instances}\n"
            "Decide if each instance is actually an error or not. Return strictly valid JSON with the format:\n"
            f"{format_step_2}"
        )
        list_of_instances_filtered, log_filtered = self.task_execution(system_prompt, user_prompt_step_2)

        # --- Step 3
        rating_format = (
            '{\n'
            '  "reasoning": "<your reasoning>",\n'
            '  "confidence": <0-10>,\n'
            '  "rating": <0-5>\n'
            '}'
        )
        user_prompt_step_3 = (
            f"Step 3: Rate the summary considering the filtered error instances.\n"
            f"Criteria: *** {criteria}: {self.criteria[criteria]} ***\n"
            f"Summary: ** {data['summary']} **\n"
            f"Transcript: ** {data['transcript']} **\n"
            f"Error instances: {list_of_instances_filtered}\n"
            "Rate how badly these errors affect the summary on a scale 0-5 (0 = no impact, 5 = severe). "
            "Also provide a short reasoning and a confidence score (0-10). "
            f"Return strictly valid JSON in this format:\n{rating_format}"
        )

        # If we have any in-context examples:
        if self.in_context_learning_samples.get(criteria):
            system_prompt += (
                "\nIn-context examples:\n"
                f"{self.in_context_learning_samples[criteria]}"
            )

        final_score, log_final = self.task_execution(system_prompt, user_prompt_step_3)

        logger.info("[MESA - EVAL PIPELINE] --> %s : rating=%s, confidence=%s", #, reasoning=%s \n\n",
            criteria,
            final_score.get("rating"),
            final_score.get("confidence"),
            #final_score.get("reasoning"),
        )

        protocol = {
            "instances": log_instances,
            "filter": log_filtered,
            "final": log_final,
        }
        return final_score, list_of_instances, list_of_instances_filtered, protocol

    def task_execution(self, system_prompt, user_prompt):
        """
        If single_model, do a direct GPT call; if not, use the Panel approach.
        """
        if self.single_model:
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response_raw = ModelHandler.call_model_with_retry(
                self.client, message, self.model_id, max_tokens=4000
            )
            response_text = response_raw.choices[0].message.content.strip()
            response_text = response_text.strip("```json").strip("```").strip()
            logger.info("Response: %r", response_text)

            try:
                parsed = json.loads(response_text)
                return parsed, parsed
            except json.JSONDecodeError:
                logger.warning("[MESA - EVAL PIPELINE] Could not parse JSON. Returning raw text.")
                return response_text, response_text

        else:
            # Multi-agent approach
            panel = Panel(self.client, self.model_id, self.multi_family)
            final_answer, protocol_log = panel.ask("brainstorming", system_prompt, user_prompt)
            return final_answer, protocol_log
