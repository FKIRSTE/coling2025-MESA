#!/usr/bin/env python3
"""
Main script for running MESA pipeline (no fitting).

Usage:
  python mesa.py
"""

import json
import logging
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from eval_pipeline import EvalPipeline
from baseline_pipeline import BaselinePipeline
from simple_baseline_pipeline import SimpleBaselinePipeline
from client_handler import create_client


def configure_logging():
    """ Set up your logging configuration. """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )




def load_config():
    """ Load environment variables and parse the JSON config file. """
    load_dotenv()
    config_path = os.getenv("CONFIG_PATH", "src/config.json")
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"CONFIG_PATH not found or invalid: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def init_pipeline(config):
    """ 
    Create and return the pipeline instance based on config.
    Returns:
        pipeline (object): An instance of one of our pipeline classes (Eval, Baseline, etc.)
        in_context_samples (str): Path to in-context samples (if any)
        logging_directory (str)
    """
    mode = config.get("MODE", 0)
    single_model = config.get("SINGLE_MODEL", True)
    multi_family = config.get("MULTI_FAMILY", False)

    # Pipelines
    mode_mapping = {
        0: {"postfix": "_ALL_AT_ONCE", "pipeline": SimpleBaselinePipeline},
        1: {"postfix": "_ONE_BY_ONE", "pipeline": BaselinePipeline},
        2: {"postfix": "_THREE_STEP", "pipeline": EvalPipeline},
    }

    pipeline_cls = mode_mapping[mode]["pipeline"]
    postfix = mode_mapping[mode]["postfix"]
    single_model_str = "SInstance" if single_model else "MInstance"
    multi_family_str = "SFamily" if multi_family else "MFamily"

    # Paths from environment
    criteria_path = os.getenv("CRITERIA_PATH", "criteria/")
    in_context_samples = os.getenv("IN_CONTEXT_LEARNING_SAMPLES", "report_generation/outputs/reports/feedback_3step.csv")
    logging_directory = os.getenv("LOGGING_DIRECTORY", "./_logs/")
    logging_directory = os.path.join(logging_directory, postfix)

    samples_path = os.getenv("SAMPLES_PATH", "HumanJudgment/MistekQMSum.csv")
    if not criteria_path:
        raise ValueError("CRITERIA_PATH is not set in environment variables.")
    if not samples_path or not os.path.exists(samples_path):
        raise FileNotFoundError(f"SAMPLES_PATH not found: {samples_path}")

    # Create OpenAI/Azure client
    client = create_client()
    model_name = os.getenv("MODEL_NAME", "gpt-4o")

    logging.info("[MESA] Using model: %s", model_name)
    logging.info("[MESA] Pipeline Mode: %d (%s)", mode, postfix)

    # Initialize the pipeline
    pipeline_instance = pipeline_cls(
        criteria_path,
        in_context_samples,
        client,
        model_name,
        single_model=single_model,
        multi_family=multi_family,
    )

    return pipeline_instance, in_context_samples, logging_directory, samples_path

def run_pipeline(config, pipeline_instance, samples_df, iteration_log_dir):
    """
    Runs the pipeline for each row in samples_df, collecting scores.
    Optionally slices the dataset according to START_INDEX / STOP_INDEX in config.
    
    Returns:
        scores (dict): pipeline outputs keyed by row index
        human_scores_all (list): list of human score dicts if DO_FITTING is enabled
    """
    logger = logging.getLogger(__name__)
    do_fitting = config.get("DO_FITTING", False)
    start_index = config.get("START_INDEX", 0)
    stop_index = config.get("STOP_INDEX", None)

    # Slice the dataframe if necessary
    if stop_index is None or stop_index <= 0:
        samples_df = samples_df.iloc[start_index:]
    else:
        samples_df = samples_df.iloc[start_index:stop_index]

    logger.info("[MESA] Processing rows from index %d to %d (end-exclusive).",
                start_index, stop_index or samples_df.shape[0])

    scores = {}
    human_scores_all = []

    for index, row in samples_df.iterrows():
        sample = {
            "summary": row["Predicted"],
            "transcript": row["Input"],
        }
        # Pipeline run
        result = pipeline_instance.run(sample)
        scores[index] = result

        # Optionally gather human scores
        if do_fitting:
            human_scores_row = {
                    "repetition" : {
                    "existence": row["Redundancy - Existence"],
                    "reasoning": row["Redundancy - Reasoning"],
                    "impact": row["Redundancy - Impact"]
                },
                "incoherence" : {
                    "existence": row["Incoherence - Existence"],
                    "reasoning": row["Incoherence - Reasoning"],
                    "impact": row["Incoherence - Impact"]
                },
                "language" : {
                    "existence": row["Language - Existence"],
                    "reasoning": row["Language - Reasoning"],
                    "impact": row["Language - Impact"]
                },
                "omission" : {
                    "existence": row["Omission - Existence"],
                    "reasoning": row["Omission - Reasoning"],
                    "impact": row["Omission - Impact"]
                },
                "coreference" : {
                    "existence": row["Coreference - Existence"],
                    "reasoning": row["Coreference - Reasoning"],
                    "impact": row["Coreference - Impact"]
                },
                "hallucination" : {
                    "existence": row["Hallucination - Existence"],
                    "reasoning": row["Hallucination - Reasoning"],
                    "impact": row["Hallucination - Impact"]
                },
                "structure" : {
                    "existence": row["Structur - Existence"],
                    "reasoning": row["Structure - Reasoning"],
                    "impact": row["Structure - Impact"]
                },
                "irrelevance" : {
                    "existence": row["Irrelevance - Existence"],
                    "reasoning": row["Irrelevance - Reasoning"],
                    "impact": row["Irrelevance - Impact"]
                }
            }
            human_scores_all.append(human_scores_row)

        # Write each sample’s result (and optional human scores) to JSON
        combined_data = {"score": result}
        if do_fitting:
            combined_data["human_scores"] = human_scores_row

        file_name = os.path.join(iteration_log_dir, f"sample_{index}.json")
        with open(file_name, "w", encoding="utf-8") as log_file:
            json.dump(combined_data, log_file, indent=4)
        logger.info("[MESA] Logged sample %d to %s", index, file_name)

    return scores, human_scores_all

def fitting_stage(config, scores, human_scores, iteration_log_dir):
    """
    Performs the optional fitting comparison stage if 'DO_FITTING' is True.
    """
        
    logger = logging.getLogger(__name__)
    if not config.get("DO_FITTING", False):
        logger.info("[MESA] Fitting stage disabled.")
        return


    logger.info("\n\n************************** \n [MESA - FITTING] Number of samples: %s \n**************************\n", len(scores))


    logger.info("[MESA] Fitting stage enabled, starting comparison.")
    from fitting import Fitting  # import here or at top
    from client_handler import create_client

    client = create_client()
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    criteria_path = os.getenv("CRITERIA_PATH")

    fitter = Fitting(client, model_name, criteria_path)
    learning_samples, distance_quality_per_criteria = fitter.run(scores, human_scores)

    # Save results
    closeness_file = os.path.join(iteration_log_dir, "total_closeness.json")
    with open(closeness_file, "w", encoding="utf-8") as log_file:
        json.dump(distance_quality_per_criteria, log_file, indent=4)
    logger.info("[MESA] Fitting results saved to %s", closeness_file)

    return learning_samples

def self_training_stage(config, learning_samples, in_context_samples_path):
    """
    If SELF_TRAINING is set, performs the logic to update or generate
    new in-context learning files, etc.
    """
        
    logger = logging.getLogger(__name__)
    if not config.get("SELF_TRAINING", False):
        logger.info("[MESA] Self-training disabled.")
        return


    logger.info("\n\n************************** \n [MESA - SELF TRAINING] Considering samples: %s \n**************************\n", len(learning_samples))


    logger.info("[MESA] Self-training is enabled.")
    
    if not os.path.exists(in_context_samples_path):
        os.makedirs(in_context_samples_path)

    # Step 1: Group data by error type and save to individual files
    grouped_data = {}

    for entry in learning_samples:
        error_type = entry['criteria']
        
        if error_type not in grouped_data:
            grouped_data[error_type] = []
        
        grouped_data[error_type].append(entry)

    # Step 2: Save each error type data to a separate file
    for error_type, error_entries in grouped_data.items():
        file_path = os.path.join(in_context_samples_path, f"{error_type}_fss.json")
        
        with open(file_path, 'w') as file:
            json.dump(error_entries, file, indent=4)

    # Now each file contains the grouped error types, you can load them later.



def main():
    """
    Orchestrates MESA pipeline + optional fitting/comparison.
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    # Load config and environment
    config = load_config()

    # Prepare pipeline
    pipeline_instance, in_context_samples, logging_directory, samples_path = init_pipeline(config)

    # Load dataset
    samples_df = pd.read_csv(samples_path)
    logger.info("[MESA] Loaded %d samples from %s", len(samples_df), samples_path)

    # Ensure logging directory
    os.makedirs(logging_directory, exist_ok=True)

    iterations = config.get("ITERATIONS", 1)

    for iteration_idx in range(iterations):
        logger.info("[MESA] Starting iteration %d", (iteration_idx+1))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Unique directory for logs
        single_model_str = "SInstance" if config.get("SINGLE_MODEL", True) else "MInstance"
        multi_family_str = "SFamily" if config.get("MULTI_FAMILY", False) else "MFamily"

        iteration_log_dir = os.path.join(
            logging_directory,
            f"iteration_{iteration_idx}_{timestamp}_{single_model_str}_{multi_family_str}",
        )
        os.makedirs(iteration_log_dir, exist_ok=True)

        # Step 1: run pipeline, gather results
        scores, human_scores_all = run_pipeline(config, pipeline_instance, samples_df, iteration_log_dir)

        # Step 2 (optional): fitting
        learning_samples = fitting_stage(config, scores, human_scores_all, iteration_log_dir)

        # Step 3 (optional): self-training
        self_training_stage(config, learning_samples, in_context_samples)

        logger.info("[MESA] Iteration %d completed.", iteration_idx)


    logger.info("\n\n************************** \n [MESA] 🏁 All iterations finished. 🏁 \n**************************\n")

if __name__ == "__main__":
    main()
