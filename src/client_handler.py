#!/usr/bin/env python3
"""
client_factory.py

Defines a function that creates and returns an AzureOpenAI (or similar) client.
"""

import logging
import os

from openai import AzureOpenAI  # or your preferred client

logger = logging.getLogger(__name__)

def create_client() -> AzureOpenAI:
    """
    Creates and returns an AzureOpenAI client, reading API settings
    from environment variables. Adjust as needed for your environment.

    Returns:
        AzureOpenAI: An instance of AzureOpenAI client configured with your credentials.
    """
    api_key = os.getenv("API_KEY")
    api_version = os.getenv("API_VERSION")
    endpoint = os.getenv("ENDPOINT")
    model_name = os.getenv("MODEL_NAME")

    logger.info("Using model: %s with endpoint %s", model_name, endpoint)

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )
    return client
