#!/usr/bin/env python3
"""
ModelHandler module that makes calls to OpenAI GPT-based models with retry logic.
"""

import time
import random
import logging

logger = logging.getLogger(__name__)

# Suppress OpenAI logs below WARNING level
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class ModelHandler:
    """
    Provides a static interface to call GPT-based models with retry-on-failure
    functionality (mainly for rate limit errors).
    """

    @staticmethod
    def call_model_with_retry(
        client,
        messages,
        model,
        max_tokens=1000,
        max_attempts=6,
        base_delay=3.0,
        identifier="gpt",
    ):
        """
        Attempts to call the GPT model up to `max_attempts` times, implementing
        exponential backoff if a rate limit (HTTP 429) error occurs.

        Args:
            client: Your OpenAI-compatible client (e.g., AzureOpenAI or openai).
            messages (list): The conversation messages, e.g. [{"role": "user", "content": ...}, ...].
            model (str): The name of the GPT model to use.
            max_tokens (int): The maximum number of tokens to generate.
            max_attempts (int): Maximum number of retry attempts on rate-limit errors.
            base_delay (float): Base delay in seconds for each retry iteration (exponential backoff).

        Returns:
            The response from the GPT model (or None if an error is encountered).
        """
        for attempt in range(max_attempts):
            try:
                # Main call
                response = ModelHandler._call_gpt(client, model, messages, max_tokens)
                logging.info("LLM API call was successful. (%s)", model)
                return response

            except Exception as exc:
                # Check if this is likely a rate limit error
                if "429" in str(exc):
                    sleep_time = (2 ** (attempt + 1)) + (random.randint(0, 1000) / 1000)
                    logger.warning(
                        "Rate limit hit (attempt %d). Backing off for %.2f seconds.",
                        attempt + 1,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error("Error encountered during GPT call: %s", str(exc))
                    break

            finally:
                time.sleep(base_delay)

        logger.error("Exceeded maximum retry attempts (%d). Returning None.", max_attempts)
        return None

    @staticmethod
    def _call_gpt(client, model, messages, max_tokens=1000):
        """
        Actual call to the GPT model. Customize parameters as needed.

        Args:
            client: Your OpenAI-compatible client (e.g., AzureOpenAI).
            model (str): The GPT model name (e.g., 'gpt-3.5-turbo').
            messages (list): The conversation messages.
            max_tokens (int): Max tokens to generate in the completion.

        Returns:
            The full response object from the model call.
        """
        logger.debug("### Calling GPT model '%s' ###", model)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            n=1,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response
