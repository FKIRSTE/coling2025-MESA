"""
Panel module for multi-agent brainstorming or conclusion tasks,
but now only using GPT-based calls (no LLaMA, Gemini, or Phi).
"""

import logging
import json
from model_enums import ModelType

from model_handler import ModelHandler

logger = logging.getLogger(__name__)


class Panel:
    """
    A multi-agent panel that can brainstorm or reach conclusions about a given prompt.
    Currently only GPT-based calls are included.
    """

    def __init__(self, client, model_id, multi_family, size=3):
        """
        Args:
            client: The OpenAI-compatible client (e.g., AzureOpenAI).
            model_id (str): GPT model name, e.g., "gpt-3.5-turbo".
            multi_family (bool): If True, might expand to multiple GPT versions in the future.
            size (int): Number of "agents" in the panel.
        """
        self.size = size
        self.agents = []
        self.history = []
        self.tmp_agents = []
        self.client = client
        self.model_id = model_id

        self._set_up_panel()

    def _set_up_panel(self):
        """
        Initialize 'self.size' GPT-based agents.
        """
        for count in range(self.size):
            agent = {
                "id": count,
                "model": ModelType.GPT4o,  # We only have GPT
                "memory": [],
                "current_thought": "",
            }
            self.agents.append(agent)

    def process(self, agent, system_prompt, user_prompt, interprete_literal=True):
        """
        Run a single agent's GPT call.

        Args:
            agent (dict): Dictionary containing agent info.
            system_prompt (str): The system-level instruction for GPT.
            user_prompt (str): The user-level content prompt.
            interprete_literal (bool): Whether to parse output strictly as JSON, etc.

        Returns:
            The GPT model's response. Possibly JSON-loaded if interprete_literal == True.
        """
        return self.call_gpt(system_prompt, user_prompt, interprete_literal)

    def ask(self, task, system_prompt, user_prompt):
        """
        Main entry point for different tasks: brainstorming or conclusion.

        Args:
            task (str): "brainstorming" or "conclusion"
            system_prompt (str): The system prompt/instruction for GPT.
            user_prompt (str): The user content prompt.

        Returns:
            A tuple of (final_answer, protocol_log).
        """
        if task == "brainstorming":
            return self.seek_brainstorming(system_prompt, user_prompt)
        elif task == "conclusion":
            return self.seek_conclusion(system_prompt, user_prompt)
        else:
            logger.warning("[MESA - PANEL] Unknown task: %s", task)
            return "", []

    def seek_brainstorming(self, system_prompt, user_prompt, rounds=1):
        """
        Multi-step brainstorming approach.
        One "moderator" agent + the main agents collectively refine a draft.
        """
        
        protocol_log = []

        # Temporary moderator agent
        mod_agent = {
            "id": 0,
            "model": ModelType.GPT4o,
            "memory": [],
            "current_thought": "",
        }
        self.tmp_agents.append(mod_agent)

        # Initial draft by moderator
        initial_draft = self.process(mod_agent, system_prompt, user_prompt, interprete_literal=False)
        protocol_log.append({
            "stage": "Initial Draft",
            "agent": "Moderator",
            "output": initial_draft
        })

        # Agents refine the initial draft
        agent_system_prompt = (
            f"Following you will work on this task: *{system_prompt}* "
            "You will be given an initial draft and you should challenge it and possibly improve it."
        )
        agent_user_prompt = (
            f"This is your task: ***{user_prompt}***. "
            f"You are given a headstart: *{initial_draft}*. "
            "Consider it but generate your own improved version. "
            "Challenge the headstart draft if necessary. "
            "First write out what you think should be improved, then provide a new draft."
        )

        agent_feedback = []
        for round_num in range(rounds):
            for agent in self.agents:
                agent_output = self.process(agent, agent_system_prompt, agent_user_prompt, interprete_literal=False)
                agent_feedback.append(agent_output)
                protocol_log.append({
                    "stage": f"Agent Round {round_num + 1}, Agent {agent['id']}",
                    "agent": f"Agent {agent['id']}",
                    "output": agent_output
                })

        # Moderator final draft
        logging.info("[MESA - PANEL] Finalize draft.")
        
        finalize_system_prompt = (
            f"Task: *{system_prompt}*. "
            "Now consolidate the feedback from the other agents and produce a final draft."
        )
        finalize_user_prompt = (
            f"This is your task: ***{user_prompt}***. "
            f"Consider the original draft *{initial_draft}* and the revisions: **{agent_feedback}**. "
            "Produce a final, consolidated version of the draft."
        )

        final_draft = self.process(mod_agent, finalize_system_prompt, finalize_user_prompt, interprete_literal=True)
        protocol_log.append({
            "stage": "Final Consolidation",
            "agent": "Moderator",
            "output": final_draft
        })

        logger.debug("[MESA - PANEL] Brainstorming Protocol\n%s\n", protocol_log)
        return final_draft, protocol_log

    def seek_conclusion(self, system_prompt, user_prompt, rounds=1):
        """
        Example approach for drawing a conclusion after multiple agents have weighed in.
        This is a placeholder; expand as needed.
        """
        
        logging.info(f"[MESA - PANEL] Seeking conclusion with {rounds} rounds")
        
        protocol_log = []
        agent_system_prompt = (
            "You are an expert in summarizing meetings and are tasked with evaluating the quality of the summary."
            "Score the summary with a likert scale between 1 (worst) and 10 (best)."
        )
        agent_user_prompt = user_prompt

        # Each agent provides a score
        for round_num in range(rounds):
            for agent in self.agents:
                output = self.process(agent, agent_system_prompt, agent_user_prompt)
                protocol_log.append({
                    "stage": f"Scoring Round {round_num + 1}",
                    "agent": f"Agent {agent['id']}",
                    "output": output
                })

        # A final "moderator" step
        if not self.tmp_agents:
            self.tmp_agents.append({
                "id": 0,
                "model": ModelType.GPT4o,
                "memory": [],
                "current_thought": "",
            })
        mod_agent = self.tmp_agents[0]

        finalize_system_prompt = system_prompt + (
            "\nNow that the agents have provided scores, please finalize the conclusion."
        )
        finalize_user_prompt = user_prompt + f"\n Agents' feedback: {protocol_log}"

        final_answer = self.process(mod_agent, finalize_system_prompt, finalize_user_prompt)
        protocol_log.append({
            "stage": "Final Conclusion",
            "agent": "Moderator",
            "output": final_answer
        })

        logger.info("[MESA - PANEL] Conclusion Protocol\n%s\n", protocol_log)
        return final_answer, protocol_log

    def call_gpt(self, system_prompt, user_prompt, interprete_literal=True):
        """
        Single GPT call. Optionally interpret the response as JSON.
        """
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        logging.debug("[MESA - PANEL] Calling GPT with message: %s", message)
        
        response_raw = ModelHandler.call_model_with_retry(
            self.client, message, self.model_id, max_tokens=4000, identifier="gpt"
        )
        raw_text = response_raw.choices[0].message.content.strip()

        if interprete_literal:
            # Attempt to interpret as JSON
            raw_text = raw_text.strip("```json").strip("```").strip()
            logger.debug("[MESA - PANEL] GPT response: <%s>", raw_text)
            return_json = None
            try:
                return_json = json.loads(raw_text)
            except Exception as exc:
                logger.warning("[MESA - PANEL] Could not parse JSON: %s. Will proceed with unparsed string.", exc)
                return raw_text  # fallback to raw string
            return return_json

        return raw_text
