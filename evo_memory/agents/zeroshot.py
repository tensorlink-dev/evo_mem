"""Zero-shot baseline agent.

No memory, no retrieval — just sends the query directly to the LLM.
This is the simplest possible baseline for measuring how much
memory-augmented agents improve over raw LLM performance.
"""

from typing import Tuple, Optional

from .base import BaseAgent, AgentState, AgentAction, ActionType
from ..memory import Memory, Retriever, ContextBuilder
from ..memory.retriever import EmbeddingRetriever
from ..memory.context import SimpleContextBuilder
from ..llm import BaseLLM


class ZeroShotAgent(BaseAgent):
    """Zero-shot baseline: query the LLM with no memory context."""

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Optional[Retriever] = None,
        memory: Optional[Memory] = None,
        **kwargs,
    ):
        # Retriever and memory are required by BaseAgent but unused.
        # Accept and discard whatever the runner passes in.
        kwargs.pop("context_builder", None)
        super().__init__(
            llm=llm,
            retriever=retriever or EmbeddingRetriever(),
            context_builder=SimpleContextBuilder(),
            memory=memory,
            top_k=0,
            store_successful_only=False,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Single-turn: just ask the LLM
    # ------------------------------------------------------------------

    def run_single_turn(
        self,
        task_id: str,
        query: str,
        **kwargs,
    ) -> Tuple[str, AgentState]:
        self.total_tasks += 1

        state = AgentState(
            task_id=task_id,
            input_text=query,
            memory=self.memory,
        )

        # No retrieval — send raw query
        response = self.llm.generate(
            prompt=query,
            system_prompt=kwargs.get("system_prompt"),
        )

        output = self.extract_answer(response.content)

        state.final_output = output
        state.is_complete = True
        state.action_history.append(AgentAction(
            action_type=ActionType.ACT,
            content=output,
        ))

        state.feedback = kwargs.get("feedback")
        state.is_successful = kwargs.get("is_correct", False)

        if state.is_successful:
            self.successful_tasks += 1

        # Still evolve so downstream analysis can inspect trajectories,
        # but there's nothing to retrieve from later.
        self.evolve(state)

        return output, state

    # ------------------------------------------------------------------
    # Multi-turn: act without memory context
    # ------------------------------------------------------------------

    def run_multi_turn(
        self,
        task_id: str,
        goal: str,
        environment,
        **kwargs,
    ) -> Tuple[bool, float, AgentState]:
        self.total_tasks += 1

        state = AgentState(
            task_id=task_id,
            input_text=goal,
            memory=self.memory,
        )

        observation = environment.reset()
        state.observations.append(observation)

        success = False
        progress = 0.0

        for step in range(self.max_steps):
            state.current_step = step

            # Build a minimal prompt from goal + recent history
            prompt = self._build_prompt(goal, state)
            response = self.llm.generate(prompt=prompt)
            action = self.parse_response(response.content)
            state.action_history.append(action)

            if action.action_type == ActionType.ACT:
                observation, reward, done, info = environment.step(action.content)
                state.observations.append(observation)

                if "progress" in info:
                    progress = info["progress"]

                if done:
                    success = info.get("success", False)
                    break

        state.is_complete = True
        state.is_successful = success
        state.feedback = "Success" if success else "Failure"
        state.final_output = (
            state.action_history[-1].content if state.action_history else ""
        )

        if success:
            self.successful_tasks += 1

        self.evolve(state)
        return success, progress, state

    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(goal, state):
        parts = [f"Goal: {goal}"]
        # Include last few observations/actions for context
        obs = state.observations
        acts = state.action_history
        for i, o in enumerate(obs):
            parts.append(f"Observation: {o}")
            if i < len(acts):
                parts.append(f"Action: {acts[i].content}")
        parts.append(
            "What action should you take next? Respond with 'Action: <your action>'"
        )
        return "\n\n".join(parts)
