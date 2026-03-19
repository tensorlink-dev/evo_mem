"""GanglionAgent — biological memory agent for Evo-Memory.

Integrates ganglion-memory's Hebbian learning system into the
Evo-Memory eval harness.  Runs in hybrid mode by default:

  * Before each task: queries ganglion for Hebbian beliefs
    ("what works" / "what fails") and injects them into the prompt,
    *alongside* standard embedding-retrieval from Evo-Memory.
  * After each task: feeds success/failure back into ganglion's
    ``assimilate()`` loop which handles strengthening, weakening,
    contradiction apoptosis, and lateral inhibition automatically.
  * Between streams: runs ganglion's ``forget()`` for consolidation
    and eviction.

Ganglion-memory is imported lazily so the rest of Evo-Memory works
even when the package is not installed.
"""

from typing import Tuple, Optional, List, Dict, Any
import logging

from .base import BaseAgent, AgentState, AgentAction, ActionType
from ..memory import Memory, MemoryEntry, Retriever, ContextBuilder
from ..memory.retriever import RetrievalResult
from ..memory.context import SimpleContextBuilder
from ..llm import BaseLLM

logger = logging.getLogger(__name__)


def _import_ganglion():
    """Lazy-import ganglion-memory and return the module."""
    try:
        import ganglion_memory
        return ganglion_memory
    except ImportError:
        raise ImportError(
            "ganglion-memory package is required for GanglionAgent. "
            "Install with: pip install ganglion-memory"
        )


class GanglionAgent(BaseAgent):
    """
    Ganglion biological-memory agent.

    Uses Hebbian strengthening, apoptosis, lateral inhibition,
    salience gating, and consolidation instead of simple
    vector-retrieval.

    Operates in hybrid mode (default):
      - Ganglion handles *strategy-level* knowledge
        ("batch size 32 works better")
      - Evo-Memory embedding retrieval handles *example-level*
        knowledge ("here's a similar task I solved")

    Set ``hybrid_mode=False`` to use ganglion beliefs only
    (useful as an ablation).
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        context_builder: Optional[ContextBuilder] = None,
        memory: Optional[Memory] = None,
        top_k: int = 4,
        max_steps: int = 50,
        hybrid_mode: bool = True,
        db_path: str = ":memory:",
        capability: str = "general problem-solving",
        max_beliefs_in_prompt: int = 10,
        include_trajectory: bool = True,
        include_feedback: bool = True,
        **kwargs,
    ):
        """
        Initialize GanglionAgent.

        Args:
            llm: Base LLM for generation
            retriever: Embedding retriever for similarity search
            context_builder: Optional context builder (SimpleContextBuilder default)
            memory: Optional pre-initialized Evo-Memory store
            top_k: Number of experiences to retrieve from embedding memory
            max_steps: Maximum steps for multi-turn tasks
            hybrid_mode: Use both ganglion beliefs AND embedding retrieval
            db_path: SQLite path for ganglion (":memory:" for in-memory)
            capability: Description of the agent's capability for ganglion
            max_beliefs_in_prompt: Max beliefs to inject into prompt
            include_trajectory: Include trajectories in Evo-Memory context
            include_feedback: Include feedback in Evo-Memory context
        """
        context_builder = context_builder or SimpleContextBuilder(
            include_trajectory=include_trajectory,
            include_feedback=include_feedback,
        )

        # Ganglion learns from failures too
        store_successful_only = kwargs.pop("store_successful_only", False)

        super().__init__(
            llm=llm,
            retriever=retriever,
            context_builder=context_builder,
            memory=memory,
            top_k=top_k,
            max_steps=max_steps,
            store_successful_only=store_successful_only,
            **kwargs,
        )

        self.hybrid_mode = hybrid_mode
        self.db_path = db_path
        self.capability = capability
        self.max_beliefs_in_prompt = max_beliefs_in_prompt
        self.include_trajectory = include_trajectory
        self.include_feedback = include_feedback

        # Ganglion metrics (tracked per-stream for eval output)
        self.ganglion_metrics: List[Dict[str, Any]] = []

        # Initialise ganglion memory system
        self._ganglion = None
        self._init_ganglion()

    def _init_ganglion(self):
        """Initialise the ganglion-memory system."""
        gm = _import_ganglion()
        self._ganglion = gm.GanglionMemory(
            db_path=self.db_path,
            capability=self.capability,
        )

    # ------------------------------------------------------------------
    # Ganglion query helpers
    # ------------------------------------------------------------------

    def _query_beliefs(self, query: str) -> Dict[str, List[str]]:
        """
        Query ganglion for relevant beliefs.

        Returns dict with 'works' and 'fails' lists.
        """
        beliefs = {"works": [], "fails": []}
        try:
            result = self._ganglion.recall(query)
            for belief in result:
                text = belief.get("text", str(belief))
                confidence = belief.get("confidence", 0.5)
                if confidence >= 0.5:
                    beliefs["works"].append(text)
                else:
                    beliefs["fails"].append(text)
        except Exception as e:
            logger.debug(f"Ganglion recall failed (may be empty): {e}")
        return beliefs

    def _format_beliefs_context(self, beliefs: Dict[str, List[str]]) -> str:
        """Format ganglion beliefs into a prompt section."""
        parts = []
        if beliefs["works"]:
            items = beliefs["works"][:self.max_beliefs_in_prompt]
            parts.append("STRATEGIES THAT HAVE WORKED:")
            for i, b in enumerate(items, 1):
                parts.append(f"  {i}. {b}")
        if beliefs["fails"]:
            items = beliefs["fails"][:self.max_beliefs_in_prompt]
            parts.append("STRATEGIES THAT HAVE FAILED:")
            for i, b in enumerate(items, 1):
                parts.append(f"  {i}. {b}")
        return "\n".join(parts)

    def _assimilate(self, query: str, output: str, success: bool):
        """Feed outcome back into ganglion for Hebbian learning."""
        try:
            self._ganglion.assimilate(
                task=query,
                response=output,
                success=success,
            )
        except Exception as e:
            logger.warning(f"Ganglion assimilate failed: {e}")

    def _snapshot_metrics(self, task_idx: int, beliefs_injected: bool):
        """Capture ganglion-specific metrics at this task step."""
        metrics = {
            "task_idx": task_idx,
            "beliefs_injected": beliefs_injected,
        }
        try:
            stats = self._ganglion.stats()
            metrics["num_beliefs"] = stats.get("num_beliefs", 0)
            metrics["num_contradictions"] = stats.get("num_contradictions", 0)
            metrics["num_apoptosis"] = stats.get("num_apoptosis", 0)
            metrics["confidence_distribution"] = stats.get(
                "confidence_distribution", []
            )
        except Exception:
            metrics["num_beliefs"] = 0
            metrics["num_contradictions"] = 0
            metrics["num_apoptosis"] = 0
            metrics["confidence_distribution"] = []
        self.ganglion_metrics.append(metrics)

    # ------------------------------------------------------------------
    # Single-turn
    # ------------------------------------------------------------------

    def run_single_turn(
        self,
        task_id: str,
        query: str,
        **kwargs,
    ) -> Tuple[str, AgentState]:
        """
        Run GanglionAgent on a single-turn task.

        Process:
        1. Query ganglion for Hebbian beliefs
        2. (Hybrid) Retrieve similar experiences from Evo-Memory
        3. Build combined context and generate output
        4. Feed result back into ganglion
        5. Store experience in Evo-Memory for future retrieval
        """
        self.total_tasks += 1
        task_idx = self.total_tasks

        # Initialize state
        state = AgentState(
            task_id=task_id,
            input_text=query,
            memory=self.memory,
        )

        # 1. Query ganglion beliefs
        beliefs = self._query_beliefs(query)
        beliefs_context = self._format_beliefs_context(beliefs)
        beliefs_injected = bool(beliefs_context)

        # 2. Hybrid: also retrieve from Evo-Memory embedding store
        evo_context = ""
        if self.hybrid_mode:
            state.retrieved = self.search(query)
            if state.retrieved:
                evo_context = self.synthesize(query, state.retrieved)

        # 3. Build combined prompt
        prompt = self._build_single_turn_prompt(
            query, beliefs_context, evo_context,
        )

        # 4. Generate
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=kwargs.get("system_prompt"),
        )
        output = self.extract_answer(response.content)

        # 5. Update state
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

        # 6. Feed into ganglion
        self._assimilate(query, output, state.is_successful)

        # 7. Store in Evo-Memory (evolve)
        self.evolve(state)

        # 8. Snapshot ganglion metrics
        self._snapshot_metrics(task_idx, beliefs_injected)

        return output, state

    def _build_single_turn_prompt(
        self,
        query: str,
        beliefs_context: str,
        evo_context: str,
    ) -> str:
        """Build the combined prompt for single-turn tasks."""
        parts = []

        if beliefs_context:
            parts.append(
                "==================================================\n"
                "BIOLOGICAL MEMORY — LEARNED STRATEGIES (GANGLION)\n"
                "==================================================\n"
                f"{beliefs_context}"
            )

        if evo_context:
            parts.append(evo_context)
        else:
            parts.append(
                "==================================================\n"
                "YOUR CURRENT TASK\n"
                "==================================================\n"
                f"{query}"
            )

        if not beliefs_context and not evo_context:
            parts = [query]

        parts.append(
            "\nProvide your output in the following format:\n"
            "- Rationale: your short reasoning, may cite learned strategies\n"
            "- Final Answer: your final answer"
        )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Multi-turn
    # ------------------------------------------------------------------

    def run_multi_turn(
        self,
        task_id: str,
        goal: str,
        environment,
        **kwargs,
    ) -> Tuple[bool, float, AgentState]:
        """
        Run GanglionAgent on a multi-turn task.

        Same hybrid approach: ganglion beliefs + Evo-Memory retrieval
        injected at each step.
        """
        self.total_tasks += 1
        task_idx = self.total_tasks

        state = AgentState(
            task_id=task_id,
            input_text=goal,
            memory=self.memory,
        )

        observation = environment.reset()
        state.observations.append(observation)

        # Initial retrieval
        if self.hybrid_mode:
            state.retrieved = self.search(goal)

        success = False
        progress = 0.0

        for step in range(self.max_steps):
            state.current_step = step

            # Query ganglion
            beliefs = self._query_beliefs(goal)
            beliefs_context = self._format_beliefs_context(beliefs)

            # Build context
            context = self._build_multi_turn_context(
                goal=goal,
                beliefs_context=beliefs_context,
                retrieved=state.retrieved,
                observations=state.observations,
                action_history=state.action_history,
                environment_info=kwargs.get("environment_info", ""),
            )

            response = self.llm.generate(prompt=context)
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

        # Final state
        state.is_complete = True
        state.is_successful = success
        state.feedback = "Success" if success else "Failure"
        state.final_output = (
            state.action_history[-1].content if state.action_history else ""
        )

        if success:
            self.successful_tasks += 1

        # Feed into ganglion
        self._assimilate(goal, state.feedback, success)

        # Evolve Evo-Memory
        self.evolve(state)

        # Snapshot metrics
        self._snapshot_metrics(task_idx, bool(beliefs_context))

        return success, progress, state

    def _build_multi_turn_context(
        self,
        goal: str,
        beliefs_context: str,
        retrieved: List[RetrievalResult],
        observations: List[str],
        action_history: List[AgentAction],
        environment_info: str = "",
    ) -> str:
        """Build context for multi-turn tasks."""
        parts = []

        if environment_info:
            parts.append(f"Environment:\n{environment_info}")

        if beliefs_context:
            parts.append(
                "Learned Strategies (Ganglion):\n" + beliefs_context
            )

        if self.hybrid_mode and retrieved:
            exp_parts = []
            for i, result in enumerate(retrieved):
                entry = result.entry
                exp_text = entry.to_text(
                    include_trajectory=self.include_trajectory,
                    include_feedback=self.include_feedback,
                )
                exp_parts.append(f"[Experience #{i + 1}]\n{exp_text}")
            parts.append("Similar Experiences:\n" + "\n\n".join(exp_parts))

        parts.append(f"Goal: {goal}")

        if observations:
            history_parts = []
            for i, obs in enumerate(observations):
                history_parts.append(f"Observation: {obs}")
                if i < len(action_history):
                    history_parts.append(f"Action: {action_history[i].content}")
            parts.append("History:\n" + "\n".join(history_parts[-20:]))

        parts.append(
            "What action should you take next? "
            "Respond with 'Action: <your action>'"
        )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def consolidate(self):
        """
        Run ganglion's consolidation (sleep phase).

        Call this between streams to merge similar beliefs
        and evict weak ones.
        """
        try:
            self._ganglion.forget()
        except Exception as e:
            logger.warning(f"Ganglion consolidation failed: {e}")

    def reset_memory(self) -> None:
        """Reset both Evo-Memory and ganglion."""
        super().reset_memory()
        self.ganglion_metrics.clear()
        self._init_ganglion()

    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics."""
        stats = super().get_statistics()
        stats["ganglion_metrics"] = self.ganglion_metrics
        try:
            stats["ganglion_stats"] = self._ganglion.stats()
        except Exception:
            stats["ganglion_stats"] = {}
        return stats
