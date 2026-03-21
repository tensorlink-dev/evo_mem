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
import asyncio
import logging

from .base import BaseAgent, AgentState, AgentAction, ActionType
from ..memory import Memory, MemoryEntry, Retriever, ContextBuilder
from ..memory.retriever import RetrievalResult
from ..memory.context import SimpleContextBuilder
from ..llm import BaseLLM

logger = logging.getLogger(__name__)


def _extract_entities(text: str) -> tuple:
    """Extract capitalised terms from text as ganglion entities."""
    words = text.split()
    entities = [w.strip(".,;:!?") for w in words if len(w) > 2 and w[0].isupper()]
    return tuple(dict.fromkeys(entities))[:5]


def _run_sync(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _import_ganglion():
    """Lazy-import ganglion and return the memory sub-package."""
    try:
        from ganglion import memory as gm
        return gm
    except ImportError:
        raise ImportError(
            "ganglion package is required for GanglionAgent. "
            "Install with: pip install git+https://github.com/TensorLink-AI/ganglion-memory.git"
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
        relevance_threshold: float = 0.3,
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
        # Only forward kwargs that BaseAgent accepts
        max_iterations = kwargs.pop("max_iterations", 10)

        super().__init__(
            llm=llm,
            retriever=retriever,
            context_builder=context_builder,
            memory=memory,
            top_k=top_k,
            max_steps=max_steps,
            max_iterations=max_iterations,
            store_successful_only=store_successful_only,
        )

        self.hybrid_mode = hybrid_mode
        self.db_path = db_path
        self.capability = capability
        self.max_beliefs_in_prompt = max_beliefs_in_prompt
        self.include_trajectory = include_trajectory
        self.include_feedback = include_feedback
        self.relevance_threshold = relevance_threshold

        # Ganglion metrics (tracked per-stream for eval output)
        self.ganglion_metrics: List[Dict[str, Any]] = []

        # Initialise ganglion memory system
        self._init_ganglion()

    def _init_ganglion(self):
        """Initialise the ganglion memory system."""
        gm = _import_ganglion()
        backend = gm.SqliteMemoryBackend(self.db_path)

        # Reuse evo_mem's retriever model as ganglion's embedder via
        # CallableEmbedder so we don't load a second sentence-transformer.
        embedder = None
        if hasattr(self.retriever, "encode"):
            async def _embed_fn(text: str) -> list:
                return await asyncio.to_thread(self.retriever.encode, text)
            embedder = gm.CallableEmbedder(_embed_fn)

        loop_kwargs: Dict[str, Any] = {
            "backend": backend,
            "embedder": embedder,
            "relevance_threshold": self.relevance_threshold,
        }
        self._ganglion_loop = gm.MemoryLoop(**loop_kwargs)
        self._ganglion_agent = gm.MemoryAgent(
            memory=self._ganglion_loop,
            capability=self.capability,
            bot_id="evo_mem",
            context_limit=self.max_beliefs_in_prompt,
        )

    # ------------------------------------------------------------------
    # Ganglion query helpers
    # ------------------------------------------------------------------

    def _query_beliefs(self, query: str) -> str:
        """Query ganglion for relevant beliefs via embedding-ranked retrieval."""
        try:
            self._ganglion_agent.entities = _extract_entities(query)
            return _run_sync(self._ganglion_agent.remember(query=query))
        except Exception as e:
            logger.debug(f"Ganglion recall failed (may be empty): {e}")
            return ""

    def _format_beliefs_context(self, beliefs: str) -> str:
        """Format ganglion beliefs into a prompt section."""
        return beliefs.strip() if beliefs else ""

    def _assimilate(
        self,
        query: str,
        output: str,
        success: bool,
        *,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        tags: tuple = (),
    ):
        """Feed outcome into ganglion via MemoryAgent.learn().

        Uses learn() instead of raw MemoryLoop.assimilate() so that
        the observation is enriched with produced_with (dependency
        chain), input_text, and output_text automatically.
        """
        try:
            self._ganglion_agent.entities = _extract_entities(query)
            if tags:
                self._ganglion_agent.tags = tags

            result: Dict[str, Any] = {
                "success": success,
                "description": output[:500],
            }
            if metric_name is not None:
                result["metric_name"] = metric_name
            if metric_value is not None:
                result["metric_value"] = metric_value

            _run_sync(self._ganglion_agent.learn(
                result,
                input_text=query[:500],
                output_text=output[:500],
            ))
        except Exception as e:
            logger.warning(f"Ganglion learn failed: {e}")

    def _snapshot_metrics(self, task_idx: int, beliefs_injected: bool):
        """Capture ganglion-specific metrics at this task step."""
        metrics = {
            "task_idx": task_idx,
            "beliefs_injected": beliefs_injected,
        }
        try:
            stats = _run_sync(self._ganglion_loop.summary())
            metrics.update(stats)
        except Exception:
            pass
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
        self._assimilate(
            query, output, state.is_successful,
            tags=("single_turn",),
        )

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

                # Per-step assimilation: teach ganglion which actions
                # produce reward so beliefs form during the task
                if reward != 0:
                    self._assimilate(
                        query=f"{goal} | action: {action.content}",
                        output=observation,
                        success=reward > 0,
                        metric_name="step_reward",
                        metric_value=float(reward),
                        tags=("multi_turn", "step"),
                    )

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

        # Feed final outcome into ganglion
        self._assimilate(
            goal, state.feedback, success,
            metric_name="task_progress",
            metric_value=progress,
            tags=("multi_turn", "outcome"),
        )

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
            _run_sync(self._ganglion_loop.forget())
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
            stats["ganglion_stats"] = _run_sync(self._ganglion_loop.summary())
        except Exception:
            stats["ganglion_stats"] = {}
        return stats
