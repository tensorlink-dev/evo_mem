"""Simple Markdown Baseline Agent.

The dumbest possible memory baseline:
- Appends every task experience to a single markdown file
- Retrieves by reading the last N entries (rolling window)
- No embeddings, no similarity search, no fancy anything
"""

from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

from .base import BaseAgent, AgentState, AgentAction, ActionType
from ..memory import Memory, MemoryEntry, Retriever, ContextBuilder
from ..memory.retriever import RetrievalResult, RecencyRetriever
from ..llm import BaseLLM


class MarkdownMemory:
    """
    Stores everything in a single markdown file. That's it.

    Each entry is appended as a markdown section. On retrieval,
    we just read the last N sections (rolling window).
    """

    def __init__(self, path: str = "memory.md", window_size: int = 4):
        self.path = Path(path)
        self.window_size = window_size
        self._entries: List[str] = []

        # Load existing file if present
        if self.path.exists():
            self._load()

    def _load(self):
        """Load entries from markdown file by splitting on entry headers."""
        text = self.path.read_text()
        if not text.strip():
            return
        # Split on entry delimiters
        chunks = text.split("\n---\n")
        self._entries = [c.strip() for c in chunks if c.strip()]

    def _save(self):
        """Write all entries back to the markdown file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n---\n".join(self._entries) + "\n")

    def append(
        self,
        task_id: str,
        question: str,
        answer: str,
        correct_answer: Optional[str] = None,
        feedback: Optional[str] = None,
        is_correct: bool = False,
    ):
        """Append a new entry to the markdown file."""
        lines = [
            f"## Task {task_id}",
            f"**Time:** {datetime.now().isoformat()}",
            f"**Result:** {'CORRECT' if is_correct else 'WRONG'}",
            "",
            f"**Q:** {question}",
            "",
            f"**A:** {answer}",
        ]
        if correct_answer:
            lines.append(f"\n**Correct:** {correct_answer}")
        if feedback:
            lines.append(f"\n**Feedback:** {feedback}")

        self._entries.append("\n".join(lines))
        self._save()

    def get_window(self) -> str:
        """Return the last N entries as raw markdown text."""
        window = self._entries[-self.window_size:]
        if not window:
            return ""
        return "\n---\n".join(window)

    def __len__(self):
        return len(self._entries)


class MarkdownBaselineAgent(BaseAgent):
    """
    Dead-simple markdown baseline agent.

    Memory = a markdown file.
    Retrieval = last N entries (rolling window).
    Context = dump the raw markdown into the prompt.
    No embeddings. No search. Just vibes.
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[Memory] = None,
        retriever: Optional[Retriever] = None,
        window_size: int = 4,
        markdown_path: str = "memory.md",
        **kwargs,
    ):
        # We still need a retriever for the base class, use a no-op recency one
        retriever = retriever or RecencyRetriever()
        from ..memory.context import SimpleContextBuilder
        context_builder = SimpleContextBuilder()

        super().__init__(
            llm=llm,
            retriever=retriever,
            context_builder=context_builder,
            memory=memory,
            top_k=window_size,
            **kwargs,
        )

        self.md_memory = MarkdownMemory(
            path=markdown_path,
            window_size=window_size,
        )

    def _build_prompt(self, query: str) -> str:
        """Build prompt by stuffing the rolling window into context."""
        window = self.md_memory.get_window()

        if not window:
            return query

        return f"""Here are your notes from previous tasks:

{window}

---

Now answer the following:
{query}"""

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

        # Build prompt with rolling window context
        prompt = self._build_prompt(query)

        # Generate
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=kwargs.get("system_prompt"),
        )
        output = self.extract_answer(response.content)

        # Update state
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

        # Append to markdown file
        self.md_memory.append(
            task_id=task_id,
            question=query,
            answer=output,
            correct_answer=kwargs.get("correct_answer"),
            feedback=state.feedback,
            is_correct=state.is_successful,
        )

        # Also store in the in-memory Memory object so the framework
        # can track stats consistently
        entry = MemoryEntry(
            task_id=task_id,
            input_text=query,
            output_text=output,
            feedback=state.feedback,
            is_successful=state.is_successful,
        )
        self.memory.add(entry)

        return output, state

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

            # Build context with markdown window + current history
            window = self.md_memory.get_window()
            history_lines = []
            for i, obs in enumerate(state.observations):
                history_lines.append(f"Observation: {obs}")
                if i < len(state.action_history):
                    history_lines.append(f"Action: {state.action_history[i].content}")

            parts = []
            if window:
                parts.append(f"Notes from previous tasks:\n{window}")
            parts.append(f"Goal: {goal}")
            parts.append("History:\n" + "\n".join(history_lines[-20:]))
            parts.append("What action should you take next? Respond with 'Action: <your action>'")

            prompt = "\n\n".join(parts)
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
        state.final_output = state.action_history[-1].content if state.action_history else ""

        if success:
            self.successful_tasks += 1

        # Append to markdown
        self.md_memory.append(
            task_id=task_id,
            question=goal,
            answer=state.final_output,
            feedback=state.feedback,
            is_correct=success,
        )

        # Also store in Memory for framework stats
        entry = MemoryEntry(
            task_id=task_id,
            input_text=goal,
            output_text=state.final_output,
            feedback=state.feedback,
            is_successful=success,
        )
        self.memory.add(entry)

        return success, progress, state
