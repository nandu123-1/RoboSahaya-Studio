import json
from pathlib import Path
from typing import Dict, List, Optional
from robosahaya.evidence_models import TaskEvent, TaskStage

class TaskStore:
    """Stores ordered task events and lifecycle state."""

    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = (store_dir or Path("/tmp/robosahaya_tasks")).resolve()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._memory_events: Dict[str, List[TaskEvent]] = {}
        self._memory_states: Dict[str, TaskStage] = {}

    def record_event(self, event: TaskEvent):
        """Records an event in memory and persists to disk."""
        tid = event.task_id
        if tid not in self._memory_events:
            self._memory_events[tid] = []
        self._memory_events[tid].append(event)
        self._memory_states[tid] = event.stage

        # Append to jsonl file
        task_file = self.store_dir / f"{tid}.jsonl"
        with open(task_file, "a") as f:
            f.write(event.model_dump_json() + "\n")

    def get_events(self, task_id: str) -> List[TaskEvent]:
        """Retrieves all ordered events for a task."""
        if task_id in self._memory_events:
            return self._memory_events[task_id]

        task_file = self.store_dir / f"{task_id}.jsonl"
        if not task_file.exists():
            return []

        events = []
        with open(task_file, "r") as f:
            for line in f:
                if line.strip():
                    events.append(TaskEvent.model_validate_json(line))
        self._memory_events[task_id] = events
        return events

    def get_current_stage(self, task_id: str) -> TaskStage:
        """Returns the current stage for a task."""
        if task_id in self._memory_states:
            return self._memory_states[task_id]
        events = self.get_events(task_id)
        if events:
            return events[-1].stage
        return TaskStage.RECEIVED
