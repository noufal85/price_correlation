"""Pipeline state management - track steps, cache intermediate results."""

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import get_cache, serialize_dataframe, deserialize_dataframe

logger = logging.getLogger(__name__)

# TTL for intermediate results (24 hours - longer than regular cache)
TTL_INTERMEDIATE = 86400

# Pipeline steps in order
PIPELINE_STEPS = [
    "universe",      # Step 1: Fetch stock universe
    "prices",        # Step 2: Fetch price data
    "preprocess",    # Step 3: Compute returns, normalize
    "correlation",   # Step 4: Compute correlation matrix
    "clustering",    # Step 5: Run clustering algorithm
    "export",        # Step 6: Export results
]

STEP_DESCRIPTIONS = {
    "universe": "Fetch Stock Universe",
    "prices": "Fetch Price Data",
    "preprocess": "Preprocess & Compute Returns",
    "correlation": "Compute Correlations",
    "clustering": "Run Clustering",
    "export": "Export Results",
}


@dataclass
class PipelineState:
    """Tracks pipeline execution state."""

    session_id: str = ""
    created_at: str = ""
    updated_at: str = ""

    # Configuration used
    config: dict = field(default_factory=dict)

    # Step completion status
    completed_steps: list[str] = field(default_factory=list)
    current_step: str = ""

    # Step results summary (not full data)
    step_results: dict = field(default_factory=dict)

    # Step timing info: {step_name: {"started_at": ..., "completed_at": ..., "duration": ...}}
    step_timing: dict = field(default_factory=dict)

    # Error tracking
    error: str = ""
    error_step: str = ""

    def is_step_complete(self, step: str) -> bool:
        return step in self.completed_steps

    def get_next_step(self) -> str | None:
        for step in PIPELINE_STEPS:
            if step not in self.completed_steps:
                return step
        return None

    def mark_step_complete(self, step: str, result_summary: dict | None = None):
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        self.current_step = ""
        self.updated_at = datetime.now().isoformat()
        if result_summary:
            self.step_results[step] = result_summary

        # Record completion time and calculate duration
        now = datetime.now()
        if step not in self.step_timing:
            self.step_timing[step] = {}
        self.step_timing[step]["completed_at"] = now.isoformat()

        # Calculate duration if we have start time
        if "started_at" in self.step_timing[step]:
            try:
                started = datetime.fromisoformat(self.step_timing[step]["started_at"])
                self.step_timing[step]["duration"] = (now - started).total_seconds()
            except (ValueError, TypeError):
                pass

    def mark_step_started(self, step: str):
        self.current_step = step
        self.updated_at = datetime.now().isoformat()

        # Record start time
        if step not in self.step_timing:
            self.step_timing[step] = {}
        self.step_timing[step]["started_at"] = datetime.now().isoformat()

    def mark_error(self, step: str, error: str):
        self.error = error
        self.error_step = step
        self.current_step = ""
        self.updated_at = datetime.now().isoformat()

    def clear_from_step(self, step: str):
        """Clear this step and all subsequent steps."""
        step_idx = PIPELINE_STEPS.index(step) if step in PIPELINE_STEPS else -1
        if step_idx >= 0:
            steps_to_remove = PIPELINE_STEPS[step_idx:]
            self.completed_steps = [s for s in self.completed_steps if s not in steps_to_remove]
            for s in steps_to_remove:
                self.step_results.pop(s, None)
                self.step_timing.pop(s, None)
        self.error = ""
        self.error_step = ""
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineState":
        return cls(**data)


class PipelineStateManager:
    """Manages pipeline state and intermediate results storage."""

    def __init__(self, session_id: str | None = None):
        """
        Initialize state manager.

        Args:
            session_id: Unique session identifier. If None, generates new one.
        """
        self.session_id = session_id or self._generate_session_id()
        self._state: PipelineState | None = None
        self._output_dir = Path("output")
        self._output_dir.mkdir(exist_ok=True)

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"session_{timestamp}_{rand}"

    def _state_key(self) -> str:
        return f"price_correlation:pipeline_state:{self.session_id}"

    def _data_key(self, step: str) -> str:
        return f"price_correlation:pipeline_data:{self.session_id}:{step}"

    @property
    def state(self) -> PipelineState:
        """Get current pipeline state."""
        if self._state is None:
            self._state = self.load_state()
        return self._state

    def _state_file_path(self) -> Path:
        """Get file path for state storage."""
        return self._output_dir / f".pipeline_state_{self.session_id}.json"

    def load_state(self) -> PipelineState:
        """Load state from Redis or file, or create new."""
        cache = get_cache()

        # Try Redis first
        if cache and cache.is_connected:
            data = cache.get(self._state_key())
            if data:
                try:
                    state_dict = json.loads(data.decode())
                    logger.info(f"Loaded pipeline state for session {self.session_id} from Redis")
                    return PipelineState.from_dict(state_dict)
                except Exception as e:
                    logger.warning(f"Failed to load state from Redis: {e}")

        # Try file fallback
        state_file = self._state_file_path()
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state_dict = json.load(f)
                logger.info(f"Loaded pipeline state for session {self.session_id} from file")
                return PipelineState.from_dict(state_dict)
            except Exception as e:
                logger.warning(f"Failed to load state from file: {e}")

        # Create new state
        return PipelineState(
            session_id=self.session_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

    def save_state(self):
        """Save state to Redis and file."""
        state_data = self.state.to_dict()

        # Save to Redis
        cache = get_cache()
        if cache and cache.is_connected:
            try:
                data = json.dumps(state_data)
                cache.set(self._state_key(), data.encode(), TTL_INTERMEDIATE)
                logger.debug(f"Saved pipeline state for session {self.session_id} to Redis")
            except Exception as e:
                logger.warning(f"Failed to save state to Redis: {e}")

        # Always save to file as backup
        try:
            state_file = self._state_file_path()
            with open(state_file, "w") as f:
                json.dump(state_data, f)
            logger.debug(f"Saved pipeline state for session {self.session_id} to file")
        except Exception as e:
            logger.warning(f"Failed to save state to file: {e}")

    def set_config(self, config: dict):
        """Set pipeline configuration."""
        self.state.config = config
        self.save_state()

    def store_step_data(self, step: str, data: Any, summary: dict | None = None):
        """
        Store intermediate result data for a step.

        Args:
            step: Step name
            data: Data to store (DataFrame, ndarray, list, etc.)
            summary: Optional summary dict for display
        """
        cache = get_cache()

        # Try Redis first for smaller data
        if cache and cache.is_connected:
            try:
                serialized = pickle.dumps(data)
                # Only store in Redis if under 50MB
                if len(serialized) < 50 * 1024 * 1024:
                    cache.set(self._data_key(step), serialized, TTL_INTERMEDIATE)
                    logger.info(f"Stored {step} data in Redis ({len(serialized) / 1024:.1f} KB)")
                else:
                    # Fall back to file storage for large data
                    self._store_to_file(step, data)
            except Exception as e:
                logger.warning(f"Redis storage failed for {step}: {e}")
                self._store_to_file(step, data)
        else:
            # No Redis, use file storage
            self._store_to_file(step, data)

        # Mark step complete
        self.state.mark_step_complete(step, summary)
        self.save_state()

    def _store_to_file(self, step: str, data: Any):
        """Store data to file as fallback."""
        file_path = self._output_dir / f".pipeline_{self.session_id}_{step}.pkl"
        try:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Stored {step} data to file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to store {step} to file: {e}")

    def load_step_data(self, step: str) -> Any | None:
        """Load intermediate result data for a step."""
        cache = get_cache()

        # Try Redis first
        if cache and cache.is_connected:
            try:
                data = cache.get(self._data_key(step))
                if data:
                    logger.info(f"Loaded {step} data from Redis")
                    return pickle.loads(data)
            except Exception as e:
                logger.warning(f"Redis load failed for {step}: {e}")

        # Try file fallback
        file_path = self._output_dir / f".pipeline_{self.session_id}_{step}.pkl"
        if file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"Loaded {step} data from file")
                return data
            except Exception as e:
                logger.error(f"Failed to load {step} from file: {e}")

        return None

    def has_step_data(self, step: str) -> bool:
        """Check if step data is available."""
        cache = get_cache()

        # Check Redis
        if cache and cache.is_connected:
            try:
                if cache._client.exists(self._data_key(step)):
                    return True
            except Exception:
                pass

        # Check file
        file_path = self._output_dir / f".pipeline_{self.session_id}_{step}.pkl"
        return file_path.exists()

    def clear_step(self, step: str):
        """Clear data for a specific step and all subsequent steps."""
        self.state.clear_from_step(step)

        # Clear from Redis
        cache = get_cache()
        step_idx = PIPELINE_STEPS.index(step) if step in PIPELINE_STEPS else -1
        if step_idx >= 0:
            for s in PIPELINE_STEPS[step_idx:]:
                if cache and cache.is_connected:
                    try:
                        cache.delete(self._data_key(s))
                    except Exception:
                        pass
                # Clear file
                file_path = self._output_dir / f".pipeline_{self.session_id}_{s}.pkl"
                if file_path.exists():
                    file_path.unlink()

        self.save_state()
        logger.info(f"Cleared step {step} and subsequent steps")

    def clear_all(self):
        """Clear all state and data."""
        cache = get_cache()

        # Clear Redis
        if cache and cache.is_connected:
            try:
                cache.delete(self._state_key())
                for step in PIPELINE_STEPS:
                    cache.delete(self._data_key(step))
            except Exception:
                pass

        # Clear files
        for step in PIPELINE_STEPS:
            file_path = self._output_dir / f".pipeline_{self.session_id}_{step}.pkl"
            if file_path.exists():
                file_path.unlink()

        self._state = None
        logger.info(f"Cleared all state for session {self.session_id}")

    def refresh_state(self):
        """Force reload state from storage."""
        self._state = self.load_state()

    def get_status(self) -> dict:
        """Get current pipeline status for display."""
        # Always reload state to get latest updates
        self.refresh_state()
        state = self.state
        return {
            "session_id": self.session_id,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "config": state.config,
            "completed_steps": state.completed_steps,
            "current_step": state.current_step,
            "next_step": state.get_next_step(),
            "step_results": state.step_results,
            "error": state.error,
            "error_step": state.error_step,
            "steps": [
                {
                    "name": step,
                    "description": STEP_DESCRIPTIONS.get(step, step),
                    "completed": state.is_step_complete(step),
                    "has_data": self.has_step_data(step),
                    "result": state.step_results.get(step),
                    "error": state.error if state.error_step == step else None,
                    "completed_at": state.step_timing.get(step, {}).get("completed_at"),
                    "duration": state.step_timing.get(step, {}).get("duration"),
                }
                for step in PIPELINE_STEPS
            ],
        }


# Global state manager (per session)
_state_managers: dict[str, PipelineStateManager] = {}


def get_state_manager(session_id: str | None = None) -> PipelineStateManager:
    """Get or create state manager for session."""
    global _state_managers

    if session_id is None:
        # Create new session
        manager = PipelineStateManager()
        _state_managers[manager.session_id] = manager
        return manager

    if session_id not in _state_managers:
        _state_managers[session_id] = PipelineStateManager(session_id)

    return _state_managers[session_id]


def list_sessions() -> list[dict]:
    """List all active sessions."""
    cache = get_cache()
    sessions = []

    if cache and cache.is_connected:
        try:
            keys = list(cache._client.scan_iter(
                match="price_correlation:pipeline_state:*",
                count=100
            ))
            for key in keys:
                try:
                    data = cache.get(key.decode() if isinstance(key, bytes) else key)
                    if data:
                        state_dict = json.loads(data.decode())
                        sessions.append({
                            "session_id": state_dict.get("session_id"),
                            "created_at": state_dict.get("created_at"),
                            "updated_at": state_dict.get("updated_at"),
                            "completed_steps": len(state_dict.get("completed_steps", [])),
                            "total_steps": len(PIPELINE_STEPS),
                        })
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Failed to list sessions: {e}")

    return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
