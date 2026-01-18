"""Stock clustering system for identifying correlated equities."""

__version__ = "0.1.0"

from .pipeline import PipelineConfig, run_pipeline, run_sample_pipeline

__all__ = ["PipelineConfig", "run_pipeline", "run_sample_pipeline"]
