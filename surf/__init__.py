"""rhea-surf: Browser automation for local LLMs."""

from .browser import Browser
from .dom import (
    extract_interactive_elements,
    extract_dom_playwright,
    extract_dom_static,
    DOMState,
    Element,
    format_for_llm,
    format_dom_state,
)
from .llm import OllamaClient
from .agent import SurfAgent
from .patterns import find_dumb_pattern, PatternMatcher
from .navigator import RecursiveNavigator, NavigationResult
from .learner import Learner
from .debate import Debate, run_debate
from .recursive import RecursiveReasoner, decompose_task
from .cache import ActionCache, CachedAction, hash_dom_structure
from .vision import VisionDecider, VisionAction, vision_decide, should_use_vision
from .memory import TrajectoryMemory, Trajectory, ActionStep
from .study import StudyRunner, StudySession, LossMetrics, StudyDB

__all__ = [
    # Browser
    "Browser",
    # DOM extraction
    "extract_interactive_elements",
    "extract_dom_playwright",
    "extract_dom_static",
    "DOMState",
    "Element",
    "format_for_llm",
    "format_dom_state",
    # LLM
    "OllamaClient",
    # Agent
    "SurfAgent",
    # Patterns
    "find_dumb_pattern",
    "PatternMatcher",
    # Navigator
    "RecursiveNavigator",
    "NavigationResult",
    # Learning
    "Learner",
    # Debate
    "Debate",
    "run_debate",
    # Recursive reasoning
    "RecursiveReasoner",
    "decompose_task",
    # Caching
    "ActionCache",
    "CachedAction",
    "hash_dom_structure",
    # Vision
    "VisionDecider",
    "VisionAction",
    "vision_decide",
    "should_use_vision",
    # Memory
    "TrajectoryMemory",
    "Trajectory",
    "ActionStep",
    # Study/Meta-learning
    "StudyRunner",
    "StudySession",
    "LossMetrics",
    "StudyDB",
]
