"""rhea-surf: Browser automation for local LLMs."""

from .browser import Browser
from .dom import extract_interactive_elements
from .llm import OllamaClient
from .agent import SurfAgent

__all__ = ["Browser", "extract_interactive_elements", "OllamaClient", "SurfAgent"]
