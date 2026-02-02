"""Surf agent - the main agent loop."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .browser import Browser, PageState
from .dom import extract_interactive_elements, format_for_llm, InteractiveElement
from .llm import OllamaClient, Action

# Set up logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "agent.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """Record of a single agent step."""
    step_num: int
    timestamp: datetime
    url: str
    action: Action
    success: bool
    error: Optional[str] = None


@dataclass
class AgentResult:
    """Result of an agent run."""
    task: str
    success: bool
    steps: list[AgentStep] = field(default_factory=list)
    final_url: str = ""
    summary: str = ""
    total_tokens: int = 0


class SurfAgent:
    """
    Browser automation agent using local LLM.

    The agent loop:
    1. Get page state (DOM)
    2. Extract interactive elements
    3. Ask LLM for next action
    4. Execute action
    5. Repeat until done or max steps
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        headless: bool = False,
        max_steps: int = 20,
        verbose: bool = True,
    ):
        self.browser = Browser(headless=headless)
        self.llm = OllamaClient(model=model)
        self.max_steps = max_steps
        self.verbose = verbose
        self._elements: list[InteractiveElement] = []
        self._last_read_result: str = ""

    async def run(self, task: str, start_url: Optional[str] = None) -> AgentResult:
        """
        Run the agent on a task.

        Args:
            task: Natural language task description
            start_url: Optional starting URL

        Returns:
            AgentResult with steps taken and outcome
        """
        result = AgentResult(task=task, success=False)
        history = []

        # Create run log file
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_log = LOG_DIR / f"run_{run_id}.json"
        run_data = {
            "task": task,
            "start_url": start_url,
            "started_at": datetime.now().isoformat(),
            "steps": [],
        }

        logger.info(f"=== Starting run {run_id} ===")
        logger.info(f"Task: {task}")

        try:
            await self.browser.start()

            # Navigate to start URL if provided
            if start_url:
                logger.info(f"Navigating to {start_url}")
                await self.browser.navigate(start_url)

            # Main agent loop
            for step_num in range(1, self.max_steps + 1):
                logger.info(f"--- Step {step_num}/{self.max_steps} ---")

                # Get current state
                state = await self.browser.get_state()
                logger.info(f"URL: {state.url}")

                # Extract elements
                self._elements = extract_interactive_elements(state.html)
                elements_text = format_for_llm(self._elements)

                logger.info(f"Found {len(self._elements)} interactive elements")

                # Ask LLM for action
                action = self.llm.decide_action(
                    task=task,
                    url=state.url,
                    title=state.title,
                    elements=elements_text,
                    history=history,
                )

                logger.info(f"Action: {action.action_type}")
                logger.info(f"Selector: {action.selector}")
                logger.info(f"Value: {action.value}")
                logger.info(f"Reasoning: {action.reasoning}")

                # Record step
                step = AgentStep(
                    step_num=step_num,
                    timestamp=datetime.now(),
                    url=state.url,
                    action=action,
                    success=True,
                )

                # Log step data
                step_data = {
                    "step_num": step_num,
                    "timestamp": datetime.now().isoformat(),
                    "url": state.url,
                    "title": state.title,
                    "elements_count": len(self._elements),
                    "elements_sample": elements_text[:500],
                    "action": {
                        "type": action.action_type,
                        "selector": action.selector,
                        "value": action.value,
                        "reasoning": action.reasoning,
                    },
                }

                # Execute action
                try:
                    success = await self._execute_action(action)
                    step.success = success
                    step_data["success"] = success

                    # Build history entry with read result if applicable
                    if action.action_type == "read" and self._last_read_result:
                        history_entry = f"Step {step_num}: read -> GOT: \"{self._last_read_result[:100]}\""
                        step_data["read_result"] = self._last_read_result
                    else:
                        history_entry = f"Step {step_num}: {action.action_type} -> {'OK' if success else 'FAILED'}"
                    history.append(history_entry)
                    logger.info(f"Result: {'OK' if success else 'FAILED'}")
                except Exception as e:
                    step.success = False
                    step.error = str(e)
                    step_data["success"] = False
                    step_data["error"] = str(e)
                    history.append(f"Step {step_num}: {action.action_type} -> ERROR: {e}")
                    logger.error(f"Error: {e}")

                result.steps.append(step)
                run_data["steps"].append(step_data)

                # Check if done
                if action.action_type == "done":
                    result.success = True
                    result.summary = action.value or action.reasoning or "Task completed"
                    logger.info(f"Task complete: {result.summary}")
                    break

                # Small delay between actions
                await asyncio.sleep(0.5)

            # Get final state
            final_state = await self.browser.get_state()
            result.final_url = final_state.url

        except Exception as e:
            result.summary = f"Agent error: {e}"
            logger.error(f"Agent error: {e}")

        finally:
            await self.browser.stop()

        # Save run log
        run_data["completed_at"] = datetime.now().isoformat()
        run_data["success"] = result.success
        run_data["final_url"] = result.final_url
        run_data["summary"] = result.summary

        with open(run_log, "w") as f:
            json.dump(run_data, f, indent=2, default=str)

        logger.info(f"Run log saved to {run_log}")
        logger.info(f"=== Run {run_id} complete: {'SUCCESS' if result.success else 'FAILED'} ===")

        return result

    async def _execute_action(self, action: Action) -> bool:
        """Execute an action from the LLM."""
        action_type = action.action_type.lower()

        if action_type == "navigate":
            if action.value:
                await self.browser.navigate(action.value)
                return True
            return False

        elif action_type == "click":
            selector = self._resolve_selector(action.selector)
            if selector:
                return await self.browser.click(selector)
            return False

        elif action_type == "fill":
            selector = self._resolve_selector(action.selector)
            if selector and action.value:
                return await self.browser.fill(selector, action.value)
            return False

        elif action_type == "scroll":
            direction = action.value or "down"
            return await self.browser.scroll(direction)

        elif action_type == "press":
            if action.value:
                return await self.browser.press(action.value)
            return False

        elif action_type == "read":
            selector = self._resolve_selector(action.selector)
            if selector:
                text = await self.browser.read_text(selector)
                logger.info(f"Read text: {text[:200] if text else '(empty)'}")
                # Store the read result so we can include it in history
                self._last_read_result = text
                return bool(text)
            return False

        elif action_type == "done":
            return True

        else:
            logger.warning(f"Unknown action type: {action_type}")
            return False

    def _resolve_selector(self, selector: Optional[str]) -> Optional[str]:
        """Resolve element index to CSS selector."""
        if not selector:
            return None

        # Check for element index like "[1]" or "1"
        match = re.match(r"\[?(\d+)\]?", selector)
        if match:
            index = int(match.group(1))
            for el in self._elements:
                if el.index == index:
                    return el.selector
            print(f"[Agent] Element index {index} not found")
            return None

        # Already a CSS selector
        return selector


async def test_agent():
    """Test the agent on a simple task."""
    agent = SurfAgent(
        model="llama3.1:8b",
        headless=False,
        max_steps=10,
        verbose=True,
    )

    result = await agent.run(
        task="Go to Google and search for 'weather in Fort Worth'",
        start_url="https://www.google.com",
    )

    print(f"\n{'='*50}")
    print(f"Task: {result.task}")
    print(f"Success: {result.success}")
    print(f"Steps taken: {len(result.steps)}")
    print(f"Final URL: {result.final_url}")
    print(f"Summary: {result.summary}")


if __name__ == "__main__":
    asyncio.run(test_agent())
