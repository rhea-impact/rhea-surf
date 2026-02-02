"""Surf agent - the main agent loop."""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .browser import Browser, PageState
from .dom import extract_interactive_elements, format_for_llm, InteractiveElement
from .llm import OllamaClient, Action


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
        model: str = "llama3:latest",
        headless: bool = False,
        max_steps: int = 20,
        verbose: bool = True,
    ):
        self.browser = Browser(headless=headless)
        self.llm = OllamaClient(model=model)
        self.max_steps = max_steps
        self.verbose = verbose
        self._elements: list[InteractiveElement] = []

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

        try:
            await self.browser.start()

            # Navigate to start URL if provided
            if start_url:
                if self.verbose:
                    print(f"[Agent] Navigating to {start_url}")
                await self.browser.navigate(start_url)

            # Main agent loop
            for step_num in range(1, self.max_steps + 1):
                if self.verbose:
                    print(f"\n[Agent] Step {step_num}/{self.max_steps}")

                # Get current state
                state = await self.browser.get_state()
                if self.verbose:
                    print(f"[Agent] URL: {state.url}")

                # Extract elements
                self._elements = extract_interactive_elements(state.html)
                elements_text = format_for_llm(self._elements)

                if self.verbose:
                    print(f"[Agent] Found {len(self._elements)} interactive elements")

                # Ask LLM for action
                action = self.llm.decide_action(
                    task=task,
                    url=state.url,
                    title=state.title,
                    elements=elements_text,
                    history=history,
                )

                if self.verbose:
                    print(f"[Agent] Action: {action.action_type}")
                    if action.reasoning:
                        print(f"[Agent] Reasoning: {action.reasoning}")

                # Record step
                step = AgentStep(
                    step_num=step_num,
                    timestamp=datetime.now(),
                    url=state.url,
                    action=action,
                    success=True,
                )

                # Execute action
                try:
                    success = await self._execute_action(action)
                    step.success = success
                    history.append(f"Step {step_num}: {action.action_type} -> {'OK' if success else 'FAILED'}")
                except Exception as e:
                    step.success = False
                    step.error = str(e)
                    history.append(f"Step {step_num}: {action.action_type} -> ERROR: {e}")

                result.steps.append(step)

                # Check if done
                if action.action_type == "done":
                    result.success = True
                    result.summary = action.value or action.reasoning or "Task completed"
                    break

                # Small delay between actions
                await asyncio.sleep(0.5)

            # Get final state
            final_state = await self.browser.get_state()
            result.final_url = final_state.url

        except Exception as e:
            result.summary = f"Agent error: {e}"
            if self.verbose:
                print(f"[Agent] Error: {e}")

        finally:
            await self.browser.stop()

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

        elif action_type == "done":
            return True

        else:
            print(f"[Agent] Unknown action type: {action_type}")
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
        model="llama3:latest",
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
