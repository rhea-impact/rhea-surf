"""Ollama client for local LLM inference."""

import json
from dataclasses import dataclass
from typing import Optional
import ollama


@dataclass
class Action:
    """An action the agent wants to take."""
    action_type: str  # navigate, click, fill, scroll, done, etc.
    selector: Optional[str] = None
    value: Optional[str] = None
    reasoning: Optional[str] = None


SYSTEM_PROMPT = """You are a browser automation agent. You control a web browser to complete tasks.

You will receive:
1. The current URL and page title
2. A list of interactive elements on the page, formatted as:
   [index] element_type "text" attributes

You must respond with a JSON action:
{
  "action": "navigate|click|fill|scroll|press|done",
  "selector": "[index] or CSS selector",
  "value": "text to fill or URL to navigate",
  "reasoning": "brief explanation"
}

Actions:
- navigate: Go to a URL. value = the URL
- click: Click element. selector = element index like "[1]" or CSS selector
- fill: Type into input. selector = element index, value = text to type
- scroll: Scroll page. value = "up" or "down"
- press: Press a key. value = key name like "Enter"
- done: Task is complete. value = result summary

Rules:
- Use element indices [1], [2] etc when available
- One action at a time
- Be concise in reasoning
- If stuck, try scrolling or describe what you see
"""


class OllamaClient:
    """Client for Ollama local LLM."""

    def __init__(self, model: str = "deepseek-r1:8b"):
        self.model = model
        self.client = ollama.Client()

    def decide_action(
        self,
        task: str,
        url: str,
        title: str,
        elements: str,
        history: list[str] = None,
    ) -> Action:
        """
        Ask the LLM to decide the next action.

        Args:
            task: The user's task description
            url: Current page URL
            title: Current page title
            elements: Formatted interactive elements from format_for_llm()
            history: Recent action history

        Returns:
            Action to take
        """
        # Build context
        context = f"""Task: {task}

Current page:
URL: {url}
Title: {title}

Interactive elements:
{elements}
"""

        if history:
            context += f"\nRecent actions:\n" + "\n".join(history[-5:])

        context += "\n\nWhat action should I take next? Respond with JSON only."

        # Call Ollama
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            options={
                "num_ctx": 16384,  # Larger context for DOM
                "temperature": 0.1,  # Low temp for deterministic actions
            },
        )

        # Parse response
        content = response["message"]["content"]
        return self._parse_action(content)

    def _parse_action(self, content: str) -> Action:
        """Parse LLM response into an Action."""
        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            # Map element index to selector if needed
            selector = data.get("selector", "")
            if selector and selector.startswith("[") and selector.endswith("]"):
                # Keep as-is, we'll resolve in agent
                pass

            return Action(
                action_type=data.get("action", "done"),
                selector=selector,
                value=data.get("value"),
                reasoning=data.get("reasoning"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Raw content: {content[:500]}")
            return Action(action_type="done", reasoning=f"Parse error: {e}")


def test_ollama():
    """Test Ollama client."""
    client = OllamaClient(model="deepseek-r1:8b")

    elements = """[1] link "Home" href="/"
[2] link "About" href="/about"
[3] input[email] placeholder="Enter email"
[4] input[password] placeholder="Password"
[5] button "Sign In"
[6] link "Forgot password?" href="/forgot"
"""

    action = client.decide_action(
        task="Log in with email test@example.com and password secret123",
        url="https://example.com/login",
        title="Login - Example",
        elements=elements,
    )

    print(f"Action: {action.action_type}")
    print(f"Selector: {action.selector}")
    print(f"Value: {action.value}")
    print(f"Reasoning: {action.reasoning}")


if __name__ == "__main__":
    test_ollama()
