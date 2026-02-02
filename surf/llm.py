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
   Elements marked [NAV] are navigation/menu items - skip these for content tasks

You must respond with ONLY a JSON object (no other text):
{
  "action": "navigate|click|fill|scroll|press|read|done",
  "selector": "[index] or CSS selector (for click/fill/read)",
  "value": "text to fill, URL to navigate, or result for done",
  "reasoning": "brief explanation"
}

Actions:
- read: Read text from an element WITHOUT clicking. Use this FIRST when you need information visible on the page.
- done: Task is COMPLETE. value = the answer or result. USE THIS after reading what you need.
- click: Click element. selector = element index like "[1]". Only use if you need to navigate to another page.
- navigate: Go to a URL. value = the full URL. Only use to go to a different website.
- fill: Type into input. selector = element index, value = text to type.
- scroll: Scroll page. value = "up" or "down".
- press: Press a key. value = key name like "Enter".

IMPORTANT - READ FIRST RULE:
If the task asks "what is...", "tell me...", "find...", or "read..." and the answer is visible in the element list:
1. DO NOT CLICK - the info is already here
2. Use "read" to extract the text from the element containing the answer
3. Then use "done" to report it

CRITICAL RULES:
1. Use element indices [1], [2] etc - not CSS selectors
2. One action at a time
3. LOOK AT THE ELEMENT TEXT - if an element's text contains your answer, use "read" on it
4. After reading, immediately use "done" to report the result
5. DO NOT click links just to "see" something - the text is already in the element list

EXAMPLE 1 - Information visible on page:
Task: "What is the top story?"
Elements: [1] link "Home"  [2] link "Breaking News: Scientists Discover New Planet"  [3] link "Sports"
CORRECT: {"action": "read", "selector": "[2]", "reasoning": "Element [2] contains the top story title"}
Then: {"action": "done", "value": "The top story is: Breaking News: Scientists Discover New Planet", "reasoning": "Read the story title"}

EXAMPLE 2 - Already read in history:
History shows: "read -> GOT: Python is a programming language"
CORRECT: {"action": "done", "value": "Python is a programming language", "reasoning": "Already read the answer"}
"""


class OllamaClient:
    """Client for Ollama local LLM."""

    def __init__(self, model: str = "llama3:latest"):
        self.model = model
        self.client = ollama.Client()

    def decide_action(
        self,
        task: str,
        url: str,
        title: str,
        elements: str,
        history: list[str] = None,
        learned_patterns: str = None,
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

        if learned_patterns:
            context += f"\n{learned_patterns}\n"

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
    client = OllamaClient(model="llama3:latest")

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
