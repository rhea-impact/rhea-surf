"""
Vision fallback for browser automation.

Uses llama3.2-vision when DOM-based extraction fails or returns
too few interactive elements (likely JS-heavy SPAs).

Triggers:
- DOM extraction returns <5 interactive elements
- Action fails and retries exhausted
- Page contains canvas/WebGL elements
- Confidence from DOM-based decision <0.5
"""

import base64
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ollama

logger = logging.getLogger(__name__)


@dataclass
class VisionAction:
    """Action decided via visual analysis."""
    action_type: str
    target_description: str  # e.g., "blue button in top right"
    value: Optional[str] = None
    confidence: float = 0.5
    reasoning: str = ""

    # Approximate coordinates if detected
    x: Optional[int] = None
    y: Optional[int] = None


# Vision prompt template - kept simple for better model compliance
VISION_PROMPT = """Task: {task}

Look at the screenshot and answer. If this is a "what is" question, just tell me the answer.

Format your response as:
ACTION: done
ANSWER: [your answer here]
CONFIDENCE: high/medium/low
"""


class VisionDecider:
    """
    Visual decision making using llama3.2-vision.

    Falls back to vision when DOM extraction fails or is incomplete.
    """

    def __init__(self, model: str = "llama3.2-vision"):
        self.model = model
        self.client = ollama.Client()

    def decide(
        self,
        screenshot_path: str,
        task: str,
        url: str = "",
        history: list = None
    ) -> VisionAction:
        """
        Analyze screenshot and decide action.

        Args:
            screenshot_path: Path to screenshot image
            task: What we're trying to accomplish
            url: Current URL (for context)
            history: Previous actions taken

        Returns:
            VisionAction with decision
        """
        # Read and encode image
        image_data = self._encode_image(screenshot_path)
        if not image_data:
            return VisionAction(
                action_type="scroll",
                target_description="page",
                value="down",
                confidence=0.0,
                reasoning="Failed to load screenshot"
            )

        # Build prompt with history context
        prompt = VISION_PROMPT.format(task=task, url=url)
        if history:
            prompt += f"\n\nPrevious actions: {history[-3:]}"

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [image_data]
                }],
                options={"temperature": 0.3, "num_ctx": 4096}
            )

            content = response["message"]["content"]
            data = self._parse_response(content)

            # Convert approximate location to coordinates
            x, y = self._location_to_coords(data.get("approximate_location", "center"))

            return VisionAction(
                action_type=data.get("action", "scroll"),
                target_description=data.get("target", ""),
                value=data.get("value"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                x=x,
                y=y
            )

        except Exception as e:
            logger.error(f"Vision decision failed: {e}")
            return VisionAction(
                action_type="scroll",
                target_description="page",
                value="down",
                confidence=0.0,
                reasoning=f"Vision error: {e}"
            )

    def decide_from_bytes(
        self,
        screenshot_bytes: bytes,
        task: str,
        url: str = "",
        history: list = None
    ) -> VisionAction:
        """
        Analyze screenshot from bytes and decide action.

        Convenience method for Playwright screenshots.
        """
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(screenshot_bytes)
            temp_path = f.name

        try:
            return self.decide(temp_path, task, url, history)
        finally:
            # Cleanup temp file
            Path(temp_path).unlink(missing_ok=True)

    def _encode_image(self, path: str) -> Optional[str]:
        """Encode image to base64."""
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def _parse_response(self, content: str) -> dict:
        """Parse vision model response (handles both JSON and natural language)."""
        import re

        result = {"action": "scroll", "value": "down", "confidence": 0.5, "target": "", "reasoning": ""}

        content_lower = content.lower()

        # Try JSON first
        try:
            match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group(0).replace("'", '"'))
                return data
        except:
            pass

        # Parse structured format (ACTION: / ANSWER: / CONFIDENCE:)
        action_match = re.search(r'action:\s*(\w+)', content_lower)
        if action_match:
            result["action"] = action_match.group(1)

        answer_match = re.search(r'answer:\s*(.+?)(?:\n|confidence:|$)', content, re.IGNORECASE)
        if answer_match:
            result["value"] = answer_match.group(1).strip()
            result["action"] = "done"

        conf_match = re.search(r'confidence:\s*(\w+)', content_lower)
        if conf_match:
            conf_word = conf_match.group(1)
            result["confidence"] = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(conf_word, 0.5)

        # Fallback: if response looks like an answer, extract it
        if result["action"] == "scroll" and len(content) < 200:
            # Check if it's a direct answer
            if "heading" in content_lower or "title" in content_lower or "domain" in content_lower:
                # Extract quoted text or the main content
                quoted = re.search(r'"([^"]+)"', content)
                if quoted:
                    result["value"] = quoted.group(1)
                    result["action"] = "done"
                    result["confidence"] = 0.7
                else:
                    # Use the whole response as the answer
                    result["value"] = content.strip()
                    result["action"] = "done"
                    result["confidence"] = 0.6

        result["reasoning"] = content[:200]
        return result

    def _location_to_coords(self, location: str, width: int = 1280, height: int = 720) -> tuple:
        """
        Convert approximate location to viewport coordinates.

        Returns (x, y) tuple for approximate center of region.
        """
        location = location.lower().replace("-", "_").replace(" ", "_")

        locations = {
            "top_left": (width * 0.15, height * 0.15),
            "top_center": (width * 0.5, height * 0.15),
            "top_right": (width * 0.85, height * 0.15),
            "middle_left": (width * 0.15, height * 0.5),
            "center": (width * 0.5, height * 0.5),
            "middle_right": (width * 0.85, height * 0.5),
            "bottom_left": (width * 0.15, height * 0.85),
            "bottom_center": (width * 0.5, height * 0.85),
            "bottom_right": (width * 0.85, height * 0.85),
        }

        coords = locations.get(location, (width * 0.5, height * 0.5))
        return int(coords[0]), int(coords[1])


async def vision_decide(
    page,
    task: str,
    url: str = "",
    history: list = None,
    model: str = "llama3.2-vision"
) -> VisionAction:
    """
    Convenience function: take screenshot and decide action.

    Args:
        page: Playwright Page object
        task: What we're trying to accomplish
        url: Current URL
        history: Previous actions
        model: Vision model to use

    Returns:
        VisionAction
    """
    # Take screenshot
    screenshot_bytes = await page.screenshot(type="png")

    # Decide
    decider = VisionDecider(model=model)
    return decider.decide_from_bytes(screenshot_bytes, task, url or page.url, history)


def should_use_vision(
    dom_element_count: int,
    dom_confidence: float = 1.0,
    action_failed: bool = False,
    retry_count: int = 0
) -> bool:
    """
    Determine if we should fallback to vision.

    Returns True if:
    - DOM has very few elements (<5 interactive)
    - DOM-based confidence is low (<0.5)
    - Action has failed multiple times
    """
    # Too few elements - likely JS-heavy SPA
    if dom_element_count < 5:
        return True

    # Low confidence from DOM analysis
    if dom_confidence < 0.5:
        return True

    # Action keeps failing
    if action_failed and retry_count >= 2:
        return True

    return False


# Test function
async def test_vision():
    """Test vision fallback with a screenshot."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto("https://news.ycombinator.com")
        await page.wait_for_load_state("networkidle")

        # Test vision decision
        action = await vision_decide(
            page,
            task="Find and click on the top story",
            url="https://news.ycombinator.com"
        )

        print(f"Vision decision:")
        print(f"  Action: {action.action_type}")
        print(f"  Target: {action.target_description}")
        print(f"  Value: {action.value}")
        print(f"  Confidence: {action.confidence}")
        print(f"  Reasoning: {action.reasoning}")
        print(f"  Coords: ({action.x}, {action.y})")

        await browser.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_vision())
