"""Playwright browser wrapper for surf agents."""

import asyncio
import base64
from dataclasses import dataclass
from typing import Optional
from playwright.async_api import async_playwright, Browser as PWBrowser, Page


@dataclass
class PageState:
    """Current state of a browser page."""
    url: str
    title: str
    html: str
    screenshot_base64: Optional[str] = None


class Browser:
    """Async Playwright browser wrapper."""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self._playwright = None
        self._browser: Optional[PWBrowser] = None
        self._page: Optional[Page] = None

    async def start(self) -> None:
        """Start the browser."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._page = await self._browser.new_page()

    async def stop(self) -> None:
        """Stop the browser."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> PageState:
        """Navigate to a URL and return page state."""
        await self._page.goto(url, wait_until=wait_until)
        return await self.get_state()

    async def get_state(self, include_screenshot: bool = False) -> PageState:
        """Get current page state."""
        html = await self._page.content()
        screenshot_b64 = None

        if include_screenshot:
            screenshot_bytes = await self._page.screenshot(type="png")
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        return PageState(
            url=self._page.url,
            title=await self._page.title(),
            html=html,
            screenshot_base64=screenshot_b64,
        )

    async def click(self, selector: str) -> bool:
        """Click an element by selector."""
        try:
            await self._page.click(selector, timeout=5000)
            await self._page.wait_for_load_state("domcontentloaded")
            return True
        except Exception as e:
            print(f"Click failed: {e}")
            return False

    async def fill(self, selector: str, text: str) -> bool:
        """Fill a form field."""
        try:
            await self._page.fill(selector, text)
            return True
        except Exception as e:
            print(f"Fill failed: {e}")
            return False

    async def press(self, key: str) -> bool:
        """Press a key (e.g., 'Enter')."""
        try:
            await self._page.keyboard.press(key)
            return True
        except Exception as e:
            print(f"Press failed: {e}")
            return False

    async def scroll(self, direction: str = "down", amount: int = 500) -> bool:
        """Scroll the page."""
        try:
            delta = amount if direction == "down" else -amount
            await self._page.mouse.wheel(0, delta)
            return True
        except Exception as e:
            print(f"Scroll failed: {e}")
            return False

    async def wait(self, selector: str, timeout: int = 5000) -> bool:
        """Wait for an element to appear."""
        try:
            await self._page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception as e:
            print(f"Wait failed: {e}")
            return False


async def test_browser():
    """Quick test of browser functionality."""
    browser = Browser(headless=False)
    await browser.start()

    state = await browser.navigate("https://www.google.com")
    print(f"URL: {state.url}")
    print(f"Title: {state.title}")
    print(f"HTML length: {len(state.html)}")

    await asyncio.sleep(2)
    await browser.stop()


if __name__ == "__main__":
    asyncio.run(test_browser())
