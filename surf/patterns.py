"""
Dumb patterns - deterministic actions that don't need LLM.

These are common web interaction patterns that can be executed
without consulting an LLM, saving tokens and time.
"""

import re
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable
from .dom import InteractiveElement


@dataclass
class Pattern:
    """A deterministic action pattern."""
    name: str
    description: str
    match: Callable[[str, list[InteractiveElement]], bool]
    execute: Callable  # async function


class PatternMatcher:
    """Match and execute common web patterns without LLM."""

    def __init__(self):
        self.patterns: list[Pattern] = []
        self._register_builtin_patterns()

    def _register_builtin_patterns(self):
        """Register built-in dumb patterns."""

        # Pattern: Cookie consent -> dismiss (highest priority)
        self.patterns.append(Pattern(
            name="cookie_dismiss",
            description="Dismiss cookie consent banner",
            match=self._match_cookie_banner,
            execute=self._execute_cookie_dismiss,
        ))

        # Pattern: Search box visible + task mentions search -> fill and submit
        self.patterns.append(Pattern(
            name="search_and_submit",
            description="Fill search box and press Enter",
            match=self._match_search,
            execute=self._execute_search,
        ))

        # Pattern: Login form + task mentions login -> fill and submit
        self.patterns.append(Pattern(
            name="login_form",
            description="Fill login form and submit",
            match=self._match_login,
            execute=self._execute_login,
        ))

        # NOTE: Removed click_first_result - too aggressive
        # Let LLM decide when to click vs read

    # --- Search Pattern ---
    def _match_search(self, task: str, elements: list[InteractiveElement]) -> bool:
        """Match if task mentions search and we have a search input."""
        if not re.search(r'\b(search|find|look for|query)\b', task.lower()):
            return False
        return any(
            el.element_type in ('text', 'search') and
            any(kw in str(el.attributes).lower() for kw in ['search', 'query', 'q'])
            for el in elements
        )

    async def _execute_search(self, browser, task: str, elements: list[InteractiveElement]) -> dict:
        """Fill search box and press Enter."""
        # Extract search query from task
        match = re.search(r'(?:search|find|look for|query)[:\s]+["\']?([^"\']+)["\']?', task.lower())
        if not match:
            match = re.search(r'["\']([^"\']+)["\']', task)
        query = match.group(1).strip() if match else ""

        if not query:
            return {"success": False, "reason": "Could not extract search query"}

        # Find search input
        search_input = None
        for el in elements:
            if el.element_type in ('text', 'search'):
                attrs = str(el.attributes).lower()
                if any(kw in attrs for kw in ['search', 'query', 'q']):
                    search_input = el
                    break

        if not search_input:
            return {"success": False, "reason": "No search input found"}

        # Execute: fill + Enter
        await browser.fill(search_input.selector, query)
        await browser.press("Enter")

        return {
            "success": True,
            "actions": [
                f"fill [{search_input.index}] with '{query}'",
                "press Enter"
            ]
        }

    # --- Login Pattern ---
    def _match_login(self, task: str, elements: list[InteractiveElement]) -> bool:
        """Match if task mentions login and we have email/password fields."""
        if not re.search(r'\b(login|sign in|log in)\b', task.lower()):
            return False
        has_email = any(el.element_type in ('email', 'text') for el in elements)
        has_password = any(el.element_type == 'password' for el in elements)
        return has_email and has_password

    async def _execute_login(self, browser, task: str, elements: list[InteractiveElement]) -> dict:
        """Fill login form and submit."""
        # Extract credentials from task (simplified)
        email_match = re.search(r'email[:\s]+(\S+)', task.lower())
        pass_match = re.search(r'password[:\s]+(\S+)', task.lower())

        if not email_match or not pass_match:
            return {"success": False, "reason": "Could not extract credentials"}

        email = email_match.group(1)
        password = pass_match.group(1)

        # Find fields
        email_field = next((el for el in elements if el.element_type in ('email', 'text')), None)
        pass_field = next((el for el in elements if el.element_type == 'password'), None)
        submit_btn = next((el for el in elements if el.element_type == 'button' and
                          any(kw in el.text.lower() for kw in ['login', 'sign in', 'submit'])), None)

        if not email_field or not pass_field:
            return {"success": False, "reason": "Login fields not found"}

        # Execute
        await browser.fill(email_field.selector, email)
        await browser.fill(pass_field.selector, password)
        if submit_btn:
            await browser.click(submit_btn.selector)
        else:
            await browser.press("Enter")

        return {"success": True, "actions": ["filled email", "filled password", "submitted"]}

    # --- Cookie Banner Pattern ---
    def _match_cookie_banner(self, task: str, elements: list[InteractiveElement]) -> bool:
        """Match cookie consent banners."""
        return any(
            el.element_type == 'button' and
            any(kw in el.text.lower() for kw in ['accept', 'agree', 'got it', 'ok', 'dismiss'])
            for el in elements
        )

    async def _execute_cookie_dismiss(self, browser, task: str, elements: list[InteractiveElement]) -> dict:
        """Click accept/dismiss on cookie banner."""
        for el in elements:
            if el.element_type == 'button':
                text = el.text.lower()
                if any(kw in text for kw in ['accept all', 'accept', 'agree', 'got it', 'ok']):
                    await browser.click(el.selector)
                    return {"success": True, "actions": [f"clicked '{el.text}'"]}
        return {"success": False, "reason": "No dismiss button found"}

    # --- Search Results Pattern ---
    def _match_search_results(self, task: str, elements: list[InteractiveElement]) -> bool:
        """Match search results page."""
        # Look for multiple links that look like results
        result_links = [el for el in elements if el.element_type == 'link' and len(el.text) > 20]
        return len(result_links) >= 3

    async def _execute_click_first_result(self, browser, task: str, elements: list[InteractiveElement]) -> dict:
        """Click the first meaningful search result."""
        for el in elements:
            if el.element_type == 'link' and len(el.text) > 20:
                # Skip navigation links
                if any(skip in el.text.lower() for skip in ['next', 'previous', 'page', 'skip']):
                    continue
                await browser.click(el.selector)
                return {"success": True, "actions": [f"clicked first result: '{el.text[:50]}...'"]}
        return {"success": False, "reason": "No results to click"}

    def find_pattern(self, task: str, elements: list[InteractiveElement]) -> Optional[Pattern]:
        """Find a matching pattern for the current state."""
        for pattern in self.patterns:
            if pattern.match(task, elements):
                return pattern
        return None


# Singleton instance
_matcher = PatternMatcher()


def find_dumb_pattern(task: str, elements: list[InteractiveElement]) -> Optional[Pattern]:
    """Find a dumb pattern that matches the current state."""
    return _matcher.find_pattern(task, elements)


async def execute_dumb_pattern(pattern: Pattern, browser, task: str, elements: list[InteractiveElement]) -> dict:
    """Execute a dumb pattern."""
    return await pattern.execute(browser, task, elements)
