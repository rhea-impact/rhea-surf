"""
Recursive navigator with backtracking, caching, and multi-language reasoning.

Uses a layered decision approach:
1. Action cache (100% deterministic if hit)
2. Dumb patterns (no LLM)
3. Learned patterns (semi-deterministic)
4. LLM decision (probabilistic)
5. Multi-agent debate (for uncertain decisions)
6. RSA aggregation (for complex pages)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from .browser import Browser, PageState
from .dom import (
    DOMState, Element,
    extract_dom_playwright, extract_dom_static,
    format_for_llm, format_dom_state
)
from .llm import OllamaClient, Action
from .patterns import find_dumb_pattern, execute_dumb_pattern
from .learner import Learner
from .debate import run_debate, DebateResult
from .cache import ActionCache, hash_dom_structure
from .recursive import RecursiveReasoner, decompose_task, AggregatedDecision
from .memory import TrajectoryMemory, Trajectory, ActionStep

logger = logging.getLogger(__name__)


@dataclass
class NavNode:
    """A node in the navigation tree."""
    id: int
    url: str
    depth: int
    action_taken: Optional[str] = None
    parent: Optional['NavNode'] = None
    children: List['NavNode'] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    visited_at: datetime = field(default_factory=datetime.now)


@dataclass
class NavigationResult:
    """Result of a navigation attempt."""
    success: bool
    result: Optional[str] = None
    depth: int = 0
    history: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    cache_hits: int = 0
    llm_calls: int = 0


class RecursiveNavigator:
    """
    Navigate web pages using recursion with backtracking.

    Decision hierarchy (fastest to slowest):
    1. Action cache - replay cached successful actions
    2. Dumb patterns - regex-based, no LLM
    3. Learned patterns - embedding similarity search
    4. LLM decision - single model query
    5. Debate - multi-agent voting for uncertain decisions
    6. RSA aggregation - multi-model + multi-language for complex pages
    """

    # Thresholds
    CACHE_CONFIDENCE_THRESHOLD = 0.85
    PATTERN_SIMILARITY_THRESHOLD = 0.85
    DEBATE_CONFIDENCE_THRESHOLD = 0.6
    COMPLEX_PAGE_ELEMENT_COUNT = 20  # Trigger RSA if more elements

    def __init__(
        self,
        model: str = "qwen3:14b",
        max_depth: int = 10,
        max_retries: int = 3,
        use_learning: bool = True,
        use_cache: bool = True,
        use_recursive: bool = True,
    ):
        self.browser = Browser(headless=True)
        self.llm = OllamaClient(model=model)
        self.max_depth = max_depth
        self.max_retries = max_retries
        self.use_learning = use_learning
        self.use_cache = use_cache
        self.use_recursive = use_recursive
        self.use_debate = True
        self.debate_threshold = self.DEBATE_CONFIDENCE_THRESHOLD

        # Systems
        self.learner = Learner() if use_learning else None
        self.cache = ActionCache() if use_cache else None
        self.memory = TrajectoryMemory()  # Always use trajectory memory
        self.reasoner = RecursiveReasoner(
            models=["qwen3:14b", "llama3.1:8b", "gemma3n:e4b"],
            languages=["en", "zh", "es"]
        ) if use_recursive else None

        # State
        self._node_counter = 0
        self._root: Optional[NavNode] = None
        self._current: Optional[NavNode] = None
        self._visited_states: set = set()
        self._dom_state: Optional[DOMState] = None
        self._last_read_result: str = ""
        self._executed_actions: set = set()  # Track (url, action_type) to avoid loops
        self._action_history: List[ActionStep] = []  # Track for trajectory storage
        self._few_shot_context: str = ""  # Retrieved similar trajectories

        # Metrics
        self._cache_hits = 0
        self._llm_calls = 0

    async def navigate(self, task: str, start_url: Optional[str] = None) -> NavigationResult:
        """
        Navigate to complete a task.

        Returns:
            NavigationResult with success, result, metrics
        """
        logger.info(f"=== Starting recursive navigation ===")
        logger.info(f"Task: {task}")

        self._cache_hits = 0
        self._llm_calls = 0
        self._executed_actions = set()  # Reset for new session
        self._action_history = []  # Reset action history
        self._few_shot_context = ""

        try:
            await self.browser.start()

            # Navigate to start
            if start_url:
                await self.browser.navigate(start_url)

            # Retrieve similar trajectories for few-shot learning
            similar = self.memory.search(task, start_url or "", limit=2)
            if similar:
                self._few_shot_context = self.memory.format_few_shot(similar)
                logger.info(f"Found {len(similar)} similar trajectories for few-shot")

            state = await self.browser.get_state()
            self._root = self._create_node(state.url, depth=0)
            self._current = self._root

            # Check if task is complex - decompose if needed
            if self._is_complex_task(task):
                result = await self._navigate_complex(task)
            else:
                result = await self._navigate_recursive(task, depth=0)

            # Store successful trajectory (non-blocking)
            if result.get("success"):
                try:
                    self._store_trajectory(task, start_url or state.url, result)
                except Exception as e:
                    logger.warning(f"Failed to store trajectory: {e}")

            return NavigationResult(
                success=result.get("success", False),
                result=result.get("result"),
                depth=result.get("depth", 0),
                history=result.get("history", []),
                reason=result.get("reason"),
                cache_hits=self._cache_hits,
                llm_calls=self._llm_calls,
            )

        finally:
            await self.browser.stop()

    async def _navigate_complex(self, task: str) -> dict:
        """Handle complex tasks by decomposing into subtasks."""
        logger.info("Complex task detected - decomposing...")

        subtasks = decompose_task(task)
        logger.info(f"Decomposed into {len(subtasks)} subtasks")

        all_history = []
        final_result = None

        for i, subtask in enumerate(subtasks):
            logger.info(f"Subtask {i+1}/{len(subtasks)}: {subtask}")

            result = await self._navigate_recursive(subtask, depth=0, history=all_history.copy())

            if not result.get("success"):
                return {
                    "success": False,
                    "reason": f"Subtask {i+1} failed: {result.get('reason')}",
                    "history": all_history,
                }

            all_history.extend(result.get("history", []))
            final_result = result.get("result")

        return {
            "success": True,
            "result": final_result,
            "depth": len(subtasks),
            "history": all_history,
        }

    async def _navigate_recursive(self, task: str, depth: int, history: List[str] = None) -> dict:
        """Recursive navigation with backtracking and caching."""

        if history is None:
            history = []

        if depth >= self.max_depth:
            logger.warning(f"Max depth {self.max_depth} reached")
            return {"success": False, "reason": "Max depth reached"}

        # Get current state
        state = await self.browser.get_state()
        logger.info(f"[Depth {depth}] URL: {state.url}")

        # Extract DOM (prefer Playwright, fallback to static)
        try:
            self._dom_state = await extract_dom_playwright(self.browser._page)
        except Exception as e:
            logger.warning(f"Playwright extraction failed, using static: {e}")
            self._dom_state = extract_dom_static(state.html, state.url, state.title)

        logger.info(f"Found {len(self._dom_state.elements)} elements (hash: {self._dom_state.dom_hash})")

        # === LAYER 1: Check action cache ===
        if self.cache:
            cached = self.cache.get(task, state.url, self._dom_state.dom_hash)
            if cached and cached.confidence >= self.CACHE_CONFIDENCE_THRESHOLD:
                # Prevent infinite loops: don't repeat same action on same page
                action_key = (state.url, cached.action_type, cached.selector or "")
                if action_key in self._executed_actions:
                    logger.info(f"Cache hit but already executed {cached.action_type} on this page - skipping")
                else:
                    logger.info(f"Cache HIT: {cached.action_type} (confidence={cached.confidence:.2f}, hits={cached.hit_count})")
                    self._cache_hits += 1
                    self._executed_actions.add(action_key)

                    action = Action(
                        action_type=cached.action_type,
                        selector=cached.selector,
                        value=cached.value,
                        reasoning=f"Cached action (hits={cached.hit_count})"
                    )
                    return await self._execute_and_continue(task, action, depth, history, from_cache=True)

        # === LAYER 2: Try dumb patterns ===
        pattern = find_dumb_pattern(task, self._dom_state.elements)
        if pattern:
            logger.info(f"Using dumb pattern: {pattern.name}")
            result = await execute_dumb_pattern(pattern, self.browser, task, self._dom_state.elements)
            if result["success"]:
                logger.info(f"Pattern executed: {result.get('actions', [])}")
                history.append(f"Pattern {pattern.name}: {result.get('actions', [])}")
                return await self._navigate_recursive(task, depth + 1, history)
            else:
                logger.info(f"Pattern failed: {result.get('reason')}")

        # === LAYER 3: Check learned patterns ===
        learned_context = None
        if self.learner:
            query = f"{task} {state.url} {state.title}"
            relevant = self.learner.search_patterns(query, limit=3)
            if relevant and relevant[0][1] >= self.PATTERN_SIMILARITY_THRESHOLD:
                learned_context = self.learner.format_patterns_for_prompt([p for p, _ in relevant])
                logger.info(f"Found {len(relevant)} relevant patterns (top sim={relevant[0][1]:.2f})")

        # === LAYER 4/5/6: LLM Decision (with possible debate/RSA) ===
        action = await self._decide_action(task, state, history, learned_context)

        # Check if done
        if action.action_type == "done":
            return {
                "success": True,
                "result": action.value,
                "depth": depth,
                "history": history,
            }

        return await self._execute_and_continue(task, action, depth, history, from_cache=False)

    async def _decide_action(
        self,
        task: str,
        state: PageState,
        history: List[str],
        learned_context: Optional[str]
    ) -> Action:
        """
        Multi-layer decision making.

        For simple pages: single LLM call
        For complex pages: RSA aggregation
        For uncertain results: multi-agent debate
        """
        elements_text = format_for_llm(self._dom_state.elements)

        # Include read result in history if available
        llm_history = history.copy()
        if self._last_read_result:
            llm_history.append(f"READ RESULT: \"{self._last_read_result}\"")

        # Check if page is complex (many elements)
        is_complex = len(self._dom_state.elements) > self.COMPLEX_PAGE_ELEMENT_COUNT

        if is_complex and self.reasoner:
            # Use RSA aggregation for complex pages
            logger.info(f"Complex page ({len(self._dom_state.elements)} elements) - using RSA aggregation")
            self._llm_calls += len(self.reasoner.models) * len(self.reasoner.languages)

            decision = self.reasoner.reason(
                task=task,
                url=state.url,
                title=state.title,
                elements=elements_text,
                history=llm_history,
            )

            action = Action(
                action_type=decision.action_type,
                selector=decision.selector,
                value=decision.value,
                reasoning=decision.reasoning,
            )

            logger.info(f"RSA decision: {action.action_type} (consensus={decision.consensus_level:.2f})")

        else:
            # Simple LLM call with few-shot context
            self._llm_calls += 1

            # Combine few-shot examples with learned patterns
            context = ""
            if self._few_shot_context:
                context = self._few_shot_context + "\n\n"
            if learned_context:
                context += learned_context

            action = self.llm.decide_action(
                task=task,
                url=state.url,
                title=state.title,
                elements=elements_text,
                history=llm_history,
                learned_patterns=context if context else None,
            )
            logger.info(f"LLM action: {action.action_type} - {action.reasoning}")

        # Check if we should debate
        if self.use_debate and self._is_uncertain(action, len(history), history):
            logger.info("Uncertain decision - running debate")
            action = await self._run_debate(task, state, elements_text, llm_history, action)

        # SAFEGUARD: Loop detection and force-done for invalid/repeated actions
        valid_actions = {"navigate", "click", "fill", "scroll", "press", "read", "done"}

        # Count how many times we've asked LLM on same page
        same_page_calls = sum(1 for h in history if "read" in h.lower() or "click" in h.lower())

        # Check for looping (>3 LLM calls without progress)
        is_looping = same_page_calls >= 3

        if action.action_type not in valid_actions:
            # Invalid action - use read result or try reading content
            if self._last_read_result:
                logger.info(f"Forcing 'done' - invalid action, using read result")
                action = Action(action_type="done", value=self._last_read_result, reasoning="Invalid action")
            else:
                logger.info(f"Invalid action '{action.action_type}' - forcing read")
                action = Action(action_type="read", selector="[11]", reasoning="Forcing content read")

        elif is_looping and self._last_read_result:
            # Looping with a read result - check if it's actual content
            site_names = ["hacker news", "github", "wikipedia", "google", "reddit"]
            is_just_site_name = any(s in self._last_read_result.lower() for s in site_names) and len(self._last_read_result) < 30

            if is_just_site_name:
                # Bad read result - try to read actual content
                for el in self._dom_state.elements[10:25]:  # Skip nav elements
                    if el.text and len(el.text) > 20 and el.element_type == "link":
                        href = el.attributes.get("href", "")
                        if href.startswith("http"):  # External link = likely content
                            logger.info(f"Loop with bad content - reading [{el.index}]: {el.text[:40]}")
                            action = Action(action_type="read", selector=f"[{el.index}]", reasoning="Finding real content")
                            break
            else:
                # Good read result - force done
                logger.info(f"Loop detected - forcing done with: {self._last_read_result[:50]}")
                action = Action(action_type="done", value=self._last_read_result, reasoning="Loop detected")

        elif is_looping and not self._last_read_result:
            # Looping without read result - try to read first content element
            for el in self._dom_state.elements[10:20]:  # Skip nav, look at content
                if el.text and len(el.text) > 15:
                    logger.info(f"Loop detected - forcing read of [{el.index}]")
                    action = Action(action_type="read", selector=f"[{el.index}]", reasoning="Breaking loop")
                    break

        return action

    async def _run_debate(
        self,
        task: str,
        state: PageState,
        elements_text: str,
        history: List[str],
        original_action: Action
    ) -> Action:
        """Run multi-agent debate for uncertain decisions."""
        self._llm_calls += 3  # 3 models in debate

        debate_result = run_debate(
            task=task,
            url=state.url,
            title=state.title,
            elements=elements_text,
            history=history,
        )

        logger.info(f"Debate result: {debate_result.vote_count} (consensus={debate_result.consensus})")

        if debate_result.consensus and debate_result.winning_action != original_action.action_type:
            logger.info(f"Overriding {original_action.action_type} with debate winner: {debate_result.winning_action}")
            return Action(
                action_type=debate_result.winning_action,
                selector=debate_result.winning_selector,
                value=debate_result.winning_value,
                reasoning=f"Debate consensus ({debate_result.vote_count})",
            )

        return original_action

    async def _execute_and_continue(
        self,
        task: str,
        action: Action,
        depth: int,
        history: List[str],
        from_cache: bool = False
    ) -> dict:
        """Execute action and continue navigation."""
        success, action_result = await self._execute_action_with_result(action)

        # Record action for trajectory storage
        read_result = self._last_read_result if action.action_type == "read" else None
        self._record_action(self._dom_state.url, action, success, read_result)

        if success:
            # Record success in cache
            if self.cache and not from_cache:
                self.cache.store(
                    task=task,
                    url=self._dom_state.url,
                    dom_hash=self._dom_state.dom_hash,
                    action_type=action.action_type,
                    selector=action.selector or "",
                    value=action.value,
                    confidence=0.8,
                    success=True
                )

            history.append(f"{action.action_type}: {action_result}")

            # For read actions, don't increment depth
            if action.action_type == "read":
                return await self._navigate_recursive(task, depth, history)

            # Update tree
            new_state = await self.browser.get_state()
            child = self._create_node(new_state.url, depth + 1, action.action_type, self._current)
            self._current.children.append(child)
            self._current = child

            return await self._navigate_recursive(task, depth + 1, history)

        else:
            # Record failure
            if self.cache:
                self.cache.record_failure(task, self._dom_state.url, self._dom_state.dom_hash)

            logger.warning(f"Action failed, trying alternatives")
            return await self._try_alternatives(task, depth, history)

    async def _try_alternatives(self, task: str, depth: int, history: List[str] = None) -> dict:
        """Try alternative actions when primary fails."""
        clickable = [el for el in self._dom_state.elements if el.element_type in ('link', 'button')]

        for i, el in enumerate(clickable[:self.max_retries]):
            logger.info(f"Trying alternative {i+1}: click [{el.index}] '{el.text[:30]}'")

            success = await self.browser.click(el.selector)
            if success:
                return await self._navigate_recursive(task, depth + 1)

            await self.browser.navigate("javascript:history.back()")
            await asyncio.sleep(0.5)

        return {"success": False, "reason": "All alternatives exhausted"}

    async def _execute_action_with_result(self, action: Action) -> tuple:
        """Execute an action and return (success, result_info)."""
        action_type = action.action_type.lower()

        if action_type == "navigate":
            if action.value:
                await self.browser.navigate(action.value)
                return True, action.value
            return False, "No URL"

        elif action_type == "click":
            selector = self._resolve_selector(action.selector)
            if selector:
                success = await self.browser.click(selector)
                return success, f"selector={selector}"
            return False, "No selector"

        elif action_type == "fill":
            selector = self._resolve_selector(action.selector)
            if selector and action.value:
                success = await self.browser.fill(selector, action.value)
                return success, f"filled with '{action.value[:30]}'"
            return False, "No selector or value"

        elif action_type == "press":
            if action.value:
                success = await self.browser.press(action.value)
                return success, action.value
            return False, "No key"

        elif action_type == "scroll":
            success = await self.browser.scroll(action.value or "down")
            return success, action.value or "down"

        elif action_type == "read":
            selector = self._resolve_selector(action.selector)
            text = None

            # Try specific selector first
            if selector:
                text = await self.browser.read_text(selector)

            # Fallback: find content elements (skip navigation/branding)
            if not text and self._dom_state.elements:
                text = self._find_content_text()

            # Secondary fallback: generic heading selectors
            if not text:
                generic_selectors = ["h1", "[role='heading']", "main h1"]
                for fallback in generic_selectors:
                    text = await self.browser.read_text(fallback)
                    if text and len(text) > 3:
                        logger.info(f"Read fallback '{fallback}' worked")
                        break

            if text:
                self._last_read_result = text
                logger.info(f"Read: \"{text[:100]}\"")
                return True, f"got: \"{text[:50]}...\""

            return False, "Could not read content"

        return False, f"Unknown action: {action_type}"

    def _find_content_text(self) -> Optional[str]:
        """
        Find actual content text, skipping navigation/branding elements.

        Heuristics:
        1. Skip elements with text matching common nav patterns
        2. Skip very short text (likely nav labels)
        3. Prefer links (a tags) as they often contain content titles
        4. Skip elements that look like site branding (matches domain)
        """
        # Common navigation/branding/auth patterns to skip
        skip_patterns = [
            "hacker news", "hn", "home", "menu", "login", "logout", "sign in",
            "sign up", "search", "about", "contact", "new", "past", "comments",
            "ask", "show", "jobs", "submit", "guidelines", "faq", "lists",
            "api", "security", "legal", "privacy", "apply", "more", "top",
            "notification", "watch", "star", "fork", "sponsor", "must be signed in",
            "you must be", "signed in to", "log in to", "create account",
            "cookie", "accept", "decline", "dismiss", "close", "skip",
        ]

        # Get domain name for branding detection
        domain_name = ""
        if self._dom_state.url:
            from urllib.parse import urlparse
            parsed = urlparse(self._dom_state.url)
            domain_name = parsed.netloc.replace("www.", "").split(".")[0].lower()

        candidates = []

        for el in self._dom_state.elements:
            if not el.text or len(el.text) < 15:
                continue

            text_lower = el.text.lower().strip()

            # Skip if matches skip patterns (also partial matches)
            if any(pat in text_lower for pat in skip_patterns):
                continue

            # Skip if matches domain (likely branding)
            if domain_name and text_lower == domain_name:
                continue

            # Score this element
            score = len(el.text)  # Longer = better (more likely real content)

            # Prefer links (a tags) - often story/article titles
            if el.element_type == 'link':
                score += 50
                # Extra bonus for links that appear to have external URLs
                if hasattr(el, 'href') and el.href and not el.href.startswith('#'):
                    score += 20

            # Slight preference for elements later in the DOM (skip headers)
            if el.index > 5:
                score += 10

            candidates.append((el, score))

        if candidates:
            # Sort by score descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0][0]
            logger.info(f"Read fallback: selected [{best.index}] '{best.text[:50]}...' (score={candidates[0][1]})")
            return best.text

        return None

    def _resolve_selector(self, selector: Optional[str]) -> Optional[str]:
        """Resolve element index to selector."""
        if not selector:
            return None

        import re
        match = re.match(r"\[?(\d+)\]?", selector)
        if match:
            index = int(match.group(1))
            for el in self._dom_state.elements:
                if el.index == index:
                    return el.selector
            return None
        return selector

    def _is_uncertain(self, action: Action, depth: int, history: List[str]) -> bool:
        """Detect if an action seems uncertain and should trigger debate."""
        # Early "done" without having read anything
        if action.action_type == "done" and depth < 2:
            has_read = any("read" in h.lower() for h in (history or []))
            if not has_read:
                return True

        # Missing or very short reasoning
        if not action.reasoning or len(action.reasoning) < 10:
            return True

        return False

    def _is_complex_task(self, task: str) -> bool:
        """Detect if a task is complex and needs decomposition."""
        task_lower = task.lower()

        # Simple info tasks don't need decomposition
        simple_patterns = [
            "what is", "tell me", "find", "read", "get", "show",
            "title", "heading", "name", "description"
        ]
        if any(p in task_lower for p in simple_patterns):
            return False

        # Multi-step indicators that actually require decomposition
        complex_indicators = [
            " and then ",
            " after that ",
            " finally ",
            " next step ",
            ", then ",
            "step 1",
            "step 2",
        ]
        return any(ind in task_lower for ind in complex_indicators) or len(task) > 300

    def _create_node(self, url: str, depth: int, action: str = None, parent: NavNode = None) -> NavNode:
        """Create a navigation node."""
        self._node_counter += 1
        return NavNode(
            id=self._node_counter,
            url=url,
            depth=depth,
            action_taken=action,
            parent=parent,
        )

    def _store_trajectory(self, task: str, start_url: str, result: dict):
        """Store successful trajectory to memory for future few-shot retrieval."""
        from datetime import datetime

        trajectory = Trajectory(
            id=None,
            task=task,
            start_url=start_url,
            success=result.get("success", False),
            result=result.get("result"),
            actions=self._action_history,
            created_at=datetime.now().isoformat(),
            total_llm_calls=self._llm_calls,
            total_depth=result.get("depth", 0),
        )

        self.memory.store(trajectory)
        logger.info(f"Stored trajectory with {len(self._action_history)} actions")

    def _record_action(self, url: str, action: Action, success: bool, read_result: str = None):
        """Record an action for trajectory storage."""
        step = ActionStep(
            step=len(self._action_history) + 1,
            url=url,
            action_type=action.action_type,
            selector=action.selector,
            value=action.value,
            success=success,
            read_result=read_result,
        )
        self._action_history.append(step)


def promote_cache_to_patterns(cache: ActionCache, learner: Learner, min_hits: int = 3):
    """Promote high-hit cached actions to learned patterns."""
    if not cache or not learner:
        return 0

    high_hit_actions = cache.get_high_hit_actions(min_hits=min_hits)
    promoted = 0

    for action in high_hit_actions:
        # Create pattern from cached action
        pattern_id = f"cache_{action['task'][:20]}_{action['action_type']}"
        domain = action['url'].split('/')[2] if '/' in action['url'] else None

        # Store as learned pattern
        learner.store_pattern(
            pattern_id=pattern_id,
            domain=domain,
            pattern_type="success",
            trigger=action['task'],
            action=f"{action['action_type']} {action.get('selector', '')}".strip(),
            confidence=min(0.95, action['confidence']),
        )
        promoted += 1
        logger.info(f"Promoted cache entry to pattern: {pattern_id}")

    return promoted


async def test_navigator():
    """Test recursive navigator with caching."""
    nav = RecursiveNavigator(
        model="llama3.1:8b",
        max_depth=8,
        use_cache=True,
        use_recursive=False,  # Fast mode
    )
    nav.use_debate = False

    result = await nav.navigate(
        task="Go to Hacker News and tell me the title of the top story",
        start_url="https://news.ycombinator.com",
    )

    print(f"\n=== RESULT ===")
    print(f"Success: {result.success}")
    print(f"Result: {result.result}")
    print(f"Depth: {result.depth}")
    print(f"Cache hits: {result.cache_hits}")
    print(f"LLM calls: {result.llm_calls}")

    # Promote high-hit cache entries to learned patterns
    if nav.cache and nav.learner:
        promoted = promote_cache_to_patterns(nav.cache, nav.learner, min_hits=2)
        print(f"Promoted {promoted} cached actions to learned patterns")


if __name__ == "__main__":
    asyncio.run(test_navigator())
