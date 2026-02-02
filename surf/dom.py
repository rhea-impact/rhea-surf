"""
DOM extraction and simplification for LLMs.

Two extraction modes:
1. Playwright JS injection (primary) - Full visibility, paint order, computed styles
2. BeautifulSoup static (fallback) - Fast, works without browser
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

# BeautifulSoup for fallback mode
from bs4 import BeautifulSoup


@dataclass
class Element:
    """An interactive element on the page."""
    index: int
    tag: str
    element_type: str
    text: str
    selector: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    bbox: Optional[Dict[str, int]] = None
    is_visible: bool = True
    is_interactive: bool = True


@dataclass
class DOMState:
    """Current DOM state for a page."""
    elements: List[Element]
    url: str
    title: str
    dom_hash: str = ""

    def __post_init__(self):
        if not self.dom_hash:
            self.dom_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Hash interactive elements for cache matching."""
        # Only hash interactive elements for stability
        structure = [
            (e.tag, e.text[:50], tuple(sorted(e.attributes.items())))
            for e in self.elements if e.is_interactive
        ]
        return hashlib.md5(json.dumps(structure, sort_keys=True).encode()).hexdigest()[:12]


# Load JS extraction script
_JS_SCRIPT_PATH = Path(__file__).parent / "js" / "buildDomTree.js"
_BUILD_DOM_TREE_JS: Optional[str] = None


def _get_js_script() -> str:
    """Load the buildDomTree.js script, cached."""
    global _BUILD_DOM_TREE_JS
    if _BUILD_DOM_TREE_JS is None:
        if _JS_SCRIPT_PATH.exists():
            _BUILD_DOM_TREE_JS = _JS_SCRIPT_PATH.read_text()
        else:
            raise FileNotFoundError(f"buildDomTree.js not found at {_JS_SCRIPT_PATH}")
    return _BUILD_DOM_TREE_JS


async def extract_dom_playwright(page) -> DOMState:
    """
    Extract DOM using Playwright's JS injection.

    This is the primary extraction method - uses computed styles,
    paint order, and viewport filtering.

    Args:
        page: Playwright Page object

    Returns:
        DOMState with extracted elements
    """
    js_script = _get_js_script()

    # Inject and execute the extraction script
    raw_elements = await page.evaluate(js_script)

    # Convert to Element objects
    elements = [
        Element(
            index=e['index'],
            tag=e['tag'],
            element_type=e['type'],
            text=e.get('text', ''),
            selector=e['selector'],
            attributes=e.get('attributes', {}),
            bbox=e.get('bbox'),
            is_visible=e.get('visible', True),
            is_interactive=e.get('interactive', True),
        )
        for e in raw_elements
    ]

    return DOMState(
        elements=elements,
        url=page.url,
        title=await page.title() or ""
    )


def extract_dom_static(html: str, url: str = "", title: str = "", max_elements: int = 50) -> DOMState:
    """
    Extract DOM using BeautifulSoup (static HTML parsing).

    This is the fallback method - faster but misses JS-rendered content.

    Args:
        html: Raw HTML string
        url: Page URL
        title: Page title
        max_elements: Maximum elements to extract

    Returns:
        DOMState with extracted elements
    """
    soup = BeautifulSoup(html, "html.parser")
    elements = []
    index = 1

    # Extract title if not provided
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

    # Remove script, style, and hidden elements
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    # Find interactive elements
    interactive_tags = soup.find_all(["a", "button", "input", "select", "textarea"])

    for el in interactive_tags:
        if index > max_elements:
            break

        # Skip hidden elements
        if el.get("type") == "hidden":
            continue
        if el.get("style") and "display:none" in el.get("style", "").replace(" ", ""):
            continue

        # Determine element type
        tag_name = el.name
        if tag_name == "a":
            el_type = "link"
        elif tag_name == "button":
            el_type = "button"
        elif tag_name == "input":
            el_type = el.get("type", "text")
        elif tag_name == "select":
            el_type = "dropdown"
        elif tag_name == "textarea":
            el_type = "textarea"
        else:
            el_type = tag_name

        # Get text content
        text = el.get_text(strip=True)[:100]
        if not text:
            text = el.get("placeholder", "") or el.get("value", "") or el.get("aria-label", "")

        # Build selector
        selector = _build_selector_static(el)

        # Collect relevant attributes
        attrs = {}
        for attr in ["href", "name", "id", "class", "placeholder", "value", "aria-label"]:
            if el.get(attr):
                val = el.get(attr)
                if isinstance(val, list):
                    val = " ".join(val)
                attrs[attr] = str(val)[:50]

        elements.append(Element(
            index=index,
            tag=tag_name,
            element_type=el_type,
            text=text[:100],
            selector=selector,
            attributes=attrs,
            bbox=None,  # Not available in static mode
            is_visible=True,
            is_interactive=True,
        ))
        index += 1

    return DOMState(elements=elements, url=url, title=title)


def _build_selector_static(el) -> str:
    """Build a CSS selector for an element (BeautifulSoup)."""
    # Prefer ID
    if el.get("id"):
        return f"#{el.get('id')}"

    # Try data-testid
    if el.get("data-testid"):
        return f"[data-testid='{el.get('data-testid')}']"

    # Try name
    if el.get("name"):
        return f"[name='{el.get('name')}']"

    # Fall back to tag + class
    classes = el.get("class", [])
    if classes:
        class_str = ".".join(classes[:2])
        return f"{el.name}.{class_str}"

    # Last resort: tag + text content
    text = el.get_text(strip=True)[:20]
    if text:
        return f"{el.name}:has-text('{text}')"

    return el.name


def format_for_llm(elements: List[Element], include_bbox: bool = False) -> str:
    """
    Format elements for LLM consumption.

    Output format:
    [1] button "Sign In"
    [2] input[email] placeholder="Email"
    [3] link "Forgot password?" href="/reset"

    Args:
        elements: List of Element objects
        include_bbox: Whether to include bounding box info

    Returns:
        Formatted string for LLM prompt
    """
    # Navigation patterns to label (includes site names)
    nav_patterns = [
        "home", "new", "past", "comments", "ask", "show", "jobs", "submit",
        "login", "logout", "sign in", "sign up", "search", "menu", "about",
        "contact", "help", "settings", "profile", "account", "hide", "discuss",
        "hacker news", "github", "wikipedia", "google", "reddit", "twitter"
    ]

    lines = []
    for el in elements:
        # Build description
        desc = f"[{el.index}] {el.element_type}"

        # Check if this looks like navigation
        is_nav = False
        if el.text:
            text_lower = el.text.lower().strip()
            href = el.attributes.get("href", "")
            # Mark as NAV if: matches nav pattern, OR short internal link (not http)
            is_internal = not href.startswith("http")
            is_nav = text_lower in nav_patterns or (len(text_lower) <= 8 and is_internal and el.element_type == "link")

        if el.text:
            desc += f' "{el.text}"'

        # Label navigation elements
        if is_nav:
            desc += " [NAV]"

        # Add key attributes
        if el.attributes.get("placeholder"):
            desc += f' placeholder="{el.attributes["placeholder"]}"'
        if el.attributes.get("href"):
            href = el.attributes["href"]
            if len(href) > 30:
                href = href[:27] + "..."
            desc += f' href="{href}"'

        # Optional bbox
        if include_bbox and el.bbox:
            desc += f" @({el.bbox['x']},{el.bbox['y']})"

        lines.append(desc)

    return "\n".join(lines)


def format_dom_state(state: DOMState, include_bbox: bool = False) -> str:
    """
    Format full DOMState for LLM consumption.

    Args:
        state: DOMState object
        include_bbox: Whether to include bounding box info

    Returns:
        Formatted string including URL, title, and elements
    """
    header = f"Page: {state.title}\nURL: {state.url}\n\nInteractive elements:\n"
    elements_str = format_for_llm(state.elements, include_bbox)
    return header + elements_str


# Backward compatibility aliases
InteractiveElement = Element


def extract_interactive_elements(html: str, max_elements: int = 50) -> List[Element]:
    """
    Extract interactive elements from HTML (backward compatibility).

    Use extract_dom_static() or extract_dom_playwright() for new code.
    """
    state = extract_dom_static(html, max_elements=max_elements)
    return state.elements


def test_dom_extraction():
    """Test DOM extraction on sample HTML."""
    sample_html = """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <nav>
            <a href="/" id="logo">Home</a>
            <a href="/about">About</a>
            <button class="menu-toggle">Menu</button>
        </nav>
        <main>
            <form>
                <input type="email" name="email" placeholder="Enter email">
                <input type="password" name="password" placeholder="Password">
                <button type="submit">Sign In</button>
            </form>
            <a href="/forgot">Forgot password?</a>
        </main>
    </body>
    </html>
    """

    state = extract_dom_static(sample_html, url="https://example.com", title="Test Page")
    formatted = format_dom_state(state)
    print("Extracted DOM state:")
    print(formatted)
    print(f"\nTotal: {len(state.elements)} elements")
    print(f"DOM hash: {state.dom_hash}")


if __name__ == "__main__":
    test_dom_extraction()
