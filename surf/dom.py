"""DOM extraction and simplification for LLMs."""

import re
from dataclasses import dataclass
from typing import Optional
from bs4 import BeautifulSoup


@dataclass
class InteractiveElement:
    """An interactive element on the page."""
    index: int
    tag: str
    element_type: Optional[str]  # button, link, input, etc.
    text: str
    selector: str
    attributes: dict


def extract_interactive_elements(html: str, max_elements: int = 50) -> list[InteractiveElement]:
    """
    Extract interactive elements from HTML.

    Returns a simplified list with indices for easy reference by LLMs.
    Target: ~200-400 tokens for typical pages.
    """
    soup = BeautifulSoup(html, "html.parser")
    elements = []
    index = 1

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
        text = el.get_text(strip=True)[:100]  # Truncate long text
        if not text:
            text = el.get("placeholder", "") or el.get("value", "") or el.get("aria-label", "")

        # Build selector
        selector = build_selector(el)

        # Collect relevant attributes
        attrs = {}
        for attr in ["href", "name", "id", "class", "placeholder", "value", "aria-label"]:
            if el.get(attr):
                val = el.get(attr)
                if isinstance(val, list):
                    val = " ".join(val)
                attrs[attr] = val[:50]  # Truncate

        elements.append(InteractiveElement(
            index=index,
            tag=tag_name,
            element_type=el_type,
            text=text[:100],
            selector=selector,
            attributes=attrs,
        ))
        index += 1

    return elements


def build_selector(el) -> str:
    """Build a CSS selector for an element."""
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
        class_str = ".".join(classes[:2])  # First 2 classes
        return f"{el.name}.{class_str}"

    # Last resort: tag + text content
    text = el.get_text(strip=True)[:20]
    if text:
        return f"{el.name}:has-text('{text}')"

    return el.name


def format_for_llm(elements: list[InteractiveElement]) -> str:
    """
    Format elements for LLM consumption.

    Output format:
    [1] button "Sign In"
    [2] input[email] placeholder="Email"
    [3] link "Forgot password?" href="/reset"
    """
    lines = []
    for el in elements:
        # Build description
        desc = f"[{el.index}] {el.element_type}"

        if el.text:
            desc += f' "{el.text}"'

        # Add key attributes
        if el.attributes.get("placeholder"):
            desc += f' placeholder="{el.attributes["placeholder"]}"'
        if el.attributes.get("href"):
            href = el.attributes["href"]
            if len(href) > 30:
                href = href[:27] + "..."
            desc += f' href="{href}"'

        lines.append(desc)

    return "\n".join(lines)


def test_dom_extraction():
    """Test DOM extraction on sample HTML."""
    sample_html = """
    <html>
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

    elements = extract_interactive_elements(sample_html)
    formatted = format_for_llm(elements)
    print("Extracted elements:")
    print(formatted)
    print(f"\nTotal: {len(elements)} elements")


if __name__ == "__main__":
    test_dom_extraction()
