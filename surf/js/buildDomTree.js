/**
 * buildDomTree.js - Extract interactive DOM elements for LLM consumption
 *
 * Inspired by Browser Use's approach:
 * - Paint order traversal (z-index aware)
 * - Visibility checks via getComputedStyle
 * - Viewport filtering via getBoundingClientRect
 * - Interactive element detection
 * - Sequential indexing [1], [2], [3]
 */

(function() {
    'use strict';

    const CONFIG = {
        maxElements: 100,
        maxTextLength: 100,
        viewportMargin: 50, // Include elements slightly outside viewport
        interactiveTags: new Set([
            'a', 'button', 'input', 'select', 'textarea',
            'details', 'summary', 'dialog'
        ]),
        // Content elements to capture even if not interactive (headings, descriptions)
        contentTags: new Set([
            'h1', 'h2', 'h3', 'p', 'article', 'main', 'figcaption'
        ]),
        interactiveRoles: new Set([
            'button', 'link', 'checkbox', 'radio', 'textbox',
            'combobox', 'listbox', 'menu', 'menuitem', 'tab',
            'switch', 'slider', 'spinbutton', 'searchbox',
            'heading', 'main', 'article'
        ]),
        skipTags: new Set(['script', 'style', 'noscript', 'svg', 'path']),
    };

    /**
     * Check if element is visible using computed styles
     */
    function isVisible(el) {
        if (!el || el.nodeType !== Node.ELEMENT_NODE) return false;

        const style = window.getComputedStyle(el);

        // Check display and visibility
        if (style.display === 'none') return false;
        if (style.visibility === 'hidden') return false;
        if (parseFloat(style.opacity) === 0) return false;

        // Check if element has dimensions
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return false;

        return true;
    }

    /**
     * Check if element is within viewport (with margin)
     */
    function isInViewport(el) {
        const rect = el.getBoundingClientRect();
        const margin = CONFIG.viewportMargin;

        return (
            rect.bottom >= -margin &&
            rect.right >= -margin &&
            rect.top <= (window.innerHeight + margin) &&
            rect.left <= (window.innerWidth + margin)
        );
    }

    /**
     * Check if element is interactive
     */
    function isInteractive(el) {
        const tag = el.tagName.toLowerCase();

        // Check tag
        if (CONFIG.interactiveTags.has(tag)) return true;

        // Check role attribute
        const role = el.getAttribute('role');
        if (role && CONFIG.interactiveRoles.has(role)) return true;

        // Check for click handlers
        if (el.onclick || el.hasAttribute('onclick')) return true;

        // Check for tabindex (indicates keyboard interactive)
        const tabindex = el.getAttribute('tabindex');
        if (tabindex !== null && tabindex !== '-1') return true;

        // Check contenteditable
        if (el.isContentEditable) return true;

        // Check for cursor: pointer style (common for clickable divs)
        const style = window.getComputedStyle(el);
        if (style.cursor === 'pointer') return true;

        return false;
    }

    /**
     * Check if element is important content (headings, descriptions)
     */
    function isContentElement(el) {
        const tag = el.tagName.toLowerCase();

        // Check if it's a content tag with meaningful text
        if (CONFIG.contentTags.has(tag)) {
            const text = getText(el);
            // Only include if has meaningful text (not just whitespace/short)
            return text.length >= 10;
        }

        // Check for heading role
        const role = el.getAttribute('role');
        if (role === 'heading') {
            const text = getText(el);
            return text.length >= 5;
        }

        return false;
    }

    /**
     * Get text content, preferring visible text
     */
    function getText(el) {
        // Try aria-label first
        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel) return ariaLabel.trim();

        // Try placeholder for inputs
        const placeholder = el.getAttribute('placeholder');
        if (placeholder) return placeholder.trim();

        // Try value for inputs
        if (el.value) return el.value.trim();

        // Try title
        const title = el.getAttribute('title');
        if (title) return title.trim();

        // Get inner text (visible text only)
        let text = el.innerText || el.textContent || '';
        text = text.trim().replace(/\s+/g, ' ');

        return text.substring(0, CONFIG.maxTextLength);
    }

    /**
     * Determine element type for LLM
     */
    function getElementType(el) {
        const tag = el.tagName.toLowerCase();

        switch (tag) {
            case 'a': return 'link';
            case 'button': return 'button';
            case 'input':
                const inputType = el.getAttribute('type') || 'text';
                if (inputType === 'submit') return 'button';
                if (inputType === 'checkbox') return 'checkbox';
                if (inputType === 'radio') return 'radio';
                return inputType;
            case 'select': return 'dropdown';
            case 'textarea': return 'textarea';
            case 'details': return 'expandable';
            case 'summary': return 'expander';
            default:
                // Check role
                const role = el.getAttribute('role');
                if (role) return role;
                // Check if clickable
                if (el.onclick || window.getComputedStyle(el).cursor === 'pointer') {
                    return 'clickable';
                }
                return tag;
        }
    }

    /**
     * Build best selector for element
     */
    function buildSelector(el) {
        // Prefer ID
        if (el.id) return `#${el.id}`;

        // Try data-testid
        const testId = el.getAttribute('data-testid');
        if (testId) return `[data-testid="${testId}"]`;

        // Try name
        const name = el.getAttribute('name');
        if (name) return `[name="${name}"]`;

        // Try aria-label
        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel) return `[aria-label="${ariaLabel}"]`;

        // Try unique class combination
        const classes = Array.from(el.classList).slice(0, 2);
        if (classes.length > 0) {
            const selector = `${el.tagName.toLowerCase()}.${classes.join('.')}`;
            // Check if unique
            if (document.querySelectorAll(selector).length === 1) {
                return selector;
            }
        }

        // Try text-based selector for buttons/links
        const text = getText(el);
        if (text && (el.tagName === 'BUTTON' || el.tagName === 'A')) {
            const shortText = text.substring(0, 30);
            return `${el.tagName.toLowerCase()}:has-text("${shortText}")`;
        }

        // Fall back to nth-child path
        return getNthChildPath(el);
    }

    /**
     * Build nth-child path for element
     */
    function getNthChildPath(el, maxDepth = 3) {
        const path = [];
        let current = el;
        let depth = 0;

        while (current && current !== document.body && depth < maxDepth) {
            const parent = current.parentElement;
            if (!parent) break;

            const siblings = Array.from(parent.children).filter(
                c => c.tagName === current.tagName
            );
            const index = siblings.indexOf(current) + 1;

            if (siblings.length === 1) {
                path.unshift(current.tagName.toLowerCase());
            } else {
                path.unshift(`${current.tagName.toLowerCase()}:nth-of-type(${index})`);
            }

            current = parent;
            depth++;
        }

        return path.join(' > ');
    }

    /**
     * Get bounding box for element
     */
    function getBbox(el) {
        const rect = el.getBoundingClientRect();
        return {
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            width: Math.round(rect.width),
            height: Math.round(rect.height)
        };
    }

    /**
     * Get z-index for paint order sorting
     */
    function getZIndex(el) {
        let zIndex = 0;
        let current = el;

        while (current && current !== document.body) {
            const style = window.getComputedStyle(current);
            const z = parseInt(style.zIndex, 10);
            if (!isNaN(z) && z > zIndex) {
                zIndex = z;
            }
            current = current.parentElement;
        }

        return zIndex;
    }

    /**
     * Extract relevant attributes
     */
    function getAttributes(el) {
        const attrs = {};
        const relevant = ['href', 'name', 'id', 'class', 'placeholder', 'value',
                         'aria-label', 'type', 'role', 'disabled', 'readonly'];

        for (const attr of relevant) {
            const val = el.getAttribute(attr);
            if (val) {
                // Truncate long values
                attrs[attr] = val.length > 100 ? val.substring(0, 100) : val;
            }
        }

        // Handle class as string
        if (attrs.class && Array.isArray(attrs.class)) {
            attrs.class = attrs.class.join(' ');
        }

        return attrs;
    }

    /**
     * Main extraction function - walks DOM in paint order
     */
    function extractElements() {
        const results = [];
        const seen = new Set();

        // Get all elements
        const allElements = document.querySelectorAll('*');
        const candidates = [];

        for (const el of allElements) {
            // Skip certain tags
            if (CONFIG.skipTags.has(el.tagName.toLowerCase())) continue;

            // Check visibility
            if (!isVisible(el)) continue;

            // Check viewport
            if (!isInViewport(el)) continue;

            // Check if interactive OR important content
            const interactive = isInteractive(el);
            const isContent = isContentElement(el);
            if (!interactive && !isContent) continue;

            // Skip if parent is also interactive (avoid duplicates)
            const parent = el.parentElement;
            if (parent && isInteractive(parent) &&
                parent.tagName.toLowerCase() !== 'form') {
                // But keep if this element has different text/function
                const parentText = getText(parent);
                const thisText = getText(el);
                if (parentText === thisText) continue;
            }

            candidates.push({
                el,
                zIndex: getZIndex(el),
                bbox: getBbox(el),
                interactive,
                isContent
            });
        }

        // Sort by paint order (z-index, then DOM position for same z-index)
        candidates.sort((a, b) => {
            if (a.zIndex !== b.zIndex) return b.zIndex - a.zIndex;
            // Higher y (lower on page) comes later
            return a.bbox.y - b.bbox.y;
        });

        // Assign indices and build results
        let index = 1;
        for (const { el, bbox, interactive, isContent } of candidates) {
            if (index > CONFIG.maxElements) break;

            // Dedupe by selector
            const selector = buildSelector(el);
            if (seen.has(selector)) continue;
            seen.add(selector);

            const tag = el.tagName.toLowerCase();
            let type = getElementType(el);
            // Mark content-only elements
            if (!interactive && isContent) {
                type = tag.startsWith('h') ? 'heading' : 'text';
            }

            results.push({
                index,
                tag,
                type,
                text: getText(el),
                selector,
                attributes: getAttributes(el),
                bbox,
                visible: true,
                interactive
            });

            index++;
        }

        return results;
    }

    // Execute and return
    return extractElements();
})();
