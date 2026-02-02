"""
Test suite for rhea-surf navigator with varying complexity levels.

Complexity levels:
- SIMPLE: Single action, obvious target (1-2 LLM calls)
- MEDIUM: 2-3 actions, clear path (3-5 LLM calls)
- COMPLEX: Multi-step, requires reasoning (5-10 LLM calls)
- HARD: Ambiguous, needs debate/RSA (10+ LLM calls)

Run specific level:
    pytest tests/test_navigator.py -k simple
    pytest tests/test_navigator.py -k medium
    pytest tests/test_navigator.py -k complex

Run all:
    pytest tests/test_navigator.py -v
"""

import asyncio
import pytest
import time
from dataclasses import dataclass
from typing import Optional

from surf.navigator import RecursiveNavigator, NavigationResult


@dataclass
class TestCase:
    """A browser automation test case."""
    name: str
    task: str
    start_url: str
    complexity: str  # simple, medium, complex, hard
    expected_in_result: Optional[list[str]] = None  # Strings that should appear in result
    max_depth: int = 5
    max_llm_calls: int = 10
    timeout_seconds: int = 60


# =============================================================================
# TEST CASES BY COMPLEXITY
# =============================================================================

SIMPLE_TESTS = [
    TestCase(
        name="hn_top_story",
        task="What is the title of the top story on Hacker News?",
        start_url="https://news.ycombinator.com",
        complexity="simple",
        max_depth=3,
        max_llm_calls=5,
        timeout_seconds=30,
    ),
    TestCase(
        name="example_heading",
        task="What is the main heading on this page?",
        start_url="https://example.com",
        complexity="simple",
        expected_in_result=["example", "domain"],
        max_depth=4,
        max_llm_calls=6,
        timeout_seconds=45,
    ),
    TestCase(
        name="github_repo_name",
        task="What is the name of this GitHub repository? Read it from the page.",
        start_url="https://github.com/anthropics/anthropic-sdk-python",
        complexity="simple",
        expected_in_result=["anthropic", "python"],
        max_depth=4,
        max_llm_calls=6,
        timeout_seconds=45,
    ),
]

MEDIUM_TESTS = [
    TestCase(
        name="hn_second_story",
        task="Go to Hacker News and tell me the title of the SECOND story (not the first)",
        start_url="https://news.ycombinator.com",
        complexity="medium",
        max_depth=4,
        max_llm_calls=6,
        timeout_seconds=45,
    ),
    TestCase(
        name="wikipedia_search",
        task="Go to Wikipedia and search for 'artificial intelligence', then tell me the first sentence of the article",
        start_url="https://www.wikipedia.org",
        complexity="medium",
        expected_in_result=["intelligence", "machine"],
        max_depth=5,
        max_llm_calls=8,
        timeout_seconds=60,
    ),
    TestCase(
        name="hn_comments_count",
        task="Go to Hacker News, click on the comments link for the top story, and tell me how many comments there are",
        start_url="https://news.ycombinator.com",
        complexity="medium",
        max_depth=5,
        max_llm_calls=8,
        timeout_seconds=60,
    ),
]

COMPLEX_TESTS = [
    TestCase(
        name="hn_top_3_stories",
        task="Go to Hacker News and list the titles of the top 3 stories",
        start_url="https://news.ycombinator.com",
        complexity="complex",
        max_depth=5,
        max_llm_calls=10,
        timeout_seconds=90,
    ),
    TestCase(
        name="github_stars_count",
        task="Go to the pytorch/pytorch GitHub repo and tell me how many stars it has",
        start_url="https://github.com",
        complexity="complex",
        expected_in_result=["star", "k"],
        max_depth=6,
        max_llm_calls=12,
        timeout_seconds=90,
    ),
    TestCase(
        name="wikipedia_infobox",
        task="Go to Wikipedia, search for 'Eiffel Tower', and tell me its height from the infobox",
        start_url="https://www.wikipedia.org",
        complexity="complex",
        expected_in_result=["metre", "meter", "m", "330", "300"],
        max_depth=6,
        max_llm_calls=12,
        timeout_seconds=90,
    ),
]

HARD_TESTS = [
    TestCase(
        name="hn_top_story_author",
        task="Go to Hacker News, find the top story, click into the comments, and tell me the username of the person who submitted it",
        start_url="https://news.ycombinator.com",
        complexity="hard",
        max_depth=8,
        max_llm_calls=15,
        timeout_seconds=120,
    ),
    TestCase(
        name="compare_repos",
        task="Compare the star counts of tensorflow/tensorflow and pytorch/pytorch on GitHub. Which has more stars?",
        start_url="https://github.com",
        complexity="hard",
        expected_in_result=["tensorflow", "pytorch", "star"],
        max_depth=10,
        max_llm_calls=20,
        timeout_seconds=180,
    ),
]

ALL_TESTS = SIMPLE_TESTS + MEDIUM_TESTS + COMPLEX_TESTS + HARD_TESTS


# =============================================================================
# TEST CONFIGURATIONS
# =============================================================================

def get_navigator_config(complexity: str) -> dict:
    """Get navigator config based on test complexity."""
    configs = {
        "simple": {
            "model": "llama3.1:8b",
            "use_recursive": False,
            "use_debate": False,
            "use_cache": True,
        },
        "medium": {
            "model": "llama3.1:8b",
            "use_recursive": False,
            "use_debate": True,  # Enable debate for uncertain cases
            "use_cache": True,
        },
        "complex": {
            "model": "llama3.1:8b",
            "use_recursive": True,  # Enable RSA for complex pages
            "use_debate": True,
            "use_cache": True,
        },
        "hard": {
            "model": "qwen3:14b",
            "use_recursive": True,
            "use_debate": True,
            "use_cache": True,
        },
    }
    return configs.get(complexity, configs["medium"])


# =============================================================================
# TEST RUNNER
# =============================================================================

async def run_test_case(test: TestCase) -> dict:
    """Run a single test case and return results."""
    config = get_navigator_config(test.complexity)

    nav = RecursiveNavigator(
        model=config["model"],
        max_depth=test.max_depth,
        use_cache=config["use_cache"],
        use_recursive=config["use_recursive"],
    )
    nav.use_debate = config["use_debate"]

    start_time = time.time()

    try:
        result = await asyncio.wait_for(
            nav.navigate(test.task, test.start_url),
            timeout=test.timeout_seconds
        )
        elapsed = time.time() - start_time

        # Check expected strings in result
        validation_passed = True
        if test.expected_in_result and result.result:
            result_lower = result.result.lower()
            for expected in test.expected_in_result:
                if expected.lower() not in result_lower:
                    validation_passed = False
                    break

        return {
            "test_name": test.name,
            "complexity": test.complexity,
            "success": result.success,
            "result": result.result,
            "depth": result.depth,
            "cache_hits": result.cache_hits,
            "llm_calls": result.llm_calls,
            "elapsed_seconds": round(elapsed, 2),
            "within_llm_budget": result.llm_calls <= test.max_llm_calls,
            "validation_passed": validation_passed,
            "error": None,
        }

    except asyncio.TimeoutError:
        return {
            "test_name": test.name,
            "complexity": test.complexity,
            "success": False,
            "result": None,
            "depth": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "elapsed_seconds": test.timeout_seconds,
            "within_llm_budget": False,
            "validation_passed": False,
            "error": "Timeout",
        }
    except Exception as e:
        return {
            "test_name": test.name,
            "complexity": test.complexity,
            "success": False,
            "result": None,
            "depth": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "elapsed_seconds": time.time() - start_time,
            "within_llm_budget": False,
            "validation_passed": False,
            "error": str(e),
        }


# =============================================================================
# PYTEST FIXTURES AND TESTS
# =============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Simple tests
@pytest.mark.parametrize("test_case", SIMPLE_TESTS, ids=[t.name for t in SIMPLE_TESTS])
@pytest.mark.asyncio
async def test_simple(test_case):
    """Run simple complexity tests."""
    result = await run_test_case(test_case)
    print(f"\n{test_case.name}: {result}")
    assert result["success"], f"Test failed: {result.get('error') or result.get('result')}"
    assert result["within_llm_budget"], f"Exceeded LLM budget: {result['llm_calls']} > {test_case.max_llm_calls}"


# Medium tests
@pytest.mark.parametrize("test_case", MEDIUM_TESTS, ids=[t.name for t in MEDIUM_TESTS])
@pytest.mark.asyncio
async def test_medium(test_case):
    """Run medium complexity tests."""
    result = await run_test_case(test_case)
    print(f"\n{test_case.name}: {result}")
    assert result["success"], f"Test failed: {result.get('error') or result.get('result')}"


# Complex tests
@pytest.mark.parametrize("test_case", COMPLEX_TESTS, ids=[t.name for t in COMPLEX_TESTS])
@pytest.mark.asyncio
async def test_complex(test_case):
    """Run complex tests."""
    result = await run_test_case(test_case)
    print(f"\n{test_case.name}: {result}")
    assert result["success"], f"Test failed: {result.get('error') or result.get('result')}"


# Hard tests (marked as slow)
@pytest.mark.slow
@pytest.mark.parametrize("test_case", HARD_TESTS, ids=[t.name for t in HARD_TESTS])
@pytest.mark.asyncio
async def test_hard(test_case):
    """Run hard tests (slow, may require multiple models)."""
    result = await run_test_case(test_case)
    print(f"\n{test_case.name}: {result}")
    # Hard tests may fail - we just want to see how they perform
    if not result["success"]:
        pytest.xfail(f"Hard test did not complete: {result.get('error')}")


# =============================================================================
# STANDALONE RUNNER
# =============================================================================

async def run_all_tests(complexity_filter: Optional[str] = None):
    """Run all tests and print summary."""
    tests = ALL_TESTS
    if complexity_filter:
        tests = [t for t in tests if t.complexity == complexity_filter]

    print(f"\n{'='*60}")
    print(f"Running {len(tests)} tests")
    print(f"{'='*60}\n")

    results = []
    for test in tests:
        print(f"Running: {test.name} ({test.complexity})...")
        result = await run_test_case(test)
        results.append(result)

        status = "✓" if result["success"] else "✗"
        print(f"  {status} {result['elapsed_seconds']}s, {result['llm_calls']} LLM calls")
        if result["result"]:
            print(f"    Result: {result['result'][:100]}...")
        if result["error"]:
            print(f"    Error: {result['error']}")
        print()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    by_complexity = {}
    for r in results:
        c = r["complexity"]
        if c not in by_complexity:
            by_complexity[c] = {"passed": 0, "failed": 0, "total_time": 0, "total_llm": 0}
        by_complexity[c]["total_time"] += r["elapsed_seconds"]
        by_complexity[c]["total_llm"] += r["llm_calls"]
        if r["success"]:
            by_complexity[c]["passed"] += 1
        else:
            by_complexity[c]["failed"] += 1

    for complexity in ["simple", "medium", "complex", "hard"]:
        if complexity in by_complexity:
            stats = by_complexity[complexity]
            total = stats["passed"] + stats["failed"]
            print(f"\n{complexity.upper()}:")
            print(f"  Pass rate: {stats['passed']}/{total} ({100*stats['passed']/total:.0f}%)")
            print(f"  Total time: {stats['total_time']:.1f}s")
            print(f"  Total LLM calls: {stats['total_llm']}")

    passed = sum(1 for r in results if r["success"])
    print(f"\nOVERALL: {passed}/{len(results)} passed")

    return results


if __name__ == "__main__":
    import sys

    complexity = sys.argv[1] if len(sys.argv) > 1 else None
    if complexity and complexity not in ["simple", "medium", "complex", "hard"]:
        print(f"Usage: python {sys.argv[0]} [simple|medium|complex|hard]")
        sys.exit(1)

    asyncio.run(run_all_tests(complexity))
