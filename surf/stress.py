"""
Stress test suite for rhea-surf.

Test categories:
1. complexity - Hard sites (SPAs, dynamic content)
2. speed - Same task repeated (cache warmup)
3. adversarial - Edge cases and error handling
4. vision - Vision fallback triggers
5. multistep - Complex multi-step tasks

Usage:
    python -m surf.stress --suite complexity
    python -m surf.stress --suite speed --rounds 20
    python -m surf.stress --suite all
"""

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from surf.agent import SurfAgent

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class StressResult:
    """Result of a single stress test."""
    name: str
    suite: str
    success: bool
    time_seconds: float
    result: Optional[str]
    error: Optional[str]


# =============================================================================
# TEST SUITES
# =============================================================================

COMPLEXITY_TESTS = [
    {
        "name": "static_simple",
        "task": "what is the main heading?",
        "url": "https://example.com",
        "timeout": 30,
    },
    {
        "name": "news_dynamic",
        "task": "what is the title of the top story?",
        "url": "https://news.ycombinator.com",
        "timeout": 45,
    },
    {
        "name": "wikipedia_dense",
        "task": "what is the first sentence of this article?",
        "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "timeout": 60,
    },
    {
        "name": "github_spa",
        "task": "what is the repository description?",
        "url": "https://github.com/anthropics/anthropic-sdk-python",
        "timeout": 60,
    },
    {
        "name": "reddit_infinite_scroll",
        "task": "what is the title of the first post?",
        "url": "https://old.reddit.com/r/programming",
        "timeout": 45,
    },
]

SPEED_TESTS = [
    {
        "name": "cache_warmup",
        "task": "what is the main heading?",
        "url": "https://example.com",
        "timeout": 20,
    },
]

ADVERSARIAL_TESTS = [
    {
        "name": "minimal_page",
        "task": "what text is on this page?",
        "url": "about:blank",
        "timeout": 15,
        "expect_fail": True,
    },
    {
        "name": "non_english",
        "task": "what is the main headline?",
        "url": "https://www.lemonde.fr",
        "timeout": 45,
    },
    {
        "name": "ambiguous_task",
        "task": "click the thing",
        "url": "https://example.com",
        "timeout": 30,
    },
    {
        "name": "impossible_task",
        "task": "find the shopping cart",
        "url": "https://example.com",
        "timeout": 30,
        "expect_fail": True,
    },
]

VISION_TESTS = [
    {
        "name": "vision_simple",
        "task": "what text do you see on this page?",
        "url": "https://example.com",
        "timeout": 60,
        "force_vision": True,
    },
]

MULTISTEP_TESTS = [
    {
        "name": "hn_second_story",
        "task": "what is the title of the second story?",
        "url": "https://news.ycombinator.com",
        "timeout": 60,
    },
    {
        "name": "wiki_toc_item",
        "task": "what is the third item in the table of contents?",
        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "timeout": 60,
    },
    {
        "name": "github_star_count",
        "task": "how many stars does this repository have?",
        "url": "https://github.com/anthropics/anthropic-sdk-python",
        "timeout": 60,
    },
]


async def run_single_test(test: dict, agent: SurfAgent) -> StressResult:
    """Run a single stress test."""
    name = test["name"]
    task = test["task"]
    url = test["url"]
    timeout = test.get("timeout", 30)
    expect_fail = test.get("expect_fail", False)

    start = time.time()
    try:
        result = await asyncio.wait_for(
            agent.run(task, url),
            timeout=timeout
        )
        elapsed = time.time() - start

        success = result.success
        if expect_fail:
            success = not success  # Invert for expected failures

        return StressResult(
            name=name,
            suite=test.get("suite", "unknown"),
            success=success,
            time_seconds=elapsed,
            result=result.summary if result.success else None,
            error=None if result.success else "Task failed"
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        return StressResult(
            name=name,
            suite=test.get("suite", "unknown"),
            success=expect_fail,  # Timeout is success if we expected fail
            time_seconds=elapsed,
            result=None,
            error=f"Timeout after {timeout}s"
        )
    except Exception as e:
        elapsed = time.time() - start
        return StressResult(
            name=name,
            suite=test.get("suite", "unknown"),
            success=expect_fail,
            time_seconds=elapsed,
            result=None,
            error=str(e)
        )


async def run_suite(suite_name: str, rounds: int = 1) -> List[StressResult]:
    """Run a test suite."""

    suites = {
        "complexity": COMPLEXITY_TESTS,
        "speed": SPEED_TESTS,
        "adversarial": ADVERSARIAL_TESTS,
        "vision": VISION_TESTS,
        "multistep": MULTISTEP_TESTS,
    }

    if suite_name == "all":
        tests = []
        for name, suite_tests in suites.items():
            for t in suite_tests:
                t["suite"] = name
                tests.append(t)
    elif suite_name in suites:
        tests = suites[suite_name]
        for t in tests:
            t["suite"] = suite_name
    else:
        print(f"Unknown suite: {suite_name}")
        print(f"Available: {', '.join(suites.keys())}, all")
        return []

    # For speed tests, repeat the same test
    if suite_name == "speed":
        tests = tests * rounds

    results = []
    agent = SurfAgent(headless=True)

    print(f"\n{'='*70}")
    print(f"STRESS TEST: {suite_name.upper()}")
    print(f"{'='*70}")
    print(f"Tests: {len(tests)}")
    print()

    for i, test in enumerate(tests):
        print(f"[{i+1}/{len(tests)}] {test['name']}: {test['task'][:40]}...")
        result = await run_single_test(test, agent)
        results.append(result)

        status = "✅ PASS" if result.success else "❌ FAIL"
        print(f"       {status} ({result.time_seconds:.1f}s)")
        if result.result:
            print(f"       Result: {result.result[:50]}...")
        if result.error:
            print(f"       Error: {result.error}")
        print()

    # Summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    passed = sum(1 for r in results if r.success)
    total = len(results)
    avg_time = sum(r.time_seconds for r in results) / total if total > 0 else 0

    print(f"Passed: {passed}/{total} ({100*passed/total:.0f}%)")
    print(f"Avg time: {avg_time:.1f}s")

    # Per-suite breakdown if running all
    if suite_name == "all":
        print("\nBy suite:")
        for sname in suites.keys():
            suite_results = [r for r in results if r.suite == sname]
            if suite_results:
                suite_passed = sum(1 for r in suite_results if r.success)
                print(f"  {sname}: {suite_passed}/{len(suite_results)}")

    # Speed test specific: show cache warmup
    if suite_name == "speed" and len(results) > 1:
        print("\nCache warmup analysis:")
        times = [r.time_seconds for r in results]
        print(f"  First run:  {times[0]:.1f}s")
        print(f"  Last run:   {times[-1]:.1f}s")
        print(f"  Speedup:    {times[0] - times[-1]:.1f}s ({100*(times[0]-times[-1])/times[0]:.0f}%)")

    print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Stress test rhea-surf")
    parser.add_argument("--suite", type=str, default="complexity",
                        help="Test suite: complexity, speed, adversarial, vision, multistep, all")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Rounds for speed test (default: 10)")
    args = parser.parse_args()

    asyncio.run(run_suite(args.suite, args.rounds))


if __name__ == "__main__":
    main()
