"""
Test harness for surf agent.

This runs autonomously - Claude Code kicks it off, local models do the work.
Run with: python -m pytest tests/test_agent.py -v -s
"""

import asyncio
import pytest
from surf.agent import SurfAgent, AgentResult


# Test scenarios for iteration
TEST_SCENARIOS = [
    {
        "name": "google_search",
        "task": "Go to Google and search for 'weather in Fort Worth'",
        "start_url": "https://www.google.com",
        "success_criteria": lambda r: "google.com/search" in r.final_url,
    },
    {
        "name": "wikipedia_navigate",
        "task": "Go to Wikipedia and find the article about Python programming language",
        "start_url": "https://www.wikipedia.org",
        "success_criteria": lambda r: "Python" in r.final_url,
    },
    {
        "name": "simple_navigation",
        "task": "Navigate to Hacker News",
        "start_url": None,
        "success_criteria": lambda r: "news.ycombinator.com" in r.final_url,
    },
]


class TestSurfAgent:
    """Test suite for surf agent."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return SurfAgent(
            model="llama3:latest",
            headless=True,  # Run headless in tests
            max_steps=15,
            verbose=True,
        )

    @pytest.mark.asyncio
    async def test_google_search(self, agent):
        """Test: Search on Google."""
        result = await agent.run(
            task="Go to Google and search for 'weather in Fort Worth'",
            start_url="https://www.google.com",
        )

        print(f"\nResult: success={result.success}, steps={len(result.steps)}")
        print(f"Final URL: {result.final_url}")

        # Assert we got to a search results page
        assert "google.com" in result.final_url
        assert len(result.steps) > 0

    @pytest.mark.asyncio
    async def test_simple_navigate(self, agent):
        """Test: Simple navigation task."""
        result = await agent.run(
            task="Navigate to https://news.ycombinator.com",
            start_url=None,
        )

        print(f"\nResult: success={result.success}, steps={len(result.steps)}")
        print(f"Final URL: {result.final_url}")

        assert "ycombinator" in result.final_url.lower()


async def run_scenario(scenario: dict, model: str = "deepseek-r1:8b") -> dict:
    """Run a single test scenario and return results."""
    agent = SurfAgent(
        model=model,
        headless=True,
        max_steps=15,
        verbose=True,
    )

    result = await agent.run(
        task=scenario["task"],
        start_url=scenario.get("start_url"),
    )

    passed = scenario["success_criteria"](result)

    return {
        "name": scenario["name"],
        "passed": passed,
        "steps": len(result.steps),
        "final_url": result.final_url,
        "summary": result.summary,
    }


async def run_all_scenarios(model: str = "deepseek-r1:8b"):
    """
    Run all test scenarios.

    Use this for batch testing different models or prompts.
    """
    print(f"\n{'='*60}")
    print(f"Running {len(TEST_SCENARIOS)} scenarios with model: {model}")
    print(f"{'='*60}\n")

    results = []
    for scenario in TEST_SCENARIOS:
        print(f"\n--- Running: {scenario['name']} ---")
        result = await run_scenario(scenario, model)
        results.append(result)
        print(f"Result: {'PASS' if result['passed'] else 'FAIL'}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for r in results if r["passed"])
    print(f"Passed: {passed}/{len(results)}")
    for r in results:
        status = "✓" if r["passed"] else "✗"
        print(f"  {status} {r['name']}: {r['steps']} steps -> {r['final_url'][:50]}")

    return results


if __name__ == "__main__":
    # Run all scenarios
    asyncio.run(run_all_scenarios())
