"""
Learning loop - prove rhea-surf gets smarter over time.

Runs multiple study sessions and tracks improvement in:
1. Cache hit rate (should increase as patterns are learned)
2. Task completion time (should decrease)
3. Total loss (should decrease)

Usage:
    python -m surf.learning_loop --rounds 5
"""

import argparse
import asyncio
import time
from datetime import datetime

from surf.study import StudyRunner, StudyDB, LossMetrics


async def run_learning_loop(rounds: int = 5, pause_between: int = 2):
    """
    Run multiple study sessions and track improvement.

    Args:
        rounds: Number of study sessions to run
        pause_between: Seconds to pause between rounds
    """
    print("=" * 70)
    print("LEARNING LOOP - Proving rhea-surf gets smarter")
    print("=" * 70)
    print(f"\nRunning {rounds} study sessions...\n")

    db = StudyDB()
    runner = StudyRunner()

    results = []

    for i in range(rounds):
        print(f"\n{'='*70}")
        print(f"ROUND {i+1}/{rounds}")
        print(f"{'='*70}")

        start = time.time()
        session = await runner.run_session()
        elapsed = time.time() - start

        loss = LossMetrics.from_session(session)

        results.append({
            "round": i + 1,
            "passed": session.tasks_passed,
            "total": session.tasks_run,
            "cache_pct": session.cache_utilization * 100,
            "llm_calls": session.total_llm_calls,
            "time": elapsed,
            "loss": loss.total_loss,
        })

        print(f"\nRound {i+1} complete:")
        print(f"  Passed: {session.tasks_passed}/{session.tasks_run}")
        print(f"  Cache: {session.cache_utilization*100:.0f}%")
        print(f"  Loss: {loss.total_loss:.3f}")

        if i < rounds - 1:
            print(f"\nPausing {pause_between}s before next round...")
            await asyncio.sleep(pause_between)

    # Summary
    print("\n" + "=" * 70)
    print("LEARNING SUMMARY")
    print("=" * 70)

    print("\nRound  Pass  Cache%  LLM   Time    Loss")
    print("-" * 50)
    for r in results:
        print(f"  {r['round']:2d}    {r['passed']}/{r['total']}   {r['cache_pct']:5.1f}%   {r['llm_calls']:2d}   {r['time']:5.1f}s  {r['loss']:.3f}")

    # Calculate improvement
    first = results[0]
    last = results[-1]

    cache_improvement = last["cache_pct"] - first["cache_pct"]
    loss_improvement = first["loss"] - last["loss"]
    time_improvement = first["time"] - last["time"]

    print("\n" + "-" * 50)
    print("IMPROVEMENT (first → last):")
    print(f"  Cache hit rate: {first['cache_pct']:.1f}% → {last['cache_pct']:.1f}% ({cache_improvement:+.1f}%)")
    print(f"  Total loss:     {first['loss']:.3f} → {last['loss']:.3f} ({loss_improvement:+.3f})")
    print(f"  Time:           {first['time']:.1f}s → {last['time']:.1f}s ({time_improvement:+.1f}s)")

    # Verdict
    print("\n" + "=" * 70)
    if cache_improvement > 5 or loss_improvement > 0.02:
        print("✅ LEARNING DETECTED - System is getting smarter!")
    elif cache_improvement >= 0 and loss_improvement >= 0:
        print("→ STABLE - No regression, learning may need more rounds")
    else:
        print("⚠️  REGRESSION - Performance decreased, investigate")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run learning loop")
    parser.add_argument("--rounds", type=int, default=5, help="Number of study rounds")
    parser.add_argument("--pause", type=int, default=2, help="Seconds between rounds")
    args = parser.parse_args()

    asyncio.run(run_learning_loop(rounds=args.rounds, pause_between=args.pause))


if __name__ == "__main__":
    main()
