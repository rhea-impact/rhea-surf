# Meta-Study Framework

Self-improvement tracking for rhea-surf using loss functions and trend analysis.

## Overview

The meta-study framework tracks rhea-surf's performance over time, measuring how well it completes browser automation tasks. By defining loss functions (lower = better), we can objectively measure improvement.

```
┌─────────────────────────────────────────────────────────────┐
│                    Study Session                             │
├─────────────────────────────────────────────────────────────┤
│  Run Tasks → Collect Metrics → Compute Loss → Store History  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Loss Metrics                              │
├─────────────────────────────────────────────────────────────┤
│  Task Failure (50%) │ Efficiency (25%) │ Speed (15%) │ Cache (10%) │
└─────────────────────────────────────────────────────────────┘
```

## Usage

```bash
# Run a study session (executes test tasks, records metrics)
python -m surf.study

# View history and trends
python -m surf.study --history

# Deep analysis with recommendations
python -m surf.study --analyze
```

## Loss Functions

### Task Failure Loss (50% weight)

Measures task completion rate.

```
task_failure_loss = 1.0 - success_rate
```

- 0.0 = all tasks passed
- 1.0 = all tasks failed

### Efficiency Loss (25% weight)

Measures LLM call overhead. Target: 2 calls per task.

```
efficiency_loss = clamp((avg_llm_calls - 2) / 10, 0, 1)
```

- 0.0 = 2 or fewer LLM calls per task
- 1.0 = 12+ LLM calls per task

### Speed Loss (15% weight)

Measures execution time. Target: 5 seconds per task.

```
speed_loss = clamp((avg_time - 5) / 30, 0, 1)
```

- 0.0 = 5 seconds or less
- 1.0 = 35+ seconds

### Cache Miss Loss (10% weight)

Measures cache utilization (deterministic replay).

```
cache_miss_loss = 1.0 - cache_utilization
```

- 0.0 = 100% cache hits
- 1.0 = 0% cache hits

### Total Loss

Weighted combination:

```
total_loss = 0.50 * task_failure
           + 0.25 * efficiency
           + 0.15 * speed
           + 0.10 * cache_miss
```

**Target: < 0.10** (excellent), **< 0.25** (good), **< 0.50** (needs work)

## Test Tasks

The study framework runs a set of benchmark tasks:

### Simple Tasks
| Task | Description | Success Criteria |
|------|-------------|------------------|
| `hn_top_story` | Read top story from Hacker News | Returns story title |
| `example_heading` | Read heading from example.com | Returns "Example Domain" |
| `github_repo` | Read repo name from GitHub | Returns repo identifier |

### Medium Tasks (multi-step)
| Task | Description | Success Criteria |
|------|-------------|------------------|
| `hn_second_story` | Find 2nd story on HN | Returns 2nd title |
| `hn_comments_count` | Count comments on top story | Returns number |
| `wiki_first_para` | Read first paragraph of Wiki page | Returns paragraph text |

## Trend Analysis

The framework tracks performance over time:

```
================================================================================
STUDY HISTORY (most recent first)
================================================================================
Timestamp            Pass     LLM      Cache%   Loss     Trend
--------------------------------------------------------------------------------
2026-02-02T11:33:37  3/3      1.7      17%      0.108
2026-02-02T11:29:38  3/3      1.7      17%      0.116     → same
2026-02-02T11:28:31  3/3      1.7      17%      0.112     → same

================================================================================
Overall: STABLE (+2.0%)
```

Trend indicators:
- `↑ improving` - Loss decreased >5%
- `→ same` - Loss within ±5%
- `↓ regressing` - Loss increased >5%

## Deep Analysis

The `--analyze` flag provides detailed insights:

```bash
python -m surf.study --analyze
```

Output includes:
- Session-by-session breakdown
- Loss component analysis
- Improvement recommendations
- Historical trends

## Data Storage

Sessions are stored in SQLite:

```
~/.rhea-surf/study.db

Tables:
- study_sessions: timestamp, task results, metrics, loss values
```

## Programmatic Usage

```python
from surf.study import StudyRunner, StudyDB, LossMetrics

# Run a study session
runner = StudyRunner()
session = await runner.run_session()

# Compute loss
loss = LossMetrics.from_session(session)
print(f"Total loss: {loss.total_loss:.3f}")

# Check history
db = StudyDB()
sessions = db.get_recent_sessions(limit=10)
trend = db.compute_trend()
```

## Improvement Strategies

Based on loss component analysis:

### High Task Failure Loss
- Review failed task patterns
- Improve DOM extraction for those sites
- Add site-specific patterns to learner

### High Efficiency Loss
- Increase cache utilization
- Add more patterns to learner
- Improve LLM prompts for faster decisions

### High Speed Loss
- Use faster models for simple tasks
- Increase cache hits
- Optimize DOM extraction

### High Cache Miss Loss
- Run same tasks multiple times
- Promote successful actions to patterns
- Review cache key generation

## Architecture

```python
# surf/study.py

@dataclass
class LossMetrics:
    task_failure_loss: float
    efficiency_loss: float
    speed_loss: float
    cache_miss_loss: float
    total_loss: float

    @classmethod
    def from_session(cls, session: StudySession) -> 'LossMetrics':
        # Compute all loss components
        ...

class StudyRunner:
    async def run_session(self) -> StudySession:
        # Execute tasks, collect metrics
        ...

class StudyDB:
    def store_session(self, session: StudySession): ...
    def get_recent_sessions(self, limit: int): ...
    def compute_trend(self) -> str: ...
```

## Integration with Other Components

The study framework integrates with:

- **Action Cache** (`surf/cache.py`) - Measures cache utilization
- **Navigator** (`surf/navigator.py`) - Executes tasks, counts LLM calls
- **Learner** (`surf/learner.py`) - Patterns improve over time
- **Trajectory Memory** (`surf/memory.py`) - Few-shot learning from past runs

## Best Practices

1. **Run daily** - Build trend data to spot regressions early
2. **After changes** - Run study session after modifying core components
3. **Add new tasks** - Expand benchmark suite as capabilities grow
4. **Review failures** - Use `--analyze` to understand what's not working
