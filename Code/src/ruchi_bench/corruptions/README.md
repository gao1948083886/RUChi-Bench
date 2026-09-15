# Seven corruption strategies

The canonical registry is `ruchi_bench.corruptions.registry.CORRUPTOR_REGISTRY`.
Every corruptor accepts a clean `Sample`, a non-negative seed, and one of
`low`/`medium`/`high`; it returns a `CorruptionResult` with code-point edit traces.
Labels, questions, options, and non-target fields are never changed.

| ID | Strategy | Implementation |
|---|---|---|
| `homo` | tone-agnostic homophone replacement | JioNLP `homophone_substitution` |
| `vis` | visual-similarity character replacement | Confused_Chinese ranked resource snapshot |
| `del` | character deletion | JioNLP `random_add_delete` with insertion disabled |
| `swap` | neighboring character transposition | JioNLP `swap_char_position` |
| `add_noise` | empirical character/symbol insertion | JioNLP `random_add_delete` with deletion disabled |
| `red_char` | Chinese character repetition | transparent deterministic implementation |
| `red_word` | Chinese word repetition | deterministic `jieba` segmentation + repetition |

## Pilot execution rule

Use `ruchi_bench.data.strategy_assignment.strategy_assignment` for the planned
experiment. It preflights eligible strategies, assigns exactly one strategy per
sample within each dataset, and writes three separate levels under:

```text
corrupted/<dataset>/<strategy>/{low,medium,high}.jsonl
```

Each level receives the original clean sample as input. The runner never feeds a
previously corrupted level into another level, and it never combines the five
datasets into one file. `run_corruptions.run_all_corruptions` remains available as an
explicit exhaustive diagnostic runner, but it is not the pilot protocol.

## Open-source notices

- JioNLP: Apache-2.0, <https://github.com/dongrixinyu/JioNLP>
- Confused_Chinese: Apache-2.0, <https://github.com/Macielyoung/Confused_Chinese>

