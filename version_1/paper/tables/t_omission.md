| variant | false alarm | sentence recall | extra sentence appended / inserted found |
|:--|--:|--:|--:|
| v1 planner (legacy Gale-Church) | 0.3% | 0% | 42% / 40% |
| legacy cost, omission/extra priors raised to 0.3 | 0.6% | 0% | 42% / 40% |
| shortfall check only (legacy alignment) | 1.3% | 53% | 42% / 40% |
| language-aware DP, cost 3 | 2.6% | 61% | 94% / 92% |
| language-aware DP, cost 4 | 2.1% | 56% | 92% / 91% |
| DP cost 4 + shortfall 0.7 (chosen, per-language cost) | 2.2% (0.6% without ewe) | 56% | 93% / 92% |
