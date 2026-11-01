| key | phase 2c | final2 |
|:--|:--|:--|
| `steps` | 1433 | 700 |
| `prompts_per_step` | 84 | 84 |
| `group_size` | 16 | 16 |
| `greedy_in_group` | False | True |
| `greedy_stats` | -- | sampled |
| `greedy_adv_scale` | -- | 0.3 |
| `temperature` | 1.0 | 1.0 |
| `top_p` | 1.0 | 1.0 |
| `top_k` | 0 | 0 |
| `learning_rate` | 2e-06 | 1e-06 |
| `warmup_steps` | 10 | 20 |
| `weight_decay` | 0.0 | 0.0 |
| `beta1` | 0.9 | 0.9 |
| `beta2` | 0.95 | 0.95 |
| `grad_clip` | 1.0 | 1.0 |
| `advantage_norm` | mean_std | mean_std |
| `std_eps` | 0.0001 | 0.0001 |
| `kl_coef` | 0.05 | 0.1 |
| `kl_estimator` | k3c | k3c |
| `kl_guard_threshold` | 0.15 | 0.15 |
| `kl_guard_window` | 5 | 5 |
| `kl_guard_factor` | 0.5 | 0.5 |
| `kl_guard_min_scale` | 0.125 | 0.125 |
| `num_updates_per_batch` | 1 | 1 |
| `loss_agg` | token | token |
| `max_new_tokens` | 1024 | 1024 |
| `max_src_tokens` | 2048 | 2048 |
| `language_alpha` | 0.0 | 0.0 |
| `language_floor_steps` | -- | 20 |
| `language_floor_prompts` | -- | 2 |
| `select_by` | benchmark_chrf | benchmark_chrf |
| `select_min_gain` | 0.1 | 0.2 |
| `select_min_finished_rate` | 0.95 | 0.95 |
| `seed` | 1234 | 1234 |
| `curriculum` | b256 391, b512 446, b1024 149, b2048 298, mixed 149 steps | mixed from step 0 (70% science / 30% general) |
| `max_new_tokens_by_bin` | {"b64": 384, "b256": 1280, "b512": 2304, "b1024": 3584, "b2048": 7168, "mixed": 7168} | {"b64": 384, "b256": 1280, "b512": 2304, "b1024": 3584, "b2048": 7168, "mixed": 7168} |
| `prompts_per_step_by_bin` | {"b64": 64, "b256": 48, "b512": 42, "b1024": 42, "b2048": 21, "mixed": 42} | {"b64": 64, "b256": 48, "b512": 42, "b1024": 42, "b2048": 21, "mixed": 42} |
| `start model` | 2b step-488 (b64-stage) model | science SFT stage A2W, alpha 0.7 |
