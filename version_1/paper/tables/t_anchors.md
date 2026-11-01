| detector | detection of corrupted human references | false positives on held-out human text | source |
|:--|:--|:--|:--|
| entity (numbers), 3 weak languages | 100% (other source's translation); 59% (second half dropped) | 0.7% mean, 2.0% max | anchors\_2026-10-21.md l.55-67, l.45-53 |
| repetition | 100% (clause x6, whole translation twice) | 1.8% mean, 6.7% max (sentences) | same |
| boilerplate buffer | 100% (stock sentence appended) | 0.4% mean, 3.0% max | same |
| substring copy (v2) | 100% exact / appended / half copies; 78% of a copied 25% tail; 96% of a 50% tail | 0.5% mean, 1.3% max (sentences), 0.0% (documents) | anchors\_v2\_2026-10-22.md sec. 3 |
| stop-word profile (v2) | AUC 0.99 vs English copy, 0.97 vs stuffed bags (sentences) | 1.2% sentences, 0.6% documents | anchors\_v2 sec. 4 |
| padding, v2 (length bug) | bag/repeat attacks caught | 5 / 14 / 31 / 81 / 100% of human documents of <100 / 100-200 / 200-400 / 400-800 / >800 words | anchors\_v3 sec. 1 |
| padding, v3 (segmental type/token) | -- | 0.0% median out-of-domain FP (max 76.4%, sid) | anchors\_v3 sec. 3 |
