| configuration | 1 thread | 2 threads | 4 threads | speed-up over 1 thread (2 / 4 threads) |
|:--|--:|--:|--:|--:|
| Q8\_0, batch 1 | 0.32 | 0.55 | 0.97 | 1.74x / 3.06x |
| Q8\_0, batch 8 | 0.63 | 1.14 | 1.85 | 1.81x / 2.93x |
| CTranslate2 int8, batch 8 | 1.46 | 1.71 | 1.95 | 1.17x / 1.34x |
| Q8\_0, 4 processes x 1 thread, batch 8 each (aggregate) |  |  | 2.08 | range 1.88-2.22 |
| CTranslate2 int8, 4 processes x 1 thread, batch 8 each (aggregate) |  |  | 3.54 | range 3.42-3.71 |
