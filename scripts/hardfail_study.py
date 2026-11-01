#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-61: what does masking the hard-failed samples do to the policy gradient? A two-outcome softmax policy (fail with probability p, reward -2; otherwise a translation with reward ~ N(0.5, 0.3)),
G=16 samples per prompt, the group-normalised advantage of rl/grpo.py. Reports the expected gradient on the FAILURE LOGIT (negative = the update lowers the failure probability), per rule:
standard (failures keep their negative advantage), zero (failed advantage 0, survivors as before), exclude (baseline/std over survivors only). Pure numpy, ~10 s."""
import numpy as np

rng = np.random.default_rng(0)
G, N = 16, 40000


def adv(r, ok, mode):
    if mode == "standard":
        m, s = r.mean(1, keepdims=True), r.std(1, keepdims=True)
        a = (r - m) / (s + 1e-4)
        return np.where(s < 1e-6, 0.0, a)
    if mode == "zero":
        a = adv(r, ok, "standard")
        return np.where(ok, a, 0.0)
    n = ok.sum(1, keepdims=True)
    m = (r * ok).sum(1, keepdims=True) / np.maximum(n, 1)
    s = np.sqrt((((r - m) ** 2) * ok).sum(1, keepdims=True) / np.maximum(n, 1))
    a = (r - m) / (s + 1e-4) * ok
    return np.where((s < 1e-6) | (n < 2), 0.0, a)


print("p_fail | rule      | E[grad on failure logit] (descent direction: negative lowers P(fail)) | mean |adv| of failed | of survivors")
for p in (0.03, 0.1, 0.25, 0.5):
    fail = rng.random((N, G)) < p
    r = np.where(fail, -2.0, rng.normal(0.5, 0.3, (N, G)))
    ok = ~fail
    for mode in ("standard", "zero", "exclude"):
        a = adv(r, ok, mode)
        dlogp = np.where(fail, 1 - p, -p)                      # d log pi(y) / d theta_fail
        g = (a * dlogp).sum(1).mean() / G                       # ascent direction on theta_fail
        print(f"{p:5.2f}  | {mode:9s} | {g:+.4f} | {np.abs(a[fail]).mean() if fail.any() else 0:.3f} | {np.abs(a[ok]).mean():.3f}")
