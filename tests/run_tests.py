#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Minimal test runner for machines WITHOUT pytest (this laptop). With pytest installed just run `pytest tests`.

    python tests/run_tests.py [-k substring] [-v] [path ...]

It understands the small pytest subset the tests use: pytest.skip / importorskip / raises / approx,
@pytest.mark.parametrize, @pytest.mark.skipif, and the `tmp_path` fixture. Exit code 1 on any failure.
"""

import importlib.util
import inspect
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
for _k, _v in (("HF_ORG", "example-org"), ("HF_ORG_LEGACY", "example-legacy-org"), ("S3_BUCKET", "example-bucket"), ("S3_ENDPOINT_URL", "https://s3.example.invalid"), ("S3_PREFIX", "run-prefix")):
    os.environ.setdefault(_k, _v)


class Skipped(Exception):
    pass


def _install_shim():
    try:
        import pytest  # noqa: F401
        return False
    except ImportError:
        pass
    import importlib.machinery

    m = types.ModuleType("pytest")
    m.__spec__ = importlib.machinery.ModuleSpec("pytest", None)   # libraries probe `find_spec("pytest")`
    m.__shim__ = True

    def skip(reason=""):
        raise Skipped(reason)

    def importorskip(name, reason=None):
        try:
            return importlib.import_module(name)
        except ImportError:
            raise Skipped(reason or f"could not import {name!r}")

    class _Raises:
        def __init__(self, exc, match=None):
            self.exc, self.match = exc, match

        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(f"DID NOT RAISE {self.exc}")
            if not issubclass(et, self.exc):
                return False
            if self.match is not None:
                import re
                assert re.search(self.match, str(ev)), f"{ev!r} does not match {self.match!r}"
            return True

    class approx:
        def __init__(self, v, rel=1e-6, abs=1e-12):
            self.v, self.rel, self.abs = v, rel, abs

        def __eq__(self, other):
            if isinstance(self.v, (list, tuple)):
                return len(other) == len(self.v) and all(
                    math.isclose(o, v, rel_tol=self.rel, abs_tol=self.abs) for o, v in zip(other, self.v))
            return math.isclose(other, self.v, rel_tol=self.rel, abs_tol=self.abs)

        def __repr__(self):
            return f"approx({self.v})"

    class _Mark:
        def parametrize(self, names, values):
            def deco(fn):
                fn._params = (names, list(values))
                return fn
            return deco

        def skipif(self, cond, reason=""):
            def deco(fn):
                if cond:
                    fn._skip = reason
                return fn
            return deco

        def __getattr__(self, name):
            return lambda *a, **k: (lambda fn: fn)

    m.skip, m.importorskip, m.raises, m.approx, m.mark = skip, importorskip, _Raises, approx, _Mark()
    m.fixture = lambda *a, **k: (lambda fn: fn) if not (a and callable(a[0])) else a[0]
    m.Skipped = Skipped
    sys.modules["pytest"] = m
    return True


def _collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.startswith("test_") and f.endswith(".py"))
        else:
            files.append(p)
    return files


def _call(fn, kwargs):
    sig = inspect.signature(fn)
    tmp = None
    if "tmp_path" in sig.parameters:
        import pathlib
        tmp = tempfile.mkdtemp(prefix="gemma_mt_test_")
        kwargs = dict(kwargs, tmp_path=pathlib.Path(tmp))
    try:
        fn(**kwargs)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def main():
    args = sys.argv[1:]
    verbose = "-v" in args
    sub = None
    if "-k" in args:
        sub = args[args.index("-k") + 1]
    paths = [a for a in args if not a.startswith("-") and a != sub] or [HERE]
    _install_shim()
    sys.argv = sys.argv[:1]
    passed = failed = skipped = 0
    t0 = time.time()
    for f in _collect(paths):
        name = os.path.splitext(os.path.basename(f))[0]
        spec = importlib.util.spec_from_file_location(name, f)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Skipped as e:
            print(f"SKIP  {name} (module): {e}")
            skipped += 1
            continue
        except Exception:
            print(f"ERROR {name}: import failed")
            traceback.print_exc()
            failed += 1
            continue
        for tname, fn in sorted(vars(mod).items()):
            if not tname.startswith("test_") or not callable(fn):
                continue
            if sub and sub not in tname and sub not in name:
                continue
            cases = [{}]
            if hasattr(fn, "_params"):
                names, vals = fn._params
                names = [n.strip() for n in names.split(",")]
                cases = [dict(zip(names, v if len(names) > 1 else (v,))) for v in vals]
            for ci, kw in enumerate(cases):
                label = f"{name}::{tname}" + (f"[{ci}]" if len(cases) > 1 else "")
                if getattr(fn, "_skip", None):
                    print(f"SKIP  {label}: {fn._skip}")
                    skipped += 1
                    continue
                t1 = time.time()
                try:
                    _call(fn, kw)
                    passed += 1
                    if verbose:
                        print(f"PASS  {label} ({time.time() - t1:.1f}s)")
                except Skipped as e:
                    skipped += 1
                    print(f"SKIP  {label}: {e}")
                except Exception:
                    failed += 1
                    print(f"FAIL  {label}")
                    traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped in {time.time() - t0:.1f}s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
