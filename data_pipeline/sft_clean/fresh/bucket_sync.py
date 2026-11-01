#!/usr/bin/env python3
"""Move big directories between the boxes through the object-storage bucket (the direct vast-to-vast ssh route stalls at tens of KB/s).

    python bucket_sync.py put <prefix> <path> [<path> ...]      # keys: <prefix>/<basename of path>/<relative file>; files whose size already matches are skipped
    python bucket_sync.py get <prefix> <dest_dir> [<name> ...]   # downloads <prefix>/<name>/... into <dest_dir>/<name>/... (all names when none given), size-checked
    python bucket_sync.py has <prefix>/<key>                     # exit 0 when the key exists (marker files)
    python bucket_sync.py mark <prefix>/<key> [text]             # writes a small marker object

Credentials: environment AK / SK (env.sh). Works with the system python of the A100 (boto3 in /workspace/s3lib) and with venv_data.
"""
import os, sys, time
sys.path.insert(0, "/workspace/s3lib")
import boto3
from boto3.s3.transfer import TransferConfig

if "S3_ACCESS_KEY_ID" not in os.environ and os.path.exists(os.environ.get("ENV_FILE", ".env")):                      # the .env file of the repository (see .env.example); never printed
    for line in open(os.environ.get("ENV_FILE", ".env")):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1); os.environ[k] = v
B = os.environ['S3_BUCKET']
s3 = boto3.client("s3", endpoint_url=os.environ['S3_ENDPOINT_URL'], aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"], region_name=os.environ.get("S3_REGION", "us-east-1"))
cfg = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20, max_concurrency=8)


def size_of(key):
    try:
        return s3.head_object(Bucket=B, Key=key)["ContentLength"]
    except Exception:
        return None


def put(prefix, paths):
    for root in paths:
        root = root.rstrip("/"); name = os.path.basename(root)
        files = [root] if os.path.isfile(root) else [os.path.join(d, f) for d, _, fs in os.walk(root) for f in sorted(fs)]
        for p in files:
            rel = os.path.basename(p) if os.path.isfile(root) else os.path.relpath(p, root)
            key = f"{prefix}/{name}/{rel}"
            if "__pycache__" in key or key.endswith(".tmp"): continue
            sz = os.path.getsize(p)
            for t in range(5):
                try:
                    if size_of(key) != sz: s3.upload_file(p, B, key, Config=cfg)
                    break
                except Exception as e:
                    print("retry", key, type(e).__name__, flush=True); time.sleep(5 * (t + 1))
            else:
                sys.exit(f"FAILED {key}")
        print("put", name, len(files), "files", flush=True)


def get(prefix, dest, names):
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=prefix + "/"):
        for o in page.get("Contents", []):
            rel = o["Key"][len(prefix) + 1:]
            if names and rel.split("/", 1)[0] not in names: continue
            out = os.path.join(dest, rel)
            if os.path.exists(out) and os.path.getsize(out) == o["Size"]: continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            for t in range(5):
                try:
                    s3.download_file(B, o["Key"], out + ".part", Config=cfg); os.replace(out + ".part", out); break
                except Exception as e:
                    print("retry", o["Key"], type(e).__name__, flush=True); time.sleep(5 * (t + 1))
            else:
                sys.exit(f"FAILED {o['Key']}")
            if os.path.getsize(out) != o["Size"]: sys.exit(f"SIZE MISMATCH {out}")
            n += 1
    print("downloaded", n, "files", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "put": put(sys.argv[2], sys.argv[3:])
    elif cmd == "get": get(sys.argv[2], sys.argv[3], set(sys.argv[4:]))
    elif cmd == "has": sys.exit(0 if size_of(sys.argv[2]) is not None else 1)
    elif cmd == "mark": s3.put_object(Bucket=B, Key=sys.argv[2], Body=(sys.argv[3] if len(sys.argv) > 3 else time.strftime("%FT%TZ")).encode())
