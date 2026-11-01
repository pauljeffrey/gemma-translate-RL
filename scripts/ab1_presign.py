#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AB1 helper (run on the LAPTOP, where the bucket keys live): print a time-limited presigned GET URL for one object of the object-storage bucket, so the GPU box can stream it without ever
holding the S3 keys.   python scripts/ab1_presign.py gemma_mt/data/gemma_packed_fwd_sft_windows.tar.zst [seconds=7200]
Then on the box:  URL='<printed>' bash /workspace/jobs/ab1/build_weak_data.sh 3500   (streams the 2.27 GB archive once; writes ~2.3 GB; the archive's sha256 is checked on the fly).
Needs boto3 and S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY in ../.env (never printed)."""
import os
import sys

import boto3

env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
for line in open(env):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()
s3 = boto3.client("s3", endpoint_url=os.environ['S3_ENDPOINT_URL'], aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
                  aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"], region_name=os.environ.get("S3_REGION", "us-east-1"))
print(s3.generate_presigned_url("get_object", Params={"Bucket": os.environ['S3_BUCKET'], "Key": sys.argv[1]}, ExpiresIn=int(sys.argv[2]) if len(sys.argv) > 2 else 7200))
