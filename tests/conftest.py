# SPDX-License-Identifier: AGPL-3.0-or-later
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# private identifiers are read from the environment (see .env.example); tests run with neutral stand-ins
for _k, _v in (("HF_ORG", "example-org"), ("HF_ORG_LEGACY", "example-legacy-org"), ("S3_BUCKET", "example-bucket"), ("S3_ENDPOINT_URL", "https://s3.example.invalid"), ("S3_PREFIX", "run-prefix")):
    os.environ.setdefault(_k, _v)
