"""Raw store: every byte downloaded from an external source is kept, so nothing is fetched twice.

Backends
  local   files under data/raw/store (default)
  s3      any S3-compatible store, e.g. MinIO in a personal datalake

Configuration (environment)
  MLM_RAWSTORE         "local" (default) or "s3://bucket/optional/prefix"
  MLM_S3_ENDPOINT      e.g. https://minio.example.com (omit for AWS S3)
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION   standard S3 credentials

Key layout (same in every backend)
  sec/edgar/filings/{cik}/{accession}/{file}.gz       raw filing HTML, gzip
  sec/edgar/index/{cik}/{accession}.html.gz           filing index (subject vs filer)
  sec/submissions/{cik}.json.gz                       company records
  companies_house/{endpoint}/{id}.json.gz             API responses
  prices/{source}/{ticker}.parquet                    daily bars
"""
import gzip
import os
from pathlib import Path
from typing import Protocol

from mlm import config


class RawStore(Protocol):
    def exists(self, key: str) -> bool: ...
    def get(self, key: str) -> bytes | None: ...
    def put(self, key: str, data: bytes) -> None: ...
    def list(self, prefix: str) -> list[str]: ...


class LocalStore:
    def __init__(self, root: Path = config.RAW / "store"):
        self.root = Path(root)

    def _p(self, key: str) -> Path:
        return self.root / key

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def get(self, key: str) -> bytes | None:
        p = self._p(key)
        return p.read_bytes() if p.exists() else None

    def put(self, key: str, data: bytes) -> None:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    def list(self, prefix: str) -> list[str]:
        base = self._p(prefix)
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file())


class S3Store:
    def __init__(self, url: str, endpoint: str | None = None):
        import boto3  # optional dependency: pip install boto3
        bucket, _, prefix = url.removeprefix("s3://").partition("/")
        self.bucket, self.prefix = bucket, prefix.strip("/")
        self.s3 = boto3.client("s3", endpoint_url=endpoint or os.environ.get("MLM_S3_ENDPOINT"))

    def _k(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    # Only a genuine "not found" means missing. A 403, an expired key or a network error must raise:
    # treating it as missing would re-download, re-put and add a new object version every time.
    _MISSING = {"404", "NoSuchKey", "NotFound"}

    def _is_missing(self, err) -> bool:
        return str(err.response.get("Error", {}).get("Code")) in self._MISSING

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except ClientError as err:
            if self._is_missing(err):
                return False
            raise

    def get(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self._k(key))["Body"].read()
        except ClientError as err:
            if self._is_missing(err):
                return None
            raise

    def put(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=data)

    def list(self, prefix: str) -> list[str]:
        out, token = [], None
        while True:
            kw = dict(Bucket=self.bucket, Prefix=self._k(prefix))
            if token:
                kw["ContinuationToken"] = token
            r = self.s3.list_objects_v2(**kw)
            out += [o["Key"].removeprefix(f"{self.prefix}/") for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return out
            token = r["NextContinuationToken"]


_STORE: RawStore | None = None


def get_store() -> RawStore:
    global _STORE
    if _STORE is None:
        url = os.environ.get("MLM_RAWSTORE", "local")
        _STORE = S3Store(url) if url.startswith("s3://") else LocalStore()
    return _STORE


def put_text(key: str, text: str, store: RawStore | None = None) -> None:
    (store or get_store()).put(key, gzip.compress(text.encode("utf-8")))


def get_text(key: str, store: RawStore | None = None) -> str | None:
    data = (store or get_store()).get(key)
    return gzip.decompress(data).decode("utf-8") if data is not None else None
