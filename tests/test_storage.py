from mlm.storage import LocalStore, get_text, put_text


def test_local_store_roundtrip(tmp_path):
    store = LocalStore(tmp_path)
    assert not store.exists("sec/edgar/filings/1/a.htm.gz")
    put_text("sec/edgar/filings/1/a.htm.gz", "<html>pledged</html>", store)
    assert store.exists("sec/edgar/filings/1/a.htm.gz")
    assert get_text("sec/edgar/filings/1/a.htm.gz", store) == "<html>pledged</html>"
    assert store.list("sec/edgar") == ["sec/edgar/filings/1/a.htm.gz"]
    assert get_text("missing", store) is None


def _stubbed_s3store():
    import boto3
    from botocore.stub import Stubber
    from mlm.storage import S3Store
    store = S3Store.__new__(S3Store)
    store.bucket, store.prefix = "mlm-raw", ""
    store.s3 = boto3.client("s3", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x",
                            endpoint_url="https://example.invalid")
    return store, Stubber(store.s3)


def test_s3_missing_object_is_none_but_403_raises():
    import pytest
    from botocore.exceptions import ClientError
    store, stub = _stubbed_s3store()
    stub.add_client_error("head_object", service_error_code="404", http_status_code=404)
    stub.add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404)
    stub.add_client_error("get_object", service_error_code="AccessDenied", http_status_code=403)
    stub.add_client_error("head_object", service_error_code="403", http_status_code=403)
    with stub:
        assert store.exists("sec/a.gz") is False
        assert store.get("sec/a.gz") is None
        with pytest.raises(ClientError):
            store.get("sec/a.gz")          # expired or wrong key: must not look like "missing"
        with pytest.raises(ClientError):
            store.exists("sec/a.gz")
