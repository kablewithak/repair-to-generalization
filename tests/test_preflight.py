"""Offline contract tests for the SubQ model-access preflight."""

import io
import json
import unittest
from urllib.error import HTTPError, URLError

from repair_to_generalization.preflight import (
    AccessStatus,
    MODELS_URL,
    TARGET_MODEL,
    check_model_access,
)


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def opener_for(payload):
    def open_url(request, *, timeout):
        assert request.full_url == MODELS_URL
        assert request.get_method() == "GET"
        assert timeout > 0
        return Response(json.dumps(payload).encode("utf-8"))

    return open_url


class ModelAccessTests(unittest.TestCase):
    def test_missing_key_does_not_touch_network(self):
        def forbidden(*args, **kwargs):
            self.fail("no network request allowed without a key")

        result = check_model_access(None, open_url=forbidden)
        self.assertEqual(result.status, AccessStatus.MISSING_KEY)

    def test_matching_model_is_available(self):
        result = check_model_access("dummy", open_url=opener_for({"data": [{"id": TARGET_MODEL}]}))
        self.assertEqual(result.status, AccessStatus.AVAILABLE)
        self.assertEqual(result.models_seen, 1)

    def test_missing_model_is_not_silently_assumed(self):
        result = check_model_access("dummy", open_url=opener_for({"data": [{"id": "subq-snap-1.2"}]}))
        self.assertEqual(result.status, AccessStatus.NOT_LISTED)

    def test_malformed_response_is_rejected(self):
        result = check_model_access("dummy", open_url=opener_for({"data": [{"name": "missing id"}]}))
        self.assertEqual(result.status, AccessStatus.INVALID_RESPONSE)

    def test_authentication_failure_is_explicit(self):
        def unauthorized(request, *, timeout):
            raise HTTPError(request.full_url, 401, "Unauthorized", hdrs=None, fp=None)

        result = check_model_access("dummy", open_url=unauthorized)
        self.assertEqual(result.status, AccessStatus.AUTH_FAILED)
        self.assertEqual(result.http_status, 401)

    def test_network_failure_is_explicit(self):
        def offline(request, *, timeout):
            raise URLError("offline")

        result = check_model_access("dummy", open_url=offline)
        self.assertEqual(result.status, AccessStatus.NETWORK_ERROR)

    def test_api_key_is_only_transmitted_in_auth_header(self):
        def inspect(request, *, timeout):
            self.assertEqual(request.get_header("Authorization"), "Bearer secret-test-value")
            self.assertNotIn("secret-test-value", request.full_url)
            return Response(b'{"data": []}')

        result = check_model_access("secret-test-value", open_url=inspect)
        self.assertEqual(result.status, AccessStatus.NOT_LISTED)


if __name__ == "__main__":
    unittest.main()
