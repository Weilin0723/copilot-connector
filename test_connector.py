"""Offline protocol checks. Run: python -m unittest -v test_connector.py"""

import http.client
import io
import json
import threading
import time
import unittest
import urllib.error
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

from copilot_connector import AzureBackend, MAX_BODY, SCOPE, make_handler, upstream_url

try:
    from azure.core.credentials import AccessToken
    import azure.identity
except ImportError:
    AccessToken = None


class Response(io.BytesIO):
    status = 200
    headers = {"Content-Type": "text/event-stream"}


@contextmanager
def running(backend):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(backend, "deployment", "local-test-key"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def call(port, method="POST", path="/v1/chat/completions", body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    auth = {"Authorization": "Bearer local-test-key", "Content-Type": "application/json"}
    auth.update(headers or {})
    connection.request(method, path, body=body, headers=auth)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


class ConnectorTests(unittest.TestCase):
    @unittest.skipIf(AccessToken is None, "Install azure-identity to check SDK token refresh")
    def test_sdk_token_refresh_and_user_assigned_identity_selection(self):
        class Credential:
            calls = 0
            def get_token(self, *scopes, **kwargs):
                assert scopes == (SCOPE,)
                self.calls += 1
                expiry = 20 if self.calls == 1 else 3600
                return AccessToken("fake-token-" + str(self.calls), int(time.time()) + expiry)
        credential = Credential()
        args = SimpleNamespace(endpoint="https://example.openai.azure.com", deployment="deployment",
                               api_version="2024-10-21", auth="managed-identity",
                               client_id="11111111-2222-3333-4444-555555555555")
        with patch("azure.identity.ManagedIdentityCredential", return_value=credential) as factory:
            backend = AzureBackend(args)
            factory.assert_called_once_with(client_id=args.client_id)
        tokens = []
        def capture(request, timeout):
            tokens.append(request.get_header("Authorization"))
            return Response(b"{}")
        with patch.object(backend.opener, "open", side_effect=capture):
            for _ in range(3):
                backend(b"{}").close()
        self.assertEqual(tokens, ["Bearer fake-token-1", "Bearer fake-token-2", "Bearer fake-token-2"])
        self.assertEqual(credential.calls, 2)

    def test_endpoint_validation_and_deployment_encoding(self):
        url = upstream_url("https://example.openai.azure.com/", "a/b", "2024-10-21")
        self.assertEqual(url, "https://example.openai.azure.com/openai/deployments/a%2Fb/chat/completions?api-version=2024-10-21")
        for endpoint in ("http://example.com", "https://user:pass@example.com", "https://example.com/path", "https://example.com?x=1", "https://attacker.example", "https://example.openai.azure.com.attacker.example", "https://example.openai.azure.com:8443"):
            with self.assertRaises(ValueError):
                upstream_url(endpoint, "deployment", "v1")

    def test_request_boundaries_and_model_discovery(self):
        def never(body):
            self.fail("Invalid request reached Azure")
        with running(never) as port:
            self.assertEqual(call(port, "GET", "/v1/models", headers={"Authorization": "Bearer wrong"})[0], 401)
            self.assertEqual(call(port, "GET", "/v1/models", headers={"Origin": "https://example.com"})[0], 403)
            status, _, body = call(port, "GET", "/v1/models")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["data"][0]["id"], "deployment")
            self.assertEqual(call(port, body=b"not-json")[0], 400)
            self.assertEqual(call(port, body=b'{"messages":[],"model":"wrong"}')[0], 400)
            self.assertEqual(call(port, path="/v1/responses", body=b"{}")[0], 404)
            self.assertEqual(call(port, body=b"{}", headers={"Content-Length": str(MAX_BODY + 1)})[0], 413)

    def test_stream_arrives_before_upstream_finishes_and_tools_are_preserved(self):
        payload = json.dumps({"model": "deployment", "messages": [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
            ]}, {"role": "tool", "tool_call_id": "call1", "content": "file contents"}
        ], "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
            "stream": True}).encode()
        first = b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}\n\n'
        release = threading.Event()

        class SlowResponse(Response):
            calls = 0
            def read1(self, size):
                self.calls += 1
                if self.calls == 1:
                    return first
                if self.calls == 2:
                    release.wait(5)
                    return b"data: [DONE]\n\n"
                return b""

        def backend(body):
            self.assertEqual(body, payload)
            return SlowResponse()

        with running(backend) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            try:
                connection.request("POST", "/v1/chat/completions", body=payload,
                                   headers={"Authorization": "Bearer local-test-key"})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(len(first)), first)
                release.set()
                self.assertEqual(response.read(), b"data: [DONE]\n\n")
            finally:
                release.set()
                connection.close()

    def test_azure_errors_and_retry_after_survive(self):
        error = b'{"error":{"code":"RateLimitReached"}}'
        def backend(body):
            raise urllib.error.HTTPError("https://example.com", 429, "Too Many Requests",
                                         {"Content-Type": "application/json", "Retry-After": "15"}, io.BytesIO(error))
        with running(backend) as port:
            status, headers, body = call(port, body=b'{"model":"deployment","messages":[]}')
            self.assertEqual((status, headers["Retry-After"], body), (429, "15", error))


if __name__ == "__main__":
    unittest.main()
