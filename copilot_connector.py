#!/usr/bin/env python3
"""VS Code BYOK -> loopback proxy -> Azure OpenAI using VM managed identity.

Python 3.10+. Install: python -m pip install azure-identity
Run on the Azure VM that owns the identity, not an unrelated workstation.

  python copilot_connector.py check --endpoint https://RESOURCE.openai.azure.com --deployment DEPLOYMENT --client-id UAMI_CLIENT_ID
  python copilot_connector.py serve --endpoint https://RESOURCE.openai.azure.com --deployment DEPLOYMENT --client-id UAMI_CLIENT_ID

Omit --client-id for a system-assigned identity. --auth azure-cli explicitly
uses an authorized developer's `az login` identity instead of the VM identity.
Only Chat Completions is supported; this does not replace Copilot completions.
"""

import argparse
import http.client
import hmac
import json
import os
import secrets
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import UUID

MAX_BODY = 16 * 1024 * 1024
SCOPE = "https://cognitiveservices.azure.com/.default"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never send an Azure bearer token to a redirected destination.
        return None


def upstream_url(endpoint, deployment, api_version):
    parsed = urllib.parse.urlsplit(endpoint)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ValueError("--endpoint must be an HTTPS resource root, e.g. https://RESOURCE.openai.azure.com")
    if not parsed.hostname.endswith(".openai.azure.com") or parsed.port not in (None, 443):
        raise ValueError("This connector only accepts Azure public-cloud *.openai.azure.com endpoints on port 443")
    if not deployment.strip():
        raise ValueError("--deployment must be the Azure deployment name")
    return (endpoint.rstrip("/") + "/openai/deployments/"
            + urllib.parse.quote(deployment, safe="") + "/chat/completions?"
            + urllib.parse.urlencode({"api-version": api_version}))


class AzureBackend:
    def __init__(self, args):
        from azure.identity import AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider

        self.url = upstream_url(args.endpoint, args.deployment, args.api_version)
        if args.auth == "azure-cli":
            self.credential = AzureCliCredential()
        else:
            self.credential = ManagedIdentityCredential(client_id=args.client_id) if args.client_id else ManagedIdentityCredential()
        self.token = get_bearer_token_provider(self.credential, SCOPE)
        self.token_lock = threading.Lock()
        self.opener = urllib.request.build_opener(NoRedirect())

    def __call__(self, body):
        # SDK caches and refreshes tokens. Serialize its shared cache access.
        with self.token_lock:
            token = self.token()
        request = urllib.request.Request(self.url, data=body, headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Accept-Encoding": "identity",
        })
        return self.opener.open(request, timeout=180)


def make_handler(backend, deployment, local_key):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0 close-delimited bodies allow immediate SSE forwarding without
        # inventing Content-Length or buffering an entire model response.
        protocol_version = "HTTP/1.0"

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *args):
            pass  # Do not log prompts, auth headers, or response bodies.

        def reply(self, status, data):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            # A web page must not be able to use this local endpoint via CORS.
            if self.headers.get("Origin"):
                self.reply(403, {"error": {"message": "Browser-origin requests are not accepted"}})
                return False
            actual = self.headers.get("Authorization", "").encode("utf-8")
            expected = ("Bearer " + local_key).encode("utf-8")
            if not hmac.compare_digest(actual, expected):
                self.reply(401, {"error": {"message": "Use the connector's local API key"}})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path != "/v1/models":
                self.reply(404, {"error": {"message": "Supported: /v1/models, /v1/chat/completions"}})
                return
            self.reply(200, {"object": "list", "data": [
                {"id": deployment, "object": "model", "created": 0, "owned_by": "azure"}
            ]})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != "/v1/chat/completions":
                self.reply(404, {"error": {"message": "Only /v1/chat/completions is supported; select Chat Completions in VS Code"}})
                return
            if self.headers.get("Transfer-Encoding"):
                self.reply(400, {"error": {"message": "Send a JSON body with Content-Length"}})
                return
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1:
                    raise ValueError("One Content-Length header is required")
                length = int(lengths[0])
                if not 0 < length <= MAX_BODY:
                    self.reply(413, {"error": {"message": "Request body must be between 1 byte and 16 MiB"}})
                    return
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body")
                payload = json.loads(raw)
                if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
                    raise ValueError("JSON must contain a messages array")
                if payload.get("model") != deployment:
                    raise ValueError("model must match the configured Azure deployment name")
            except (ValueError, UnicodeError) as exc:
                self.reply(400, {"error": {"message": str(exc)}})
                return
            except (TimeoutError, OSError):
                self.reply(408, {"error": {"message": "Timed out reading request body"}})
                return

            try:
                try:
                    response = backend(raw)  # Preserve tools, tool results, and stream options.
                except urllib.error.HTTPError as exc:
                    response = exc  # Preserve Azure status, error body, and Retry-After.
            except Exception as exc:
                print("Upstream connection/authentication failed: " + type(exc).__name__, file=sys.stderr)
                self.reply(502, {"error": {"message": "Azure connection/authentication failed. Run the check command on this machine."}})
                return

            with response:
                self.send_response(response.status)
                for name in ("Content-Type", "Retry-After", "apim-request-id"):
                    value = response.headers.get(name)
                    if value:
                        self.send_header(name, value)
                self.send_header("Connection", "close")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while True:
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (OSError, urllib.error.URLError, http.client.HTTPException):
                    # Headers may already be sent: close, never append JSON to SSE.
                    print("Stream interrupted (client disconnected or upstream timeout).", file=sys.stderr)
    return Handler


def check(backend, deployment):
    body = json.dumps({"model": deployment, "messages": [
        {"role": "user", "content": "Reply with only OK."}
    ], "stream": False}).encode()
    try:
        with backend(body) as response:
            data = json.load(response)
        if not isinstance(data.get("choices"), list) or not data["choices"]:
            print("FAIL: response did not contain Chat Completions choices.", file=sys.stderr)
            return 1
        print("PASS: identity, network, deployment and Chat Completions work.")
        return 0
    except urllib.error.HTTPError as exc:
        hints = {400: "Check deployment/API compatibility.", 401: "Check token audience/identity.",
                 403: "Check OpenAI User RBAC, firewall and private network access.",
                 404: "Check endpoint and deployment name.", 429: "Azure quota/rate limit reached."}
        print(f"FAIL: Azure HTTP {exc.code}. {hints.get(exc.code, 'Check Azure service/network status.')}", file=sys.stderr)
        request_id = exc.headers.get("apim-request-id")
        if request_id:
            print("Azure request ID: " + request_id, file=sys.stderr)
        exc.close()
        return 1
    except Exception as exc:
        print("FAIL: " + type(exc).__name__ + ". Check VM identity assignment, DNS, HTTPS access and corporate certificates.", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "serve"])
    parser.add_argument("--endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT"))
    parser.add_argument("--deployment", default=os.getenv("AZURE_OPENAI_DEPLOYMENT"))
    parser.add_argument("--client-id", default=os.getenv("AZURE_CLIENT_ID"), help="User-assigned identity CLIENT ID; omit for system identity")
    parser.add_argument("--api-version", default="2024-10-21")
    parser.add_argument("--auth", choices=["managed-identity", "azure-cli"], default="managed-identity")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.endpoint or not args.deployment:
        parser.error("Provide --endpoint and --deployment (or AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT)")
    try:
        upstream_url(args.endpoint, args.deployment, args.api_version)
        if args.client_id:
            UUID(args.client_id)
        if not 1 <= args.port <= 65535:
            raise ValueError("--port must be 1..65535")
    except ValueError as exc:
        parser.error(str(exc))
    if args.auth == "azure-cli" and args.client_id:
        parser.error("--auth azure-cli uses your user identity; remove --client-id and AZURE_CLIENT_ID")
    try:
        backend = AzureBackend(args)
    except ImportError:
        parser.error("Install the dependency: python -m pip install azure-identity")
    try:
        if args.command == "check":
            return check(backend, args.deployment)
        if check(backend, args.deployment):
            return 1
        local_key = secrets.token_urlsafe(32)
        with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(backend, args.deployment, local_key)) as server:
            print(f"\nListening: http://127.0.0.1:{args.port}/v1/chat/completions")
            print("VS Code: Manage Language Models -> Add Models -> Custom Endpoint")
            print("API type: Chat Completions")
            print("Model ID: " + args.deployment)
            print("Local API key (new on each restart): " + local_key)
            print("Set token limits and tool-calling capability to match your deployed model.")
            print("Keep this terminal open. Ctrl+C stops the connector.", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    except OSError as exc:
        print("Cannot start connector: " + str(exc), file=sys.stderr)
        return 1
    finally:
        backend.credential.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
