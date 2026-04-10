#!/usr/bin/env python3

import os
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ["/", "/health"]:
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {
                "status": "healthy",
                "service": "mwd-copilot-test",
                "message": "Hello from Railway!",
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting simple HTTP server on port {port}")
    server = HTTPServer(("0.0.0.0", port), SimpleHandler)
    print("Server ready!")
    server.serve_forever()
