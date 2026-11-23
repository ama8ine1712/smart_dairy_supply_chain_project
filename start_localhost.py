"""
Zero-Dependency Localhost Launcher for Smart Dairy Supply Chain DSS
Serves index.html on http://127.0.0.1:8080 and opens it in your default browser.
"""

import http.server
import os
import socketserver
import webbrowser
from functools import partial

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def main():
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=DIRECTORY)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        url = f"http://127.0.0.1:{PORT}/index.html"
        print("=" * 64)
        print(f"Smart Dairy DSS running on: {url}")
        print("Press Ctrl+C in this terminal to stop the server.")
        print("=" * 64)
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
