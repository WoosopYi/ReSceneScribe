#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description='Serve the packaged ReSceneScribe viewer.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8132)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(root), **kw)
    url = f'http://{args.host}:{args.port}/viewer/index.html?quality=ultra&trail'
    print(f'Serving {root}')
    print(url)
    with ReusableTCPServer((args.host, args.port), handler) as httpd:
        httpd.serve_forever()


if __name__ == '__main__':
    main()
