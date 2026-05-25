#!/usr/bin/env python3
"""Launch the AudioCraft Studio web UI."""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="AudioCraft Studio Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        "ui.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
