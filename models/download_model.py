"""Download the LLaMA 3.2 3B Instruct (Q6_K_L) GGUF model into ./models."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError as exc:  # requests is convenient but optional
    raise SystemExit(
        "The download helper requires the 'requests' package.\n"
        "Install it with: python -m pip install requests"
    ) from exc

DEFAULT_FILENAME = "Llama-3.2-3B-Instruct-Q6_K_L.gguf"
DEFAULT_URL = (
    "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct-GGUF/resolve/main/"
    + DEFAULT_FILENAME
)


def build_headers(token: Optional[str]) -> dict[str, str]:
    headers: dict[str, str] = {"User-Agent": "ai-chatbot-model-downloader"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def download(url: str, destination: Path, token: Optional[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, headers=build_headers(token), stream=True, timeout=60) as response:
        if response.status_code == 302:
            # Resolve Hugging Face redirect manually when tokenless.
            redirect = response.headers.get("Location")
            if redirect:
                return download(redirect, destination, token)
        response.raise_for_status()

        total = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024  # 1 MB
        written = 0

        with destination.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    written += len(chunk)
                    if total:
                        done = written / total * 100
                        sys.stdout.write(f"\rDownloaded {written/1_048_576:.1f} MiB ({done:.1f}%)")
                    else:
                        sys.stdout.write(f"\rDownloaded {written/1_048_576:.1f} MiB")
                    sys.stdout.flush()

    sys.stdout.write("\nDownload complete: " + str(destination) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Download URL (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / DEFAULT_FILENAME,
        help="Output path for the GGUF file (default: %(default)s)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HUGGINGFACE_TOKEN"),
        help="Hugging Face access token; defaults to HUGGINGFACE_TOKEN env var.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    download(args.url, args.output, token=args.token)


if __name__ == "__main__":
    main()
