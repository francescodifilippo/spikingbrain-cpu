"""Download a pinned checkpoint and verify every file against its SHA-256.

Usage: python -m spikingbrain_cpu.fetch MANIFEST.json TARGET_DIR [BASE_URL]

The manifest lists ``{"path", "size", "sha256", "revision"}`` entries. Files
already present with the right digest are skipped; partial downloads are
written to ``*.part`` and renamed only after verification.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_BASE = "https://www.modelscope.cn/models/{model}/resolve/{revision}/{path}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(manifest: dict, target: Path, base: str = DEFAULT_BASE, attempts: int = 3) -> list[dict]:
    target.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in manifest["files"]:
        dest = target / entry["path"]
        if dest.exists() and dest.stat().st_size == entry["size"] and sha256_of(dest) == entry["sha256"]:
            results.append({"path": entry["path"], "state": "PRESENT_VERIFIED"})
            continue
        url = base.format(model=manifest["model"], revision=entry["revision"], path=entry["path"])
        part = dest.with_name(dest.name + ".part")
        for attempt in range(1, attempts + 1):
            try:
                started = time.monotonic()
                digest = hashlib.sha256()
                with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as out:
                    while chunk := response.read(8 * 1024 * 1024):
                        digest.update(chunk)
                        out.write(chunk)
                if part.stat().st_size != entry["size"] or digest.hexdigest() != entry["sha256"]:
                    raise ValueError("size or sha256 mismatch")
                part.rename(dest)
                results.append({"path": entry["path"], "state": "DOWNLOADED_VERIFIED",
                                "seconds": round(time.monotonic() - started, 1)})
                print(json.dumps(results[-1]), flush=True)
                break
            except (OSError, ValueError) as exc:
                part.unlink(missing_ok=True)
                print(json.dumps({"path": entry["path"], "attempt": attempt, "error": type(exc).__name__}), flush=True)
                if attempt == attempts:
                    raise
                time.sleep(5 * attempt)
    return results


def main() -> None:
    manifest = json.loads(Path(sys.argv[1]).read_text())
    base = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_BASE
    results = fetch(manifest, Path(sys.argv[2]), base)
    print(json.dumps({"files": len(results), "state": "COMPLETE"}))


if __name__ == "__main__":
    main()
