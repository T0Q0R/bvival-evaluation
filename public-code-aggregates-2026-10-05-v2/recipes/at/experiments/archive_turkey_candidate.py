"""Archive only the pinned licensed December workbook; never parse its cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

COMMIT = "df15151fc655a54a43e47ea6917c400eaa7f8670"
FILENAME = "Aralık_2025.xlsx"
URL = f"https://raw.githubusercontent.com/masoudmaleki/used-car-temporal-ml/{COMMIT}/{quote(FILENAME)}"
EXPECTED_SIZE = 4155327
EXPECTED_SHA256 = "99aa8018807a72082704047a72d93aca202c4e30126c62f1dec8e01d1ad8f77b"
EXPECTED_BLOB = "6d477e3c0f6b15abcb7efd2e3714f348f8909643"


class ExactArchiveRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if newurl != URL:
            raise ValueError("Redirect away from the pinned raw archive URL blocked")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def verify_archive(path: Path) -> dict:
    size = path.stat().st_size
    if size != EXPECTED_SIZE:
        raise ValueError("Archive size differs from the pinned publisher declaration")
    sha = hashlib.sha256()
    blob = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            sha.update(block)
            blob.update(block)
    if sha.hexdigest() != EXPECTED_SHA256 or blob.hexdigest() != EXPECTED_BLOB:
        raise ValueError("Archive differs from the pinned SHA-256 or Git blob")
    return {"bytes": size, "sha256": sha.hexdigest(), "git_blob_sha1": blob.hexdigest()}


def fetch_range(start: int, end: int, logs: list) -> bytes:
    """Bounded exact HTTP ranges; do not join responses without range checks."""
    for number in (1, 2):
        try:
            request = Request(URL, headers={"User-Agent": "BVI-Val-Feature-Audit/1.0",
                                           "Range": f"bytes={start}-{end}"})
            with build_opener(ExactArchiveRedirect()).open(request, timeout=25) as response:
                if (response.geturl() != URL or response.getcode() != 206
                        or response.headers.get("Content-Range") != f"bytes {start}-{end}/{EXPECTED_SIZE}"):
                    raise ValueError("Range response does not match pinned URL/byte coordinates")
                parts, received = [], 0
                while True:
                    block = response.read(min(65536, end - start + 2 - received))
                    if not block:
                        break
                    parts.append(block)
                    received += len(block)
                    if received > end - start + 1:
                        break
                data = b"".join(parts)
                if len(data) != end - start + 1:
                    raise ValueError(f"Range response length mismatch: expected {end - start + 1}, received {len(data)}")
            logs.append({"start": start, "end": end, "attempt": number, "success": True,
                         "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            return data
        except Exception as exc:
            logs.append({"start": start, "end": end, "attempt": number, "success": False,
                         "error": type(exc).__name__ + ": " + str(exc)})
            if number == 2:
                raise
    raise AssertionError("Unreachable range state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transport", choices=["full", "ranged", "curl"], default="full")
    args = parser.parse_args()
    metadata = json.loads(args.metadata_audit.read_text())
    candidate = next(c for c in metadata["candidates"] if c["candidate_id"] == "turkey_arabam_december_2025")
    if (candidate["pinned_commit"] != COMMIT or candidate["dataset_license_declared"] != "CC BY 4.0"
            or candidate["publisher_sha256"] != EXPECTED_SHA256
            or candidate["git_blob_sha1"] != EXPECTED_BLOB or candidate["publisher_bytes"] != EXPECTED_SIZE):
        raise ValueError("Previously audited version/license declaration does not match")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError("Archive output directory must be empty")
    attempts, verified = [], None
    for number in (1, 2):
        partial = args.output_dir / f"attempt_{number}.opaque"
        range_logs = []
        try:
            if args.transport == "curl":
                with partial.open("xb"):
                    partial.chmod(0o600)
                transfer = subprocess.run(["/usr/bin/curl", "--fail", "--silent", "--show-error",
                    "--proto", "=https", "--max-time", "40", "--output", str(partial),
                    "--write-out", "%{http_code}|%{url_effective}", URL],
                    capture_output=True, text=True, timeout=50, check=True)
                if transfer.stdout != "200|" + URL:
                    raise ValueError("curl did not return the exact pinned URL with HTTP 200")
            elif args.transport == "ranged":
                with partial.open("xb") as stream:
                    partial.chmod(0o600)
                    for start in range(0, EXPECTED_SIZE, 262144):
                        stream.write(fetch_range(start, min(start + 262143, EXPECTED_SIZE - 1), range_logs))
            else:
                request = Request(URL, headers={"User-Agent": "BVI-Val-Feature-Audit/1.0"})
                with build_opener(ExactArchiveRedirect()).open(request, timeout=25) as response:
                    if response.geturl() != URL:
                        raise ValueError("Unexpected archive final URL")
                    declared = response.headers.get("Content-Length")
                    if declared is not None and int(declared) != EXPECTED_SIZE:
                        raise ValueError("HTTP size differs from expected archive size")
                    received = 0
                    with partial.open("xb") as stream:
                        partial.chmod(0o600)
                        while True:
                            block = response.read(min(65536, EXPECTED_SIZE + 1 - received))
                            if not block:
                                break
                            stream.write(block)
                            received += len(block)
                            if received > EXPECTED_SIZE:
                                raise ValueError("Archive exceeds pinned byte limit")
            verified = verify_archive(partial)
            destination = args.output_dir / FILENAME
            partial.rename(destination)
            attempts.append({"attempt": number, "success": True, **verified, "range_requests": range_logs})
            break
        except Exception as exc:
            # Preserve failed opaque bytes; never silently overwrite an audit attempt.
            attempts.append({"attempt": number, "success": False, "error": type(exc).__name__ + ": " + str(exc), "range_requests": range_logs})
    audit = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "url": URL,
             "commit": COMMIT, "license_declared": "CC BY 4.0", "metadata_audit_sha256":
             hashlib.sha256(args.metadata_audit.read_bytes()).hexdigest(), "attempts": attempts,
             "transport": args.transport,
             "archive_verified": verified is not None, "individual_cells_parsed": False,
             "no_data_redistribution": True, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / "archive_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.output_dir / "archive_runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"archive_verified": audit["archive_verified"], "cells_parsed": False, "attempts": len(attempts)}))
    if verified is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
