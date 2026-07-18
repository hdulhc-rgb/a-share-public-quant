#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


BLOCKED_EXTENSIONS = {
    ".pdf", ".xlsx", ".xls", ".docx", ".pptx", ".m4a",
    ".png", ".jpg", ".jpeg", ".heic",
}
BLOCKED_NAME_FRAGMENTS = {
    "holdings", "positions", "cost_basis", "vesting", "rsu",
    "broker", "account_statement", "mortgage", "private_data",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "assigned API secret": re.compile(
        r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
    "email address": re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    "mainland phone number": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
}


def tracked_files(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
        relative = [item for item in result.stdout.split(b"\0") if item]
        return [root / item.decode("utf-8") for item in relative]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [
            path
            for path in root.rglob("*")
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
        ]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for path in tracked_files(root):
        relative = path.relative_to(root).as_posix()
        lowered = relative.lower()
        if path.suffix.lower() in BLOCKED_EXTENSIONS:
            violations.append(f"blocked extension: {relative}")
            continue
        if any(fragment in lowered for fragment in BLOCKED_NAME_FRAGMENTS):
            violations.append(f"blocked private filename: {relative}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append(f"unexpected binary file: {relative}")
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                violations.append(f"{label}: {relative}")

    if violations:
        print("PRIVACY_GUARD_FAILED", file=sys.stderr)
        for violation in sorted(set(violations)):
            print(f"- {violation}", file=sys.stderr)
        return 2
    print("PRIVACY_GUARD_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
