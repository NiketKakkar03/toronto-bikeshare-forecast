"""Content-addressed, immutable storage for validated source payloads."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RawCapture(BaseModel):
    """Manifest entry stored next to an immutable raw response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_url: str = Field(min_length=1)
    feed_name: str = Field(min_length=1)
    retrieved_at: datetime
    feed_updated_at: datetime
    status: str = "validated"
    schema_version: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=0)
    raw_path: str = Field(min_length=1)

    @field_validator("retrieved_at", "feed_updated_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("timestamp must be UTC")
        return value


class RawCaptureStore:
    """Write validated payloads once and return the original manifest on retries."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def capture(
        self,
        payload: BaseModel,
        *,
        source_url: str,
        feed_name: str,
        retrieved_at: datetime,
        feed_updated_at: datetime,
        schema_version: str,
        row_count: int,
    ) -> tuple[RawCapture, bool]:
        """Persist canonical validated JSON and its manifest; return whether it was new."""
        document = (
            json.dumps(
                payload.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        content_hash = hashlib.sha256(document).hexdigest()
        directory = self.root / feed_name / "objects" / content_hash[:2]
        raw_path = directory / f"{content_hash}.json"
        manifest_path = directory / f"{content_hash}.manifest.json"
        manifest = RawCapture(
            source_url=source_url,
            feed_name=feed_name,
            retrieved_at=retrieved_at,
            feed_updated_at=feed_updated_at,
            schema_version=schema_version,
            content_hash=content_hash,
            row_count=row_count,
            raw_path=str(raw_path.relative_to(self.root)),
        )

        if manifest_path.exists():
            return RawCapture.model_validate_json(manifest_path.read_text(encoding="utf-8")), False

        directory.mkdir(parents=True, exist_ok=True)
        created = self._write_exclusive(raw_path, document)
        self._write_exclusive(
            manifest_path,
            (manifest.model_dump_json(indent=2) + "\n").encode(),
        )
        return manifest, created

    @staticmethod
    def _write_exclusive(path: Path, contents: bytes) -> bool:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            if path.read_bytes() != contents:
                raise RuntimeError(f"immutable artifact collision: {path}") from None
            return False
        with os.fdopen(descriptor, "wb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        return True
