"""
Passport loader: discover MODEL_PASSPORT.json + SIGNOFF.json at a checkpoint
root, parse them as typed dataclasses, and degrade gracefully when files are
missing or the schema version is unknown.

The loader never raises on missing files or bad JSON; it returns a
`PassportLoadResult` with a status flag plus diagnostic text. Downstream
checks read the status and decide what to report (soft signal vs hard
fail), per the plan's degraded-mode rules:

  no passport at all (legacy)             -> LoadStatus.NO_PASSPORT
  passport ok, signoff missing            -> LoadStatus.PASSPORT_NO_SIGNOFF
  passport ok, signoff ok                 -> LoadStatus.PASSPORT_AND_SIGNOFF
  passport unparseable / unknown version  -> LoadStatus.PASSPORT_DEGRADED

Hard fail decisions (e.g. signoff-checksum-mismatch is always a hard fail)
live in the checks themselves; the loader only reports what it found.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import (
    ModelPassport,
    Signoff,
    is_passport_version_supported,
    is_signoff_version_supported,
)


PASSPORT_FILENAME = "MODEL_PASSPORT.json"
SIGNOFF_FILENAME = "SIGNOFF.json"


class LoadStatus(str, Enum):
    NO_PASSPORT = "no_passport"
    PASSPORT_NO_SIGNOFF = "passport_no_signoff"
    PASSPORT_AND_SIGNOFF = "passport_and_signoff"
    PASSPORT_DEGRADED = "passport_degraded"


@dataclass
class PassportLoadResult:
    """Result of trying to load passport + signoff from a checkpoint root."""

    checkpoint_dir: Path
    status: LoadStatus

    passport: Optional[ModelPassport] = None
    passport_path: Optional[Path] = None  # only set when file existed on disk
    passport_raw: Optional[Dict[str, Any]] = None  # raw parsed JSON (pre-schema)

    signoff: Optional[Signoff] = None
    signoff_path: Optional[Path] = None  # only set when file existed on disk
    signoff_raw: Optional[Dict[str, Any]] = None

    # We must distinguish "no file on disk" from "file present but unusable".
    # signoff_present_on_disk says "we saw a SIGNOFF.json file", regardless
    # of whether parse/schema mapping succeeded. Same for passport.
    passport_present_on_disk: bool = False
    signoff_present_on_disk: bool = False

    # True when a file existed on disk but we couldn't produce a usable
    # dataclass from it (json malformed, top-level not an object, schema
    # version unsupported, dataclass mapping raised, ...).
    passport_unparseable: bool = False
    signoff_unparseable: bool = False

    # Free-text diagnostic strings -- one entry per "thing to know about
    # how parsing went". Empty when everything loaded cleanly. Each note
    # includes the absolute path of the file it relates to where useful.
    notes: List[str] = field(default_factory=list)

    @property
    def has_passport(self) -> bool:
        return self.passport is not None

    @property
    def has_signoff(self) -> bool:
        return self.signoff is not None

    def section_present(self, name: str) -> bool:
        """True if the named top-level passport section was populated.

        We can't easily tell "field present in file" from "field defaulted by
        the dataclass" purely from the parsed object, so we check the raw
        dict. This matters for the degraded-mode rule "passport present but
        a section missing -> that section's checks become not_checked".
        """
        return self.passport_raw is not None and name in self.passport_raw


# ── Loaders ─────────────────────────────────────────────────────────────


def load_passport(checkpoint_dir: str | Path) -> PassportLoadResult:
    """Discover and parse MODEL_PASSPORT.json + SIGNOFF.json at the root.

    NEVER raises on a malformed checkpoint -- all failures surface as
    `notes` plus the appropriate `LoadStatus`. The caller may then decide
    soft-signal vs hard-fail per check.
    """
    root = Path(checkpoint_dir)

    # File-vs-directory guard: if caller passed a path to a regular file,
    # use its parent as the root. This is the common slip-up of pointing
    # at MODEL_PASSPORT.json directly.
    notes: List[str] = []
    if root.is_file():
        notes.append(
            f"checkpoint_dir={root!s} is a file; using parent directory as "
            "checkpoint root"
        )
        root = root.parent
    elif not root.exists():
        return PassportLoadResult(
            checkpoint_dir=root,
            status=LoadStatus.NO_PASSPORT,
            notes=[f"checkpoint_dir={root!s} does not exist"],
        )
    elif not root.is_dir():
        # device, socket, etc.
        return PassportLoadResult(
            checkpoint_dir=root,
            status=LoadStatus.NO_PASSPORT,
            notes=[f"checkpoint_dir={root!s} is not a directory"],
        )

    passport_path = root / PASSPORT_FILENAME
    signoff_path = root / SIGNOFF_FILENAME

    # ── Passport ──
    passport: Optional[ModelPassport] = None
    passport_raw: Optional[Dict[str, Any]] = None
    passport_present = passport_path.is_file()
    passport_unparseable = False

    if passport_present:
        passport_raw, parse_note = _read_json_object(passport_path)
        if parse_note:
            notes.append(parse_note)
            passport_unparseable = True

        if passport_raw is not None:
            version = passport_raw.get("schema_version", "<missing>")
            if not is_passport_version_supported(version):
                notes.append(
                    f"{passport_path!s}: schema_version={version!r} not in "
                    "supported versions; loaded in degraded mode"
                )
                passport_unparseable = True
                try:
                    passport = ModelPassport.from_dict(passport_raw)
                except Exception as e:
                    notes.append(
                        f"{passport_path!s}: degraded-mode load also failed: "
                        f"{type(e).__name__}: {e}"
                    )
                    passport = None
            else:
                try:
                    passport = ModelPassport.from_dict(passport_raw)
                except Exception as e:
                    notes.append(
                        f"{passport_path!s}: failed to map JSON to dataclass: "
                        f"{type(e).__name__}: {e}"
                    )
                    passport_unparseable = True
                    passport = None
    else:
        notes.append(f"{PASSPORT_FILENAME} not found at {root!s}")

    # ── Signoff ──
    signoff: Optional[Signoff] = None
    signoff_raw: Optional[Dict[str, Any]] = None
    signoff_present = signoff_path.is_file()
    signoff_unparseable = False

    if signoff_present:
        signoff_raw, parse_note = _read_json_object(signoff_path)
        if parse_note:
            notes.append(parse_note)
            signoff_unparseable = True

        if signoff_raw is not None:
            version = signoff_raw.get("schema_version", "<missing>")
            if not is_signoff_version_supported(version):
                notes.append(
                    f"{signoff_path!s}: schema_version={version!r} not in "
                    "supported versions; loaded in degraded mode"
                )
                # Per plan: unsupported signoff schema is SOFT_SIGNAL not
                # FAIL, and we still try to map it for what we can read.
            try:
                signoff = Signoff.from_dict(signoff_raw)
            except Exception as e:
                notes.append(
                    f"{signoff_path!s}: failed to map JSON to dataclass: "
                    f"{type(e).__name__}: {e}"
                )
                signoff = None
                signoff_unparseable = True

    # ── Status decision ──
    # NO_PASSPORT only when there was literally no file. A present-but-broken
    # passport is PASSPORT_DEGRADED so checks know to soft-signal rather
    # than skip entirely.
    if not passport_present:
        status = LoadStatus.NO_PASSPORT
    elif passport_unparseable or passport is None:
        status = LoadStatus.PASSPORT_DEGRADED
    elif signoff is None:
        # Covers both "signoff not on disk" and "signoff on disk but
        # unparseable" -- the new flags let downstream checks tell them apart.
        status = LoadStatus.PASSPORT_NO_SIGNOFF
    else:
        status = LoadStatus.PASSPORT_AND_SIGNOFF

    return PassportLoadResult(
        checkpoint_dir=root,
        status=status,
        passport=passport,
        passport_path=passport_path if passport_present else None,
        passport_raw=passport_raw,
        signoff=signoff,
        signoff_path=signoff_path if signoff_present else None,
        signoff_raw=signoff_raw,
        passport_present_on_disk=passport_present,
        signoff_present_on_disk=signoff_present,
        passport_unparseable=passport_unparseable,
        signoff_unparseable=signoff_unparseable,
        notes=notes,
    )


def _read_json_object(path: Path) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Read a JSON file expected to be a top-level object.

    Returns (parsed_dict, error_note). On any failure -- bad I/O, malformed
    JSON, or top-level value that is not an object -- returns (None, note).
    Never raises.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except OSError as e:
        return None, f"{path!s}: cannot read file: {type(e).__name__}: {e}"
    except json.JSONDecodeError as e:
        return None, f"{path!s}: invalid JSON: line {e.lineno}, col {e.colno}: {e.msg}"

    if not isinstance(data, dict):
        return None, (
            f"{path!s}: top-level JSON value must be an object, got "
            f"{type(data).__name__}"
        )
    return data, None
