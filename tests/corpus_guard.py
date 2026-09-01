"""
Corpus availability guard.

Three tests in this suite exercise the real-data audit corpus: cached
Strava streams carrying GPS, held in a private directory outside the
repository. They are not synthetic and cannot be made so without
fabricating data, which would make them prove nothing.

A clean checkout has no such corpus, so those tests fail with assertion
errors that look like matcher defects and are not. This module lets them
skip with a reason that names the missing directory, while remaining
fully active, with every assertion unchanged, wherever the corpus exists.

Only the genuinely corpus-dependent tests use these guards. The other 110
tests are synthetic or unit tests and stay mandatory: an empty corpus
must never turn the suite green by silently skipping it.
"""

import json

import pytest

# Imported rather than re-derived, so the guard cannot drift from the
# loader it is guarding.
from audit7.corpus import STREAM_DIR

# The route ids the geographic-trail tests assert against by name. Those
# tests encode properties of this specific corpus (which recordings are
# the same trail, and that it resolves to five trails), so a different
# corpus should skip rather than fail: the failure would be a statement
# about the data, not about the code.
AUDIT_ROUTE_IDS = frozenset({
    "19131631580", "19476565994", "19621145681",
    "19670306718", "19853326285", "19869723537",
})

_ABSENT = ("real-data audit corpus unavailable: no cached GPS streams in %s. "
           "These tests read private Strava stream data that is not part of "
           "the repository and is not synthesisable without fabricating it."
           % STREAM_DIR)

_INCOMPLETE = ("real-data audit corpus incomplete: %s does not contain the "
               "six route ids the geographic-trail assertions are written "
               "against (%s). A different corpus would fail these tests for "
               "reasons about the data rather than the code."
               % (STREAM_DIR, ", ".join(sorted(AUDIT_ROUTE_IDS))))


def gps_stream_ids():
    """Ids of cached streams that actually carry a GPS track.

    Presence of a file is not enough: several cached streams have no
    latlng, and every corpus-dependent test here needs geography.
    """
    if not STREAM_DIR.is_dir():
        return frozenset()
    found = set()
    for path in STREAM_DIR.glob("*.json"):
        try:
            obj = json.load(open(path))
        except Exception:                                   # noqa: BLE001
            continue
        if obj.get("latlng"):
            found.add(path.stem)
    return frozenset(found)


def _have_any():
    return bool(gps_stream_ids())


def _have_audit_corpus():
    return AUDIT_ROUTE_IDS <= gps_stream_ids()


requires_corpus = pytest.mark.skipif(not _have_any(), reason=_ABSENT)

requires_audit_corpus = pytest.mark.skipif(
    not _have_audit_corpus(),
    reason=_ABSENT if not _have_any() else _INCOMPLETE)
