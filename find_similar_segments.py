#!/usr/bin/env python3
"""
find_similar_segments.py

Finds Strava running segments near a given location whose CLIMBING
DEMAND most closely matches a target GPX file: similar ordered profile
AND similar steepness and vertical, at comparable length.

Both halves of that matter, and the magnitude half matters more. Measured
rank correlation of the score against axes defined outside the matcher,
over 1500 real pairs: steepness difference 0.55, magnitude-preserving
shape 0.47, ordered shape with steepness normalized away 0.13. A 3
percent and a 9 percent climb of identical profile shape are NOT reported
as similar, and that is deliberate. An earlier version of this text
described the tool as matching ordered shape rather than average grade,
which inverted the true emphasis.

HOW A MATCH IS DECIDED
----------------------
Both the target and every candidate are converted to a normalized
representation first: elevation resampled onto a uniform arc-length grid,
and grade estimated as a least squares slope over a fixed physical scale
(--grade-res-m, default 70 m). That step is what makes a target recorded
at 1 m GPS spacing comparable to a Strava stream at 10 m spacing.

Windows of varying length and offset are then generated across each
candidate, screened with a lower bound that provably never exceeds the
true distance, and the survivors scored on four named terms:

  shape        ordered grade sequence, compared by dynamic time warping
               under a bounded alignment tolerance (--max-shift-frac)
  composition  grade distribution regardless of order, exact
               Wasserstein-1
  vertical     ascent and descent against the target, measured at a
               fixed spatial interval so it describes the hill rather
               than the recording device
  length       how far the window is from the target length

The winning window is then refined off the search grid by local descent,
so the reported extent is not quantized to the offset and length steps.

Reported scores are placed against a null model built by scoring random
windows from the same candidate pool, so a weak match is labelled weak
instead of merely being ranked first.

Parameter choices are documented, with the measurements behind them, in
PARAMETERS.md. Tests covering the adversarial cases are in tests/.

--------------------------------------------------------------------------
ONE-TIME SETUP
--------------------------------------------------------------------------
1. Register an app at https://www.strava.com/settings/api
   - Website: http://localhost (or anything)
   - Authorization Callback Domain: localhost
   Note your Client ID and Client Secret.

2. Authorize once (opens a browser URL, you paste back a `code`):

    python3 find_similar_segments.py --authorize \
        --client-id YOUR_ID --client-secret YOUR_SECRET

   This walks you through the OAuth flow and saves tokens to
   ~/.strava_segment_matcher_tokens.json (refresh token doesn't expire
   unless you revoke access in Strava's settings, so this is one-time).

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python3 find_similar_segments.py \
        --gpx BTR_6th_Split_Uphill_Portion.gpx \
        --lat 40.0150 --lon -105.2705 \
        --radius-km 15 \
        --top 10

--lat/--lon is the center of your search area (e.g. Boulder, CO).
--radius-km controls how big a grid of search boxes to scan (bigger =
more API calls = slower, due to Strava rate limits of 100 req/15min).

Output: ranked candidate segments with name, distance, gain and loss,
which direction to run them, a per-term score breakdown, a match-quality
line placing the score against the null model, road access, and a direct
Strava URL.
"""

import argparse
import json
import math
import os
import re
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np
import requests
import gpxpy

from segmatch.match import (MatchConfig, prepare_target,
                             match_segment, null_scores)
from segmatch.profile import (build_profile, vertical_change,
                               detect_quantization)

TOKEN_FILE = Path.home() / ".strava_segment_matcher_tokens.json"

# --------------------------------------------------------------------------
# Persistent disk cache
#
# Strava segment geometry and elevation streams are effectively static, and
# the API rate-limits hard (100 reads / 15 min), so the slow part of a run
# is almost entirely waiting on the network. We cache both the per-tile
# segments/explore results and the per-segment streams to disk, keyed so
# that re-running with a different target GPX, different weights, or an
# overlapping search area reuses everything already fetched. Entries older
# than CACHE_TTL_DAYS are treated as stale and re-fetched; --refresh forces
# a full rebuild.
# --------------------------------------------------------------------------
CACHE_DIR = Path.home() / ".strava_segment_matcher_cache"
EXPLORE_CACHE_DIR = CACHE_DIR / "explore"
STREAM_CACHE_DIR = CACHE_DIR / "streams"
CACHE_TTL_DAYS = 30
CACHE_TTL_SECONDS = CACHE_TTL_DAYS * 24 * 3600

# Set from --refresh in main(): when True, ignore existing cache entries
# and overwrite them with freshly fetched data.
_FORCE_REFRESH = False


def _ensure_cache_dirs():
    EXPLORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STREAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_read(path):
    """Return parsed JSON if the file exists and is within TTL, else None."""
    if _FORCE_REFRESH or not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > CACHE_TTL_SECONDS:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _cache_write(path, obj):
    try:
        with open(path, "w") as f:
            json.dump(obj, f)
    except OSError:
        pass  # cache is best-effort; a write failure just means a re-fetch


def _tile_cache_key(sw_lat, sw_lon, ne_lat, ne_lon):
    # Round to ~11m so floating-point jitter in tile edges still hits the
    # same cache file across runs.
    return (f"{sw_lat:.4f}_{sw_lon:.4f}_{ne_lat:.4f}_{ne_lon:.4f}"
            .replace("-", "m").replace(".", "p") + ".json")
STRAVA_API = "https://www.strava.com/api/v3"
OAUTH_AUTHORIZE_URL = "https://www.strava.com/oauth/authorize"
OAUTH_TOKEN_URL = "https://www.strava.com/oauth/token"


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

def do_authorize(client_id, client_secret):
    redirect_uri = "http://localhost"
    auth_url = (
        f"{OAUTH_AUTHORIZE_URL}?client_id={client_id}&response_type=code"
        f"&redirect_uri={redirect_uri}&approval_prompt=force"
        f"&scope=read,activity:read"
    )
    print("Opening this URL in your browser (or copy/paste it):\n")
    print(auth_url)
    print()
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass
    print(
        "After approving, you'll land on a page at localhost that fails "
        "to load. That's expected - copy the 'code' parameter out of the "
        "browser's address bar.\n"
        "Example: http://localhost/?state=&code=THIS_PART&scope=...\n"
    )
    code = input("Paste the code here: ").strip()

    resp = requests.post(
        OAUTH_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    resp.raise_for_status()
    tokens = resp.json()
    tokens["client_id"] = client_id
    tokens["client_secret"] = client_secret
    TOKEN_FILE.write_text(json.dumps(tokens, indent=2))
    os.chmod(TOKEN_FILE, 0o600)
    print(f"Saved tokens to {TOKEN_FILE}. You're set - run the script "
          f"again without --authorize.")


def get_access_token():
    if not TOKEN_FILE.exists():
        sys.exit(
            "No saved tokens found. Run with --authorize --client-id "
            "... --client-secret ... first."
        )
    tokens = json.loads(TOKEN_FILE.read_text())

    if tokens.get("expires_at", 0) > time.time() + 60:
        return tokens["access_token"]

    # Refresh
    resp = requests.post(
        OAUTH_TOKEN_URL,
        data={
            "client_id": tokens["client_id"],
            "client_secret": tokens["client_secret"],
            "refresh_token": tokens["refresh_token"],
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    resp.raise_for_status()
    new_tokens = resp.json()
    tokens.update(new_tokens)
    TOKEN_FILE.write_text(json.dumps(tokens, indent=2))
    return tokens["access_token"]


# --------------------------------------------------------------------------
# GPX / grade profile helpers
# --------------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def load_gpx_profile(path):
    """Returns (cum_dist_m: np.array, elevation_m: np.array)."""
    with open(path) as f:
        gpx = gpxpy.parse(f)
    points = []
    for track in gpx.tracks:
        for seg in track.segments:
            points.extend(seg.points)
    if not points:
        sys.exit(f"No track points found in {path}")

    cum_dist = [0.0]
    elevs = [points[0].elevation]
    for i in range(1, len(points)):
        d = haversine_m(points[i - 1].latitude, points[i - 1].longitude,
                         points[i].latitude, points[i].longitude)
        cum_dist.append(cum_dist[-1] + d)
        elevs.append(points[i].elevation)

    cum_dist = np.asarray(cum_dist, dtype=float)
    elevs = np.asarray([np.nan if e is None else e for e in elevs],
                        dtype=float)
    # A GPX with missing <ele> values otherwise either crashes on None
    # arithmetic or - worse - silently reports zero gain, because every
    # NaN comparison is False so the ascent filter selects nothing. That
    # drives target_vert_m to 0, which makes the gain term inert without
    # any warning at all.
    bad = ~np.isfinite(elevs)
    if bad.all():
        sys.exit(f"{path} has no usable elevation data (no <ele> values).")
    if bad.any():
        print(f"  warning: {int(bad.sum())} of {len(elevs)} points in "
              f"{path} lack elevation; interpolating across them.")
        elevs[bad] = np.interp(cum_dist[bad], cum_dist[~bad], elevs[~bad])
    return cum_dist, elevs


# --------------------------------------------------------------------------
# Strava API
# --------------------------------------------------------------------------

def _norm_lon(lon):
    """Wrap a longitude into [-180, 180)."""
    return ((lon + 180.0) % 360.0) - 180.0


def km_to_deg_lat(km):
    return km / 111.0


def km_to_deg_lon(km, at_lat):
    # cos() collapses at the poles: at lat 90 this returns ~6e14 degrees,
    # silently producing one absurd tile rather than raising. Clamp to
    # roughly 85 degrees, past which longitude tiling is meaningless anyway.
    cos_lat = math.cos(math.radians(max(-85.0, min(85.0, at_lat))))
    return km / (111.0 * cos_lat)


def explore_segments(token, lat, lon, radius_km, box_km=4.0):
    """Grid-search segments/explore over a square region of half-width
    radius_km, using adjacent (NOT overlapping - the step equals the box
    size) box_km x box_km tiles. explore only returns ~10 segments per
    call, so smaller tiles cover more ground.

    Note that segments straddling a tile boundary can be missed entirely:
    explore returns segments intersecting the box, but caps at ~10, so a
    dense tile silently truncates.

    Each tile's result is cached to disk keyed by its bounding box, so
    overlapping search areas across runs reuse already-fetched tiles and
    only genuinely new tiles cost an API call."""
    _ensure_cache_dirs()
    headers = {"Authorization": f"Bearer {token}"}
    seen_ids = set()
    results = []

    dlat = km_to_deg_lat(radius_km)
    dlon = km_to_deg_lon(radius_km, lat)
    step_lat = km_to_deg_lat(box_km)
    step_lon = km_to_deg_lon(box_km, lat)

    lat_steps = np.arange(lat - dlat, lat + dlat, step_lat)
    lon_steps = np.arange(lon - dlon, lon + dlon, step_lon)

    total = len(lat_steps) * len(lon_steps)
    print(f"Scanning {total} tiles around ({lat}, {lon}), "
          f"radius {radius_km} km...")

    call_count = 0
    cache_hits = 0
    skipped_tiles = 0
    for i, la in enumerate(lat_steps):
        for lo in lon_steps:
            sw_lat, sw_lon = la, lo
            ne_lat, ne_lon = la + step_lat, lo + step_lon

            cache_path = EXPLORE_CACHE_DIR / _tile_cache_key(
                sw_lat, sw_lon, ne_lat, ne_lon)
            cached = _cache_read(cache_path)
            if cached is not None:
                cache_hits += 1
                for seg in cached:
                    if seg["id"] not in seen_ids:
                        seen_ids.add(seg["id"])
                        results.append(seg)
                continue

            # Longitudes must stay in [-180, 180]; a search centred near
            # the antimeridian otherwise sends bounds like "180.10" that
            # Strava rejects.
            sw_lon_n, ne_lon_n = _norm_lon(sw_lon), _norm_lon(ne_lon)
            if ne_lon_n <= sw_lon_n:
                skipped_tiles += 1  # tile wraps the dateline; not splittable
                continue
            bounds = f"{sw_lat},{sw_lon_n},{ne_lat},{ne_lon_n}"

            # Retry a rate-limited tile instead of abandoning it. The old
            # code slept 60s and then continued to the NEXT tile, so every
            # tile that landed inside a 429 window was lost for the whole
            # run - and never cached, so nothing in the output revealed the
            # hole in the search area.
            data = None
            for attempt in range(3):
                try:
                    resp = requests.get(
                        f"{STRAVA_API}/segments/explore",
                        headers=headers,
                        params={"bounds": bounds,
                                "activity_type": "running"},
                        timeout=30,
                    )
                except requests.RequestException as e:
                    print(f"  tile skipped (network error: {e})")
                    break
                call_count += 1
                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    print(f"  Rate limited by Strava. Waiting {wait}s "
                          f"(tile retry {attempt + 1}/3)...")
                    time.sleep(wait)
                    continue
                if resp.status_code in (401, 403):
                    sys.exit(f"Strava rejected the request (HTTP "
                             f"{resp.status_code}) - your token may have "
                             f"expired or lack scope. Re-run --authorize.")
                if resp.status_code != 200:
                    print(f"  tile skipped (HTTP {resp.status_code})")
                    break
                data = resp.json()
                break
            if data is None:
                skipped_tiles += 1
                continue

            tile_segs = data.get("segments", [])
            _cache_write(cache_path, tile_segs)
            for seg in tile_segs:
                if seg["id"] not in seen_ids:
                    seen_ids.add(seg["id"])
                    results.append(seg)

            # Strava allows 100 req/15min, 200 req/15min for some tiers.
            # Stay conservative.
            if call_count % 90 == 0:
                print("  Pausing to respect rate limits...")
                time.sleep(15 * 60)

    print(f"Found {len(results)} unique candidate segments "
          f"({call_count} API calls, {cache_hits} tiles from cache).")
    if skipped_tiles:
        # Surfaced because an unreported hole in the grid looks exactly
        # like "there are no good segments over there".
        print(f"  WARNING: {skipped_tiles} of {total} tiles could not be "
              f"fetched - those areas were NOT searched. Re-run to fill "
              f"them in (cached tiles cost no API calls).")
    return results


def get_segment_stream(token, segment_id):
    """Fetch a segment's distance/altitude/latlng streams, cached to disk
    by segment ID (streams are static, so a cached one is reused forever
    within the TTL). Returns (distance, altitude, latlng) arrays or None."""
    _ensure_cache_dirs()
    cache_path = STREAM_CACHE_DIR / f"{segment_id}.json"

    cached = _cache_read(cache_path)
    if cached is not None:
        if cached.get("_miss"):
            return None  # previously confirmed to have no usable stream
        latlng = np.array(cached["latlng"]) if cached.get("latlng") else None
        return (np.array(cached["distance"]),
                np.array(cached["altitude"]), latlng)

    headers = {"Authorization": f"Bearer {token}"}
    params = {"keys": "distance,altitude,latlng", "key_by_type": "true"}
    resp = requests.get(
        f"{STRAVA_API}/segments/{segment_id}/streams",
        headers=headers, params=params, timeout=30,
    )
    if resp.status_code == 429:
        print("  Rate limited fetching stream, waiting 60s...")
        time.sleep(60)
        resp = requests.get(
            f"{STRAVA_API}/segments/{segment_id}/streams",
            headers=headers, params=params, timeout=30,
        )
    if resp.status_code != 200:
        # Not cached - a transient failure must not become a permanent
        # "_miss". But it must also not be silent: a 429 here previously
        # made the candidate indistinguishable from one with no stream, so
        # rate-limiting could quietly drop the best match from the results.
        print(f"  warning: stream fetch for segment {segment_id} failed "
              f"(HTTP {resp.status_code}) - candidate NOT scored.")
        return None
    data = resp.json()
    if "distance" not in data or "altitude" not in data:
        _cache_write(cache_path, {"_miss": True})  # cache the genuine miss
        return None

    try:
        dist_data = data["distance"]["data"]
        alt_data = data["altitude"]["data"]
    except (KeyError, TypeError):
        # Malformed payload (key present but not the expected shape).
        _cache_write(cache_path, {"_miss": True})
        return None
    latlng_data = (data.get("latlng") or {}).get("data")
    _cache_write(cache_path, {
        "distance": dist_data,
        "altitude": alt_data,
        "latlng": latlng_data,
    })
    latlng = np.array(latlng_data) if latlng_data else None
    return (np.array(dist_data, dtype=float),
            np.array(alt_data, dtype=float),
            latlng)


def parse_segment_id(url_or_id):
    """Accept a Strava segment URL or a bare numeric ID and return the
    integer segment ID. Raises ValueError on anything unrecognizable."""
    s = str(url_or_id).strip()
    if s.isdigit():
        return int(s)
    # e.g. https://www.strava.com/segments/27983261  (optional query/frag)
    m = re.search(r"/segments/(\d+)", s)
    if m:
        return int(m.group(1))
    raise ValueError(f"Couldn't find a segment ID in: {url_or_id!r}")


def get_segment_meta(token, segment_id):
    """Fetch a segment's basic metadata (name, etc.) for labeling the GPX.
    Returns a dict or {} on failure. Not cached it's one call, rarely
    repeated."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(f"{STRAVA_API}/segments/{segment_id}",
                             headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return {}


def export_segment_gpx(token, segment_id, out_path, reverse=False,
                        name=None):
    """Write a segment's geometry (lat/lon + elevation) to a GPX file.

    reverse=True flips the point order so the track runs the opposite way
    (e.g. to run a matched climb as a descent, or vice versa). Returns the
    path written, or raises on failure."""
    stream = get_segment_stream(token, segment_id)
    if stream is None:
        raise RuntimeError(
            f"No geometry stream available for segment {segment_id} "
            f"(it may be private, hazardous-flagged, or missing latlng).")
    dist, elev, latlng = stream
    if latlng is None or len(latlng) == 0:
        raise RuntimeError(
            f"Segment {segment_id} has no lat/lon stream, can't build GPX.")

    lats = latlng[:, 0]
    lons = latlng[:, 1]
    elevs = elev
    if reverse:
        lats = lats[::-1]
        lons = lons[::-1]
        elevs = elevs[::-1]

    if not name:
        name = f"Strava Segment {segment_id}"
    if reverse:
        name = f"{name} (reversed)"

    # Build minimal GPX 1.1 by hand (no dependency needed for writing).
    def esc(t):
        return (str(t).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="find_similar_segments" '
        'xmlns="http://www.topografix.com/GPX/1/1">',
        '  <trk>',
        f'    <name>{esc(name)}</name>',
        f'    <link href="https://www.strava.com/segments/{segment_id}"/>',
        '    <type>running</type>',
        '    <trkseg>',
    ]
    n = min(len(lats), len(lons), len(elevs))
    for i in range(n):
        lines.append(
            f'      <trkpt lat="{lats[i]:.6f}" lon="{lons[i]:.6f}">'
            f'<ele>{elevs[i]:.1f}</ele></trkpt>')
    lines += ['    </trkseg>', '  </trk>', '</gpx>', '']

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

GRADE_BANDS = [
    (-100, 0, "<0%"),
    (0, 2, "0-2%"),
    (2, 4, "2-4%"),
    (4, 6, "4-6%"),
    (6, 8, "6-8%"),
    (8, 10, "8-10%"),
    (10, 100, "10%+"),
]


def grade_band_breakdown(grade_seq):
    """
    Given a per-bin grade sequence (each bin the same physical length),
    return a list of (label, fraction) for each signed grade band, i.e.
    'what share of the climb sits in each grade range'. Since bins are
    equal-length, counting bins == counting distance.
    """
    grades = np.asarray(grade_seq, dtype=float)
    total = len(grades)
    out = []
    for lo, hi, label in GRADE_BANDS:
        frac = np.mean((grades >= lo) & (grades < hi)) if total else 0.0
        out.append((label, frac))
    return out


def format_band_breakdown(grade_seq):
    """One-line human summary of the grade composition, biggest first,
    skipping empty bands."""
    bands = grade_band_breakdown(grade_seq)
    nonzero = [(label, f) for label, f in bands if f > 0]
    nonzero.sort(key=lambda x: -x[1])
    return ", ".join(f"{f*100:.0f}% {label}" for label, f in nonzero)


# --------------------------------------------------------------------------
# Road-access scoring (OpenStreetMap via Overpass API)
# --------------------------------------------------------------------------

GOOGLE_KEY_FILE = Path.home() / ".google_maps_api_key"
GOOGLE_KEY_FILE_ALT = Path.home() / ".google"


def get_google_api_key():
    """Check env var first, then key files, in that order."""
    env_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if env_key:
        return env_key.strip()
    for kf in (GOOGLE_KEY_FILE, GOOGLE_KEY_FILE_ALT):
        if kf.exists():
            key = kf.read_text().strip()
            if key:
                return key
    return None


def road_distance_m_google(lat, lon, api_key, debug=False):
    """
    Uses Google's Roads API (nearestRoads) to find the actual nearest
    mapped road to (lat, lon) and returns the real distance to it in
    meters. Unlike Overpass's radius-probe approach, this gives an exact
    distance in one call rather than 'found within 300m vs 1000m'.

    Returns None if no road is snapped (Roads API returns nothing) or
    the request fails.
    """
    try:
        resp = requests.get(
            "https://roads.googleapis.com/v1/nearestRoads",
            params={"points": f"{lat},{lon}", "key": api_key},
            timeout=8,
        )
        if resp.status_code != 200:
            if debug:
                print(f"    [access debug] Google Roads API HTTP "
                      f"{resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        if "error" in data:
            if debug:
                print(f"    [access debug] Google Roads API error: "
                      f"{data['error'].get('message', data['error'])}")
            return None
        points = data.get("snappedPoints", [])
        if not points:
            return None
        snapped = points[0]["location"]
        return haversine_m(lat, lon, snapped["latitude"], snapped["longitude"])
    except requests.RequestException as e:
        if debug:
            print(f"    [access debug] Google Roads API "
                  f"{type(e).__name__}: {e}")
        return None


_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
_OVERPASS_HEADERS = {
    # Public Overpass instances commonly reject/deprioritize requests
    # with no descriptive User-Agent the default python-requests UA
    # gets silently dropped or throttled on some mirrors.
    "User-Agent": "find_similar_segments/1.0 (personal training-route "
                   "matching script; contact: local use only)"
}
_OVERPASS_UNREACHABLE = False  # set True after a totally failed probe,
                                # so we stop hammering unreachable hosts
                                # for every remaining candidate

# Memoizes road lookups by rounded (lat, lon) so repeated probes of nearby
# window starts cost nothing. Values are metres, or None for "checked, no
# road found".
_ROAD_DIST_CACHE = {}


def road_distance_m_overpass(lat, lon, radii=(300, 1000, 2500), debug=False):
    """
    Approximate distance (meters) from (lat, lon) to the nearest OSM
    road/track, by querying Overpass with progressively larger radii and
    returning the first that finds a hit. This is an access proxy, not a
    precise perpendicular distance good enough to distinguish
    'trailhead is basically at a road' from 'this is deep backcountry'.

    If every endpoint fails at the connection level (refused/timeout,
    not an HTTP error) on the very first candidate, Overpass is almost
    certainly unreachable from this network entirely e.g. a corporate
    firewall only allowlisting specific domains. Rather than repeat that
    failure (with its timeouts) for every remaining candidate, this
    disables further Overpass lookups for the rest of the run after one
    fully-failed probe and just returns None (neutral/unknown) from then
    on. Use --no-access-check to skip this path entirely up front.
    """
    global _OVERPASS_UNREACHABLE
    if _OVERPASS_UNREACHABLE:
        return None

    key = (round(lat, 4), round(lon, 4))
    if key in _ROAD_DIST_CACHE:
        return _ROAD_DIST_CACHE[key]

    last_error = None
    any_connection_succeeded = False

    for r in radii:
        query = (
            f"[out:json][timeout:8];"
            f"way(around:{r},{lat},{lon})[highway];"
            f"out ids 1;"
        )
        for endpoint in _OVERPASS_ENDPOINTS:
            try:
                resp = requests.post(
                    endpoint, data={"data": query},
                    headers=_OVERPASS_HEADERS, timeout=8,
                )
                any_connection_succeeded = True
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("elements"):
                        _ROAD_DIST_CACHE[key] = r
                        return r
                    last_error = None
                    break
                else:
                    last_error = f"HTTP {resp.status_code} from {endpoint}: " \
                                 f"{resp.text[:150]}"
                    if debug:
                        print(f"    [access debug] {last_error}")
                    continue
            except requests.RequestException as e:
                last_error = f"{type(e).__name__} from {endpoint}: {e}"
                if debug:
                    print(f"    [access debug] {last_error}")
                continue
        time.sleep(0.5)

    if not any_connection_succeeded:
        _OVERPASS_UNREACHABLE = True
        print("  [access] Overpass appears unreachable from this network "
              "(every mirror failed to even connect) disabling "
              "road-access checks for the rest of this run. This is "
              "likely a firewall/proxy blocking unlisted domains. Pass "
              "--no-access-check next time to skip this up front.")
    elif debug and last_error:
        print(f"    [access debug] all mirrors exhausted for ({lat:.4f},"
              f"{lon:.4f}), last error: {last_error}")

    _ROAD_DIST_CACHE[key] = None
    return None


# Sentinel: the access lookup was attempted but failed (network/API
# error), as opposed to succeeding with a real distance. Distinct from
# None, which we reserve for "checked, but genuinely no road found".
ACCESS_UNCHECKED = "unchecked"


def road_distance_m(lat, lon, google_api_key=None, debug=False):
    """
    Dispatcher: uses Google's Roads API if a key is available (more
    reliable, exact distance in one call), otherwise falls back to the
    free Overpass/OSM radius-probe approach.

    Returns one of:
      - a float distance in meters (road found),
      - None (checked, but genuinely no road within range Overpass
        path only; Google's nearestRoads has no range limit so this is
        rare there),
      - ACCESS_UNCHECKED (the lookup itself failed and we don't know).
    """
    if google_api_key:
        dist = road_distance_m_google(lat, lon, google_api_key, debug=debug)
        if dist is not None:
            return round(dist)
        # Google's nearestRoads has no distance cap, so a null result
        # almost always means the CALL failed rather than "no road
        # exists anywhere" mark it unchecked rather than penalizing it
        # as if we'd confirmed it's remote.
        return ACCESS_UNCHECKED
    result = road_distance_m_overpass(lat, lon, debug=debug)
    # Overpass path already returns None for "checked, none within
    # 2.5km". But if the whole backend got disabled mid-run, it also
    # returns None flag that case as unchecked instead.
    if result is None and _OVERPASS_UNREACHABLE:
        return ACCESS_UNCHECKED
    return result


def access_penalty(road_dist_m_value, near_m=400, far_m=1200):
    """
    Converts a road-proximity reading into an additive penalty on the
    combined score (same scale as DTW).

    - At or within near_m: basically accessible, no penalty.
    - Between near_m and far_m: linearly ramping small penalty.
    - At or beyond far_m: heavily penalized as effectively unreachable.
    - Confirmed nothing found at all: max penalty (remote).
    - Lookup failed (ACCESS_UNCHECKED): neutral 0.0, so missing data
      neither rewards nor punishes a segment.
    """
    if road_dist_m_value == ACCESS_UNCHECKED:
        return 0.0  # neutral don't distort ranking on missing data
    if road_dist_m_value is None:
        return 3.0  # checked: genuinely no road found treat as remote
    d = road_dist_m_value
    if d <= near_m:
        return 0.0
    if d >= far_m:
        return 2.5  # effectively unreachable for training
    # Linear ramp from 0.0 at near_m up to 2.5 at far_m
    frac = (d - near_m) / (far_m - near_m)
    return round(2.5 * frac, 2)


# --------------------------------------------------------------------------
# Windowed subsequence matching (direction-aware, no hard descent filter)
# --------------------------------------------------------------------------

def _latlng_at(seg_dist, seg_latlng, dist_m):
    """Lat/lon at a given distance along a segment, or None."""
    if seg_latlng is None or len(seg_latlng) == 0:
        return None
    lat = float(np.interp(dist_m, seg_dist, seg_latlng[:, 0]))
    lon = float(np.interp(dist_m, seg_dist, seg_latlng[:, 1]))
    return lat, lon


def window_access(seg_dist, seg_latlng, start_m, end_m, google_api_key,
                   debug=False):
    """Road distance for a window, sampled at its start, middle and end.

    Sampling only the start point reads a segment that begins at a car
    park and climbs into the backcountry as fully accessible. Taking the
    worst of three points along the window costs three lookups instead of
    one and describes reachability of the whole window rather than of its
    first metre.

    Returns a distance in metres, None (checked, no road found), or
    ACCESS_UNCHECKED if every lookup failed.
    """
    pts = [p for p in (_latlng_at(seg_dist, seg_latlng, start_m),
                       _latlng_at(seg_dist, seg_latlng,
                                  (start_m + end_m) / 2.0),
                       _latlng_at(seg_dist, seg_latlng, end_m))
           if p is not None]
    if not pts:
        return ACCESS_UNCHECKED
    seen = []
    for lat, lon in pts:
        r = road_distance_m(lat, lon, google_api_key=google_api_key,
                            debug=debug)
        seen.append(r)
    reals = [r for r in seen if isinstance(r, (int, float))]
    if reals:
        return max(reals)
    if any(r is None for r in seen):
        return None
    return ACCESS_UNCHECKED


def impute_access_penalties(rows, key="road_dist"):
    """Replace unchecked access readings with the median of the checked
    ones, and report how many were substituted.

    Scoring an unchecked lookup as 0.0 gave it the same penalty as a
    segment starting at the trailhead, so a segment whose lookup happened
    to fail was rewarded with the best possible access score. With a
    Google key any failed call returns ACCESS_UNCHECKED, so intermittent
    failures quietly promoted segments over genuinely remote ones.
    Substituting the median of what was actually measured is neutral in
    the sense that matters: it neither rewards nor punishes relative to a
    typical candidate.

    Returns (n_imputed, imputed_value).
    """
    known = [r["penalty"] for r in rows if r[key] != ACCESS_UNCHECKED]
    if not known:
        return 0, 0.0
    med = float(np.median(known))
    n = 0
    for r in rows:
        if r[key] == ACCESS_UNCHECKED:
            r["penalty"] = med
            n += 1
    return n, med


def describe_significance(score, null):
    """One line placing a score against the null distribution."""
    if null is None or len(null) == 0:
        return "no null model available"
    better_than = float((null > score).mean())
    if better_than >= 0.999:
        verdict = "far better than random terrain"
    elif better_than >= 0.99:
        verdict = "clearly better than random terrain"
    elif better_than >= 0.95:
        verdict = "better than random terrain"
    elif better_than >= 0.80:
        verdict = "WEAK - only modestly better than random"
    else:
        verdict = "NOT DISTINGUISHABLE FROM RANDOM TERRAIN"
    return f"better than {better_than * 100:.1f}% of random windows, {verdict}"


def main():
    # Defaults come FROM MatchConfig rather than being repeated here.
    # Duplicating them let two of them drift: --grade-res-m stayed at 120
    # and --weight-gain at 4.0 after the library moved to 70 and 2.0, so
    # every command line run silently used superseded parameters while the
    # library, the tests and the benchmark all used the current ones.
    _D = MatchConfig()

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authorize", action="store_true",
                     help="Run one-time OAuth setup")
    ap.add_argument("--client-id", help="Strava API Client ID")
    ap.add_argument("--client-secret", help="Strava API Client Secret")
    ap.add_argument("--gpx", help="Path to target GPX file")
    ap.add_argument("--lat", type=float, help="Search center latitude")
    ap.add_argument("--lon", type=float, help="Search center longitude")
    ap.add_argument("--radius-km", type=float, default=15.0,
                     help="Half-width of the square search area in km "
                          "(default 15, so a 30x30 km box)")

    g = ap.add_argument_group("matching")
    g.add_argument("--grade-res-m", type=float, default=_D.res_m,
                    help="Physical resolution (m) at which grade is "
                         "estimated. To tell two profiles apart at a "
                         "pitch length p you need this at or below p, so "
                         "a coarse value silently merges genuinely "
                         "different terrain: at 120 m a staircase of 60 m "
                         "pitches scores 0.19 against a uniform climb of "
                         "the same length, gain and loss. Going finer "
                         "costs tolerance to elevation noise and "
                         "rounding. See PARAMETERS.md")
    g.add_argument("--min-window-frac", type=float, default=_D.min_ratio,
                    help="Shortest matching window, as a fraction of the "
                         "target length (default 0.75)")
    g.add_argument("--max-window-frac", type=float, default=_D.max_ratio,
                    help="Longest matching window, as a fraction of the "
                         "target length (default 1.15)")
    g.add_argument("--length-steps", type=int, default=_D.length_steps,
                    help="How many window lengths to try between the min "
                         "and max fractions (default 7). 1.0x is always "
                         "included regardless. The winner is then refined "
                         "by local search, so this grid only has to be "
                         "good enough to find the right basin")
    g.add_argument("--start-step-frac", type=float, default=_D.stride_frac,
                    help="Window offset step, as a fraction of the target "
                         "length (default 0.02). As with --length-steps, "
                         "the winning offset is refined afterwards")
    g.add_argument("--max-shift-frac", type=float, default=_D.max_shift_frac,
                    help="Alignment tolerance: how far a feature may sit "
                         "from where the target has it and still match, "
                         "as a fraction of target length (default 0.03). "
                         "Wider bands let the shape term ignore how long "
                         "each section physically is")
    g.add_argument("--max-segment-mult", type=float, default=3.0,
                    help="Skip candidate segments longer than this "
                         "multiple of the target distance (default 3.0)")
    g.add_argument("--matches-per-segment", type=int, default=1,
                    help="Report up to this many non-overlapping windows "
                         "from each segment (default 1). Raise it to find "
                         "a pattern that repeats within one long segment")

    w = ap.add_argument_group("scoring weights")
    w.add_argument("--weight-shape", type=float, default=_D.w_shape,
                    help="Weight on the ordered shape term (default 1.0)")
    w.add_argument("--weight-distribution", type=float, default=_D.w_dist,
                    help="Weight on grade composition regardless of order "
                         "(default 0.6, kept below shape so composition "
                         "is not double counted)")
    w.add_argument("--weight-gain", type=float, default=_D.w_gain,
                    help="Weight on vertical deviation. The term is "
                         "bounded in [0, 1] and symmetric, so this is the "
                         "full cost of a total vertical mismatch")
    w.add_argument("--weight-length", type=float, default=_D.w_len,
                    help="Weight on window length deviation (default 2.0)")

    a = ap.add_argument_group("access")
    a.add_argument("--no-access-check", action="store_true",
                    help="Skip road-proximity lookups entirely")
    a.add_argument("--access-near-m", type=float, default=400,
                    help="Within this many m of a road counts as "
                         "accessible, no penalty (default 400)")
    a.add_argument("--access-far-m", type=float, default=1200,
                    help="Beyond this many m a match is heavily penalized "
                         "(default 1200)")
    a.add_argument("--max-road-dist", type=float, default=None,
                    help="Hard gate: discard any match further than this "
                         "many m from a road, instead of blending "
                         "reachability into the score. Use when access is "
                         "a constraint rather than a preference")
    a.add_argument("--google-api-key", default=None,
                    help="Google Maps Platform API key with the Roads API "
                         "enabled. Falls back to GOOGLE_MAPS_API_KEY, "
                         "then ~/.google_maps_api_key, then free "
                         "OpenStreetMap/Overpass lookups")
    a.add_argument("--debug-access", action="store_true",
                    help="Print the HTTP status or error behind each "
                         "failed road-proximity lookup")

    o = ap.add_argument_group("output and cache")
    o.add_argument("--top", type=int, default=10,
                    help="Number of top matches to show")
    o.add_argument("--null-samples", type=int, default=240,
                    help="Random windows scored to build the reference "
                         "distribution used for the match-quality line "
                         "(default 240, 0 to disable)")
    o.add_argument("--not-reversible", action="store_true",
                    help="Only match candidates in their as-recorded "
                         "direction")
    o.add_argument("--export", metavar="SEGMENT_URL_OR_ID", default=None,
                    help="Export mode: write a segment's geometry to GPX "
                         "instead of searching")
    o.add_argument("--output-path", default=None,
                    help="Where to write the exported GPX. Only used with "
                         "--export")
    o.add_argument("--reverse", action="store_true",
                    help="With --export, flip the segment's direction")
    o.add_argument("--refresh", action="store_true",
                    help="Ignore cached Strava data and re-fetch")
    o.add_argument("--clear-cache", action="store_true",
                    help="Delete the on-disk Strava cache and exit")
    args = ap.parse_args()

    if args.clear_cache:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print(f"Cleared cache at {CACHE_DIR}")
        else:
            print("No cache to clear.")
        return

    global _FORCE_REFRESH
    _FORCE_REFRESH = args.refresh

    if args.authorize:
        if not args.client_id or not args.client_secret:
            sys.exit("--authorize requires --client-id and --client-secret")
        do_authorize(args.client_id, args.client_secret)
        return

    if args.export:
        token = get_access_token()
        try:
            seg_id = parse_segment_id(args.export)
        except ValueError as e:
            sys.exit(str(e))
        meta = get_segment_meta(token, seg_id)
        seg_name = meta.get("name")

        def slugify(s):
            s = re.sub(r"[^\w\s-]", "", str(s)).strip().lower()
            return re.sub(r"[\s]+", "_", s) or f"segment_{seg_id}"

        base_name = slugify(seg_name) if seg_name else f"segment_{seg_id}"
        if args.reverse:
            base_name += "_reversed"
        default_filename = f"{base_name}.gpx"

        out = args.output_path
        if not out:
            out_path = os.path.join(os.getcwd(), default_filename)
        elif os.path.isdir(out) or out.endswith(os.sep):
            out_path = os.path.join(out, default_filename)
        elif out.lower().endswith(".gpx"):
            out_path = out
        else:
            out_path = os.path.join(out, default_filename)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)),
                     exist_ok=True)
        try:
            written = export_segment_gpx(token, seg_id, out_path,
                                          reverse=args.reverse,
                                          name=seg_name)
        except (RuntimeError, OSError) as e:
            sys.exit(f"Export failed: {e}")
        label = seg_name or f"segment {seg_id}"
        note = " (reversed)" if args.reverse else ""
        print(f"Exported {label}{note} to:\n  {written}")
        return

    if not (args.gpx and args.lat is not None and args.lon is not None):
        sys.exit("Need --gpx, --lat, --lon (or use --authorize first)")

    token = get_access_token()
    google_api_key = args.google_api_key or get_google_api_key()
    if args.no_access_check:
        pass
    elif google_api_key:
        print("Using Google Roads API for road-proximity checks.\n")
    else:
        print("No Google API key found, falling back to free "
              "OpenStreetMap/Overpass lookups (less reliable). Set "
              "--google-api-key, GOOGLE_MAPS_API_KEY, or "
              "~/.google_maps_api_key to use Google instead.\n")

    cfg = MatchConfig(
        res_m=args.grade_res_m,
        min_ratio=args.min_window_frac,
        max_ratio=args.max_window_frac,
        length_steps=args.length_steps,
        stride_frac=args.start_step_frac,
        max_shift_frac=args.max_shift_frac,
        w_shape=args.weight_shape,
        w_dist=args.weight_distribution,
        w_gain=args.weight_gain,
        w_len=args.weight_length,
        top_k=args.matches_per_segment,
        pool_size=max(16, 8 * args.matches_per_segment),
    )

    cum_dist, elevs = load_gpx_profile(args.gpx)
    target = prepare_target(cum_dist, elevs, cfg)
    if target is None:
        sys.exit(f"{args.gpx} is too short to represent at "
                 f"{args.grade_res_m:.0f} m resolution.")
    target_dist_mi = target.length_m / 1609.34

    print(f"\nTarget: {args.gpx}")
    print(f"  Distance: {target_dist_mi:.2f} mi")
    print(f"  Gain: {target.gain_m * 3.28084:.0f} ft | "
          f"Loss: {target.loss_m * 3.28084:.0f} ft "
          f"(measured every {cfg.vert_resample_m:.0f} m)")
    print(f"  Net grade: "
          f"{(elevs[-1] - elevs[0]) / cum_dist[-1] * 100:.1f}%")
    print(f"  Grade profile: {len(target.seq)} samples at "
          f"{args.grade_res_m:.0f} m resolution")
    print(f"  Grade composition: {format_band_breakdown(target.seq)}")
    q = detect_quantization(elevs)
    if q >= 2.0:
        print(f"  WARNING: target elevation is quantized to {q:.1f} m steps. "
              f"Rounding that coarse manufactures alternating flat and "
              f"steep samples at roughly the step spacing, which a "
              f"{args.grade_res_m:.0f} m representation partly resolves as "
              f"real terrain. Consider --grade-res-m {max(120.0, 3 * q * 10):.0f} "
              f"or finer source data.")
    if args.not_reversible:
        print("  (matching as-recorded direction only)")
    print()

    min_seg_mi = target_dist_mi * args.min_window_frac
    max_seg_mi = target_dist_mi * args.max_segment_mult
    print(f"Pre-filtering to segments {min_seg_mi:.1f}-{max_seg_mi:.1f} mi "
          f"long. Descents are scored in reverse, not thrown out.\n")

    candidates = explore_segments(token, args.lat, args.lon, args.radius_km)
    dist_filtered = [s for s in candidates
                     if min_seg_mi <= s["distance"] / 1609.34 <= max_seg_mi]
    print(f"{len(dist_filtered)} candidates pass the length filter. "
          f"Fetching profiles and searching each...\n")

    # Pass 1: fetch, build each profile once, and score terrain. Access
    # lookups are deferred so the null model can be built from the same
    # profiles without paying for network calls on segments that will not
    # survive ranking.
    rows, profiles = [], []
    for i, seg in enumerate(dist_filtered):
        stream = get_segment_stream(token, seg["id"])
        if stream is None:
            continue
        seg_dist, seg_elev, seg_latlng = stream
        try:
            prof = build_profile(seg_dist, seg_elev, cfg.res_m,
                                  cfg.oversample, cfg.vert_resample_m)
        except ValueError as e:
            print(f"  [{i+1}/{len(dist_filtered)}] {seg['name']}: "
                  f"unusable profile ({e})")
            continue
        if prof is None:
            continue
        profiles.append(prof)

        matches = match_segment(seg_dist, seg_elev, target, cfg,
                                 profile=prof)
        if not matches:
            print(f"  [{i+1}/{len(dist_filtered)}] {seg['name']}: "
                  f"too short to contain a qualifying window")
            continue
        for m in matches:
            rows.append({"seg": seg, "m": m, "seg_dist": seg_dist,
                         "seg_latlng": seg_latlng, "road_dist": None,
                         "penalty": 0.0})
        best = matches[0]
        print(f"  [{i+1}/{len(dist_filtered)}] {seg['name']}: "
              f"terrain {best.score:.2f} (shape {best.shape:.2f}/comp "
              f"{best.dist:.2f}/vert {best.gain_dev * 100:.0f}%/len "
              f"{best.len_dev * 100:.0f}%) {best.direction}")
        time.sleep(0.3)

    if not rows:
        print("\nNo scoreable candidates found.")
        return

    null = None
    if args.null_samples > 0:
        null = null_scores(profiles, target, cfg, n=args.null_samples)
        if len(null):
            print(f"\nNull model: {len(null)} random windows from the same "
                  f"candidates score between {null.min():.2f} and "
                  f"{null.max():.2f} (median {np.median(null):.2f}).")

    # Pass 2: access lookups, only for the best terrain matches, since
    # each one now costs three network calls.
    rows.sort(key=lambda r: r["m"].score)
    checked = rows[:max(args.top * 3, args.top)]
    if not args.no_access_check:
        print(f"\nChecking road access for the top {len(checked)} "
              f"terrain matches (3 points each)...")
        for r in checked:
            r["road_dist"] = window_access(
                r["seg_dist"], r["seg_latlng"], r["m"].start_m,
                r["m"].end_m, google_api_key, debug=args.debug_access)
            r["penalty"] = access_penalty(r["road_dist"],
                                           near_m=args.access_near_m,
                                           far_m=args.access_far_m)
        n_imputed, med = impute_access_penalties(checked)
        if n_imputed:
            print(f"  {n_imputed} of {len(checked)} lookups failed; using "
                  f"the median measured penalty ({med:.2f}) for those "
                  f"rather than treating them as roadside.")
        if args.max_road_dist is not None:
            before = len(checked)
            checked = [r for r in checked
                       if not isinstance(r["road_dist"], (int, float))
                       or r["road_dist"] <= args.max_road_dist]
            print(f"  --max-road-dist {args.max_road_dist:.0f} m gate: "
                  f"{before - len(checked)} of {before} matches discarded.")

    for r in checked:
        r["final"] = r["m"].score + r["penalty"]
    checked.sort(key=lambda r: r["final"])

    print(f"\n{'=' * 72}")
    print(f"TOP {min(args.top, len(checked))} MATCHES "
          f"(lower combined score = closer terrain, more reachable)")
    print(f"{'=' * 72}\n")

    for rank, r in enumerate(checked[:args.top], 1):
        m, seg = r["m"], r["seg"]
        start_mi = m.start_m / 1609.34
        end_mi = m.end_m / 1609.34
        gain_ft, loss_ft = m.gain_m * 3.28084, m.loss_m * 3.28084
        seg_total_mi = seg["distance"] / 1609.34
        url = f"https://www.strava.com/segments/{seg['id']}"
        if isinstance(r["road_dist"], (int, float)):
            access_str = f"~{r['road_dist']:.0f} m from a road"
        elif r["road_dist"] is None and not args.no_access_check:
            access_str = "no road found nearby"
        else:
            access_str = "access not confirmed"

        print(f"{rank}. {seg['name']}  "
              f"(segment is {seg_total_mi:.1f} mi total)")
        leg = "CLIMB" if gain_ft >= loss_ft else "DESCENT"
        if m.direction == "reverse":
            print(f"   Run the segment BACKWARD ({leg}): start at mile "
                  f"{end_mi:.2f}, finish at mile {start_mi:.2f}")
        else:
            print(f"   Run the segment as-recorded ({leg}): mile "
                  f"{start_mi:.2f} to {end_mi:.2f} "
                  f"({(end_mi - start_mi):.2f} mi)")
        print(f"   Gain: {gain_ft:.0f} ft | Loss: {loss_ft:.0f} ft | "
              f"length {m.length_ratio * 100:.0f}% of target "
              f"(target {target.gain_m * 3.28084:.0f} ft up / "
              f"{target.loss_m * 3.28084:.0f} ft down)")
        print(f"   Grade composition: {format_band_breakdown(m.grade_seq)}")
        print(f"   Scores: shape {m.shape:.2f} | composition {m.dist:.2f} "
              f"| vertical {m.gain_dev * 100:.0f}% | length "
              f"{m.len_dev * 100:.0f}% | access {r['penalty']:.2f} "
              f"| COMBINED {r['final']:.2f}")
        print(f"   Match quality: {describe_significance(m.score, null)}")
        print(f"   Access: {access_str}")
        print(f"   {url}\n")


if __name__ == "__main__":
    main()
