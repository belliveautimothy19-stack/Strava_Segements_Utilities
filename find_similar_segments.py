#!/usr/bin/env python3
"""
find_similar_segments.py

Finds Strava running segments near a given location whose ELEVATION/GRADE
PROFILE most closely matches a target GPX file — not just similar average
grade or distance, but similar overall shape (punchy start, rolling middle,
steep pinch, flat finish, etc.), using dynamic time warping (DTW) on
resampled grade sequences.

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

Output: ranked list of candidate segments with name, distance, elevation
gain, average grade, DTW similarity score (lower = more similar shape),
and a direct Strava URL.
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
        "to load. That's expected — copy the 'code' parameter out of the "
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
    print(f"Saved tokens to {TOKEN_FILE}. You're set — run the script "
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
    return np.array(cum_dist), np.array(elevs)


def resample_grade_profile(cum_dist, elevs, bin_size_m):
    """
    Bin the elevation profile into FIXED REAL-DISTANCE chunks (e.g. every
    0.25 mi) and return the grade (%) within each chunk, in order.

    This is deliberately distance-based rather than point-count-based:
    resampling to a fixed point count means longer windows get coarser
    per-point resolution and shorter windows get finer resolution, which
    is an artifact of window length, not a real property of the climb.
    Binning by fixed physical distance keeps resolution consistent
    regardless of window length, so 'first quarter mile at 8%, next
    quarter mile at 2%' means the same thing everywhere it's compared \u2014
    sequence AND composition are preserved, not just the average grade.
    """
    total = cum_dist[-1]
    if total <= 0:
        return None
    n_bins = max(2, round(total / bin_size_m))
    edges = np.linspace(0, total, n_bins + 1)
    binned_elev = np.interp(edges, cum_dist, elevs)
    bin_dist = np.diff(edges)
    bin_dist[bin_dist == 0] = 1e-6
    grades = np.diff(binned_elev) / bin_dist * 100
    return grades


def dtw_distance(seq_a, seq_b):
    """Simple O(n*m) DTW on 1D sequences (grade %), no external deps."""
    n, m = len(seq_a), len(seq_b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(seq_a[i - 1] - seq_b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1],
                                  cost[i - 1, j - 1])
    return cost[n, m] / max(n, m)  # normalize by path length


# --------------------------------------------------------------------------
# Strava API
# --------------------------------------------------------------------------

def km_to_deg_lat(km):
    return km / 111.0


def km_to_deg_lon(km, at_lat):
    return km / (111.0 * math.cos(math.radians(at_lat)))


def explore_segments(token, lat, lon, radius_km, box_km=4.0):
    """Grid-search segments/explore over a square region of radius_km,
    using overlapping box_km x box_km tiles (explore only returns ~10
    segments per call, so smaller tiles cover more ground).

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

            bounds = f"{sw_lat},{sw_lon},{ne_lat},{ne_lon}"
            try:
                resp = requests.get(
                    f"{STRAVA_API}/segments/explore",
                    headers=headers,
                    params={"bounds": bounds, "activity_type": "running"},
                    timeout=30,
                )
            except requests.RequestException as e:
                print(f"  tile skipped (network error: {e})")
                continue

            call_count += 1
            if resp.status_code == 429:
                print("  Rate limited by Strava. Waiting 60s...")
                time.sleep(60)
                continue
            resp.raise_for_status()
            data = resp.json()
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
        return None  # don't cache transient failures
    data = resp.json()
    if "distance" not in data or "altitude" not in data:
        _cache_write(cache_path, {"_miss": True})  # cache the genuine miss
        return None

    latlng_data = data["latlng"]["data"] if "latlng" in data else None
    _cache_write(cache_path, {
        "distance": data["distance"]["data"],
        "altitude": data["altitude"]["data"],
        "latlng": latlng_data,
    })
    latlng = np.array(latlng_data) if latlng_data else None
    return (np.array(data["distance"]["data"]),
            np.array(data["altitude"]["data"]),
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
    Returns a dict or {} on failure. Not cached \u2014 it's one call, rarely
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

def total_gain_loss(elevs):
    """Total ascent and descent (meters) from an elevation array."""
    diffs = np.diff(elevs)
    gain = np.sum(diffs[diffs > 0])
    loss = -np.sum(diffs[diffs < 0])
    return gain, loss


def wasserstein_1d(a, b, n=100):
    """
    1D Earth Mover's (Wasserstein-1) distance between two sets of grade
    values, via quantile / inverse-CDF matching. No scipy dependency.

    This scores how similar two climbs' grade COMPOSITIONS are while
    ignoring the ORDER the grades occur in \u2014 the complement to the DTW
    shape score, which cares about order. It's built on distributions,
    so it correctly treats 7%-vs-8% as a near-miss and 2%-vs-10% as a
    big miss, unlike a naive per-band difference that would call both
    equally wrong.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    qs = np.linspace(0, 1, n)
    qa = np.interp(qs, np.linspace(0, 1, len(a)), a)
    qb = np.interp(qs, np.linspace(0, 1, len(b)), b)
    return float(np.mean(np.abs(qa - qb)))


# Signed grade bands (%) for the human-readable composition breakdown.
# These are for DISPLAY ONLY \u2014 the actual distribution scoring uses EMD
# above, which needs no band edges. Extends below zero so recovery/
# descent sections land in their own bands rather than being ignored.
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
    # with no descriptive User-Agent \u2014 the default python-requests UA
    # gets silently dropped or throttled on some mirrors.
    "User-Agent": "find_similar_segments/1.0 (personal training-route "
                   "matching script; contact: local use only)"
}
_OVERPASS_UNREACHABLE = False  # set True after a totally failed probe,
                                # so we stop hammering unreachable hosts
                                # for every remaining candidate


def road_distance_m_overpass(lat, lon, radii=(300, 1000, 2500), debug=False):
    """
    Approximate distance (meters) from (lat, lon) to the nearest OSM
    road/track, by querying Overpass with progressively larger radii and
    returning the first that finds a hit. This is an access proxy, not a
    precise perpendicular distance \u2014 good enough to distinguish
    'trailhead is basically at a road' from 'this is deep backcountry'.

    If every endpoint fails at the connection level (refused/timeout,
    not an HTTP error) on the very first candidate, Overpass is almost
    certainly unreachable from this network entirely \u2014 e.g. a corporate
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
              "(every mirror failed to even connect) \u2014 disabling "
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
      - None (checked, but genuinely no road within range \u2014 Overpass
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
        # exists anywhere" \u2014 mark it unchecked rather than penalizing it
        # as if we'd confirmed it's remote.
        return ACCESS_UNCHECKED
    result = road_distance_m_overpass(lat, lon, debug=debug)
    # Overpass path already returns None for "checked, none within
    # 2.5km". But if the whole backend got disabled mid-run, it also
    # returns None \u2014 flag that case as unchecked instead.
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
        return 0.0  # neutral \u2014 don't distort ranking on missing data
    if road_dist_m_value is None:
        return 3.0  # checked: genuinely no road found \u2014 treat as remote
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

def find_best_window(seg_dist, seg_elev, seg_latlng, target_grade_seq,
                      target_dist_m, bin_size_m, target_gain_m,
                      min_frac=0.75, max_frac=1.15, length_steps=7,
                      start_step_frac=0.02, w_shape=1.0, w_dist=0.6,
                      w_gain=4.0, reversible=True):
    """
    Slide windows of varying length across a candidate segment's profile
    and score each against the target.

    By default (reversible=True) each window is scored in BOTH physical
    running directions \u2014 as-recorded and reversed \u2014 and the better fit
    wins, so a segment is never missed just because it was recorded in
    the opposite direction; the result reports which way to run it. Set
    reversible=False to score only the as-recorded direction.

    The target's grade sequence is used as given by the caller.

    Combined score blends three terrain signals:
      shape (DTW, w_shape) \u2014 grade composition AND sequence
      distribution (EMD, w_dist) \u2014 grade composition without sequence
      gain (relative deviation, w_gain) \u2014 total vertical magnitude
    combined = w_shape*dtw + w_dist*emd + w_gain*gain_deviation
    (access proximity added separately by the caller.)

    Returns (score, win_start_m, win_end_m, gain_ft, loss_ft, direction,
    start_latlng, shape_score, dist_score, gain_dev, win_grade_seq), or
    None if the segment is too short. gain_ft/loss_ft are in the chosen
    running direction; direction is 'forward' or 'reverse' (the
    candidate's recorded orientation).
    """
    total_len = seg_dist[-1]
    if total_len < target_dist_m * min_frac:
        return None

    # target_gain_m is always the magnitude of the target's climb; when
    # the caller orients the target downhill, the window's descent
    # magnitude (its loss) is what should match it.
    best = None

    for length_frac in np.linspace(min_frac, max_frac, length_steps):
        win_len = target_dist_m * length_frac
        if win_len > total_len:
            continue
        step = max(target_dist_m * start_step_frac, 10)
        starts = np.arange(0, total_len - win_len + 1e-9, step)
        if len(starts) == 0:
            starts = np.array([0.0])

        for start in starts:
            end = start + win_len
            interp_n = max(len(target_grade_seq) * 4, 100)
            sub_d = np.linspace(start, end, interp_n)
            sub_elev = np.interp(sub_d, seg_dist, seg_elev)
            fwd_grade_seq = resample_grade_profile(sub_d - start, sub_elev,
                                                     bin_size_m)
            if fwd_grade_seq is None:
                continue
            rev_grade_seq = -np.flip(fwd_grade_seq)

            raw_gain_m, raw_loss_m = total_gain_loss(sub_elev)

            # "forward" = run the window as-recorded; "reverse" = run it
            # backward (gain/loss swap). Reversible tries both and keeps
            # the better; otherwise only the as-recorded direction.
            fwd_opt = ("forward", fwd_grade_seq, raw_gain_m, raw_loss_m)
            rev_opt = ("reverse", rev_grade_seq, raw_loss_m, raw_gain_m)
            options = [fwd_opt, rev_opt] if reversible else [fwd_opt]

            for direction, grade_seq, gain_m, loss_m in options:
                shape_score = dtw_distance(target_grade_seq, grade_seq)
                dist_score = wasserstein_1d(target_grade_seq, grade_seq)

                # Gain term: relative deviation of the window's dominant
                # vertical travel from the target's magnitude.
                window_vert_m = max(gain_m, loss_m)
                if target_gain_m > 0:
                    gain_dev = abs(window_vert_m - target_gain_m) / target_gain_m
                else:
                    gain_dev = 0.0

                score = (w_shape * shape_score
                         + w_dist * dist_score
                         + w_gain * gain_dev)

                if best is None or score < best[0]:
                    start_dist = end if direction == "reverse" else start
                    start_latlng = None
                    if seg_latlng is not None:
                        lat = np.interp(start_dist, seg_dist, seg_latlng[:, 0])
                        lon = np.interp(start_dist, seg_dist, seg_latlng[:, 1])
                        start_latlng = (lat, lon)
                    best = (score, start, end, gain_m * 3.28084,
                            loss_m * 3.28084, direction, start_latlng,
                            shape_score, dist_score, gain_dev,
                            np.asarray(grade_seq))

    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authorize", action="store_true",
                     help="Run one-time OAuth setup")
    ap.add_argument("--client-id", help="Strava API Client ID")
    ap.add_argument("--client-secret", help="Strava API Client Secret")
    ap.add_argument("--gpx", help="Path to target GPX file")
    ap.add_argument("--lat", type=float, help="Search center latitude")
    ap.add_argument("--lon", type=float, help="Search center longitude")
    ap.add_argument("--radius-km", type=float, default=15.0,
                     help="Search radius in km (default 15)")
    ap.add_argument("--min-window-frac", type=float, default=0.75,
                     help="Shortest matching window to accept, as a "
                          "fraction of the target's length (default 0.75 "
                          "\u2014 e.g. a 2.87mi window qualifies for a 3.83mi "
                          "target)")
    ap.add_argument("--max-window-frac", type=float, default=1.15,
                     help="Longest matching window to try, as a fraction "
                          "of the target's length (default 1.15)")
    ap.add_argument("--max-segment-mult", type=float, default=3.0,
                     help="Skip candidate segments longer than this "
                          "multiple of the target's distance, to bound "
                          "search cost on very long routes (default 3.0)")
    ap.add_argument("--grade-bin-mi", type=float, default=0.25,
                     help="Grade is computed in fixed real-distance "
                          "chunks of this size (default 0.25 mi), so "
                          "sequence and composition matter \u2014 alternating "
                          "8%%/2%% grades score nothing like a flat 5%%, "
                          "even with identical average grade")
    ap.add_argument("--no-access-check", action="store_true",
                     help="Skip the road-proximity lookup entirely "
                          "(faster, but ranking won't account for how "
                          "reachable a match actually is)")
    ap.add_argument("--access-near-m", type=float, default=400,
                     help="Distance (m) from a road within which a match "
                          "counts as basically accessible, no penalty "
                          "(default 400)")
    ap.add_argument("--access-far-m", type=float, default=1200,
                     help="Distance (m) from a road beyond which a match "
                          "is heavily penalized as effectively "
                          "unreachable (default 1200)")
    ap.add_argument("--weight-shape", type=float, default=1.0,
                     help="Weight on the DTW shape score \u2014 grade "
                          "composition AND sequence (default 1.0)")
    ap.add_argument("--weight-distribution", type=float, default=0.6,
                     help="Weight on the grade-distribution (EMD) score "
                          "\u2014 grade composition regardless of order "
                          "(default 0.6; kept below shape so composition "
                          "isn't double-counted)")
    ap.add_argument("--weight-gain", type=float, default=4.0,
                     help="Weight on relative total-gain deviation "
                          "(default 4.0; scales a 0-1 fraction so it's "
                          "comparable to the shape/distribution terms)")
    ap.add_argument("--google-api-key", default=None,
                     help="Google Maps Platform API key (Roads API "
                          "enabled) for road-proximity checks. If not "
                          "given, checks the GOOGLE_MAPS_API_KEY env var, "
                          "then ~/.google_maps_api_key. Falls back to "
                          "free OpenStreetMap/Overpass lookups if none "
                          "of those are set.")
    ap.add_argument("--debug-access", action="store_true",
                     help="Print the actual HTTP status/error for each "
                          "failed road-proximity lookup, instead of "
                          "silently treating failures as 'no road'")
    ap.add_argument("--top", type=int, default=10,
                     help="Number of top matches to show")
    ap.add_argument("--not-reversible", action="store_true",
                     help="Only match candidates in their as-recorded "
                          "direction. By default a segment is matched in "
                          "whichever direction fits best (and the result "
                          "tells you which way to run it); this disables "
                          "that.")
    ap.add_argument("--export", metavar="SEGMENT_URL_OR_ID", default=None,
                     help="Export mode: given a Strava segment URL (or "
                          "bare ID) from the search results, write that "
                          "segment's geometry to a GPX file instead of "
                          "searching. Use --output-path and --reverse to "
                          "control the output.")
    ap.add_argument("--output-path", default=None,
                     help="Where to write the exported GPX (file path or "
                          "directory). Defaults to the current directory "
                          "with an auto-generated filename. Only used with "
                          "--export.")
    ap.add_argument("--reverse", action="store_true",
                     help="With --export, flip the segment's direction "
                          "before writing the GPX (e.g. to run a matched "
                          "climb as a descent).")
    ap.add_argument("--refresh", action="store_true",
                     help="Ignore cached Strava data and re-fetch "
                          "everything from the API, overwriting the cache "
                          "(use when you think segments have changed)")
    ap.add_argument("--clear-cache", action="store_true",
                     help="Delete the entire on-disk Strava cache and "
                          "exit")
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

        # Try to get a friendly name for the file/track
        meta = get_segment_meta(token, seg_id)
        seg_name = meta.get("name")

        # Resolve output path: if a directory (or default), auto-name the
        # file; if a full path ending in .gpx, use it as-is.
        def slugify(s):
            s = re.sub(r"[^\w\s-]", "", str(s)).strip().lower()
            return re.sub(r"[\s]+", "_", s) or f"segment_{seg_id}"

        base_name = (f"{slugify(seg_name)}" if seg_name
                     else f"segment_{seg_id}")
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
            # treat as a directory that may not exist yet
            out_path = os.path.join(out, default_filename)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        try:
            written = export_segment_gpx(token, seg_id, out_path,
                                          reverse=args.reverse, name=seg_name)
        except (RuntimeError, OSError) as e:
            sys.exit(f"Export failed: {e}")

        label = seg_name or f"segment {seg_id}"
        direction_note = " (reversed)" if args.reverse else ""
        print(f"Exported {label}{direction_note} to:\n  {written}")
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
        print("No Google API key found \u2014 falling back to free "
              "OpenStreetMap/Overpass lookups (less reliable). Set "
              "--google-api-key, GOOGLE_MAPS_API_KEY, or "
              "~/.google_maps_api_key to use Google instead.\n")

    cum_dist, elevs = load_gpx_profile(args.gpx)
    target_dist_m = cum_dist[-1]
    target_dist_mi = target_dist_m / 1609.34
    target_gain_m, target_loss_m = total_gain_loss(elevs)
    target_gain_ft = target_gain_m * 3.28084
    bin_size_m = args.grade_bin_mi * 1609.34
    target_grade_seq = resample_grade_profile(cum_dist, elevs, bin_size_m)

    # Magnitude of the target's dominant vertical travel, used by the
    # gain-consistency term regardless of running direction.
    target_vert_m = max(target_gain_m, target_loss_m)

    print(f"\nTarget: {args.gpx}")
    print(f"  Distance: {target_dist_mi:.2f} mi")
    print(f"  Elevation gain: {target_gain_ft:.0f} ft "
          f"(loss: {target_loss_m*3.28084:.0f} ft)")
    print(f"  Avg grade: {(elevs[-1]-elevs[0])/cum_dist[-1]*100:.1f}%")
    print(f"  Grade sequence (every {args.grade_bin_mi}mi): "
          f"{np.round(target_grade_seq, 1).tolist()}")
    print(f"  Grade composition: {format_band_breakdown(target_grade_seq)}")
    if args.not_reversible:
        print(f"  (matching as-recorded direction only)")
    print()

    min_seg_dist_mi = target_dist_mi * args.min_window_frac
    max_seg_dist_mi = target_dist_mi * args.max_segment_mult

    print(f"Pre-filtering to segments {min_seg_dist_mi:.1f}-"
          f"{max_seg_dist_mi:.1f} mi long. No shape-based rejection at "
          f"this stage \u2014 descents are scored in reverse, not thrown "
          f"out.\n")

    candidates = explore_segments(token, args.lat, args.lon, args.radius_km)

    dist_filtered = []
    for seg in candidates:
        dist_mi = seg["distance"] / 1609.34
        if min_seg_dist_mi <= dist_mi <= max_seg_dist_mi:
            dist_filtered.append(seg)

    print(f"{len(dist_filtered)} candidates pass length filter. "
          f"Fetching profiles and searching for best matching window "
          f"within each...\n")

    scored = []
    for i, seg in enumerate(dist_filtered):
        stream = get_segment_stream(token, seg["id"])
        if stream is None:
            continue
        seg_dist, seg_elev, seg_latlng = stream

        result = find_best_window(
            seg_dist, seg_elev, seg_latlng, target_grade_seq, target_dist_m,
            bin_size_m, target_vert_m, min_frac=args.min_window_frac,
            max_frac=args.max_window_frac, w_shape=args.weight_shape,
            w_dist=args.weight_distribution, w_gain=args.weight_gain,
            reversible=not args.not_reversible,
        )
        if result is None:
            print(f"  [{i+1}/{len(dist_filtered)}] {seg['name']}: "
                  f"too short to contain a qualifying window")
            continue

        score, win_start_m, win_end_m, gain_ft, loss_ft, direction, \
            start_latlng, shape_score, dist_score, gain_dev, \
            win_grade_seq = result

        road_dist = None
        penalty = 0.0
        if not args.no_access_check and start_latlng is not None:
            road_dist = road_distance_m(*start_latlng,
                                         google_api_key=google_api_key,
                                         debug=args.debug_access)
            penalty = access_penalty(road_dist, near_m=args.access_near_m,
                                       far_m=args.access_far_m)

        final_score = score + penalty
        scored.append((final_score, score, penalty, road_dist, seg,
                        win_start_m, win_end_m, gain_ft, loss_ft, direction,
                        shape_score, dist_score, gain_dev, win_grade_seq))

        access_note = (f"road ~{road_dist}m"
                        if isinstance(road_dist, (int, float))
                        else "no road within 2.5km"
                        if road_dist is None and not args.no_access_check
                        else "access unchecked")
        print(f"  [{i+1}/{len(dist_filtered)}] {seg['name']}: "
              f"terrain {score:.2f} (shape {shape_score:.2f}/dist "
              f"{dist_score:.2f}/gain {gain_dev*100:.0f}%) + access "
              f"{penalty:.2f} = {final_score:.2f} ({direction}, "
              f"{access_note})")
        time.sleep(0.3)

    scored.sort(key=lambda x: x[0])

    print(f"\n{'='*70}")
    print(f"TOP {min(args.top, len(scored))} MATCHES "
          f"(lower combined score = more similar shape + more reachable)")
    print(f"{'='*70}\n")

    for rank, (final_score, terrain_score, penalty, road_dist, seg,
               win_start_m, win_end_m, gain_ft, loss_ft, direction,
               shape_score, dist_score, gain_dev, win_grade_seq) in \
            enumerate(scored[:args.top], 1):
        win_start_mi = win_start_m / 1609.34
        win_end_mi = win_end_m / 1609.34
        win_len_mi = win_end_mi - win_start_mi
        net_avg_grade = (gain_ft - loss_ft) / (win_len_mi * 5280) * 100
        seg_total_mi = seg["distance"] / 1609.34
        url = f"https://www.strava.com/segments/{seg['id']}"
        access_str = (f"~{road_dist}m from a road"
                       if isinstance(road_dist, (int, float))
                       else "no road within 2.5km"
                       if road_dist is None and not args.no_access_check
                       else "access not confirmed")
        print(f"{rank}. {seg['name']}  (segment is {seg_total_mi:.1f} mi total)")
        leg_is_climb = gain_ft >= loss_ft
        leg_word = "CLIMB" if leg_is_climb else "DESCENT"
        if direction == "reverse":
            print(f"   Run the segment BACKWARD ({leg_word}) \u2014 start at "
                  f"mile {win_end_mi:.2f}, finish at mile {win_start_mi:.2f}")
        else:
            print(f"   Run the segment as-recorded ({leg_word}) \u2014 mile "
                  f"{win_start_mi:.2f} to {win_end_mi:.2f} ({win_len_mi:.2f} mi)")
        vert_ft = max(gain_ft, loss_ft)
        target_vert_ft = max(target_gain_m, target_loss_m) * 3.28084
        print(f"   Gain: {gain_ft:.0f} ft | Loss: {loss_ft:.0f} ft | "
              f"Grade: {net_avg_grade:.1f}% "
              f"(target vertical: {target_vert_ft:.0f} ft)")
        print(f"   Grade composition: {format_band_breakdown(win_grade_seq)}")
        print(f"   Scores \u2014 shape {shape_score:.2f} | distribution "
              f"{dist_score:.2f} | gain dev {gain_dev*100:.0f}% | "
              f"access {penalty:.2f} | COMBINED {final_score:.2f}")
        print(f"   {url}\n")


if __name__ == "__main__":
    main()
