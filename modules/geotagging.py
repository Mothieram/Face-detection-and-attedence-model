from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from config import (
    CACHE_GRID_DEG,
    GEOCODE_CACHE_MAX,
    GEOCODE_CACHE_TTL,
    GEOCODE_TIMEOUT,
    GEOCODER_API_KEY,
    GEOCODER_BACKEND,
    GEOFENCE_ENFORCE,
    GEOFENCE_ZONES,
    GEOTAGGING_ENABLED,
    NOMINATIM_USER_AGENT,
)

# ---------------------------------------------------------------------------
# Module setup and optional dependency loaders
# ---------------------------------------------------------------------------
logger = logging.getLogger("faceapi.geo")


def _import_geopy_async():
    from geopy.adapters import AioHTTPAdapter
    from geopy.geocoders import Nominatim

    return Nominatim, AioHTTPAdapter


def _import_shapely():
    from shapely.geometry import Point, Polygon

    return Point, Polygon


def _import_piexif():
    import piexif

    return piexif


def _import_timezonefinder():
    from timezonefinder import TimezoneFinder

    return TimezoneFinder


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class Coordinates:
    lat: float
    lon: float
    source: str = "unknown"
    accuracy_m: Optional[float] = None

    def is_valid(self) -> bool:
        return (
            -90.0 <= self.lat <= 90.0
            and -180.0 <= self.lon <= 180.0
            and not (self.lat == 0.0 and self.lon == 0.0)
        )

    def cache_key(self) -> str:
        flat = round(self.lat / CACHE_GRID_DEG) * CACHE_GRID_DEG
        flon = round(self.lon / CACHE_GRID_DEG) * CACHE_GRID_DEG
        return f"{flat:.4f},{flon:.4f}"


@dataclass
class GeoInfo:
    coordinates: Optional[Coordinates] = None
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    timezone: Optional[str] = None
    local_time: Optional[str] = None
    geofence_zone: Optional[str] = None
    geofence_passed: bool = True
    geofence_reason: str = ""
    geocode_cached: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        coords = None
        if self.coordinates:
            coords = {
                "lat": self.coordinates.lat,
                "lon": self.coordinates.lon,
                "source": self.coordinates.source,
                "accuracy_m": self.coordinates.accuracy_m,
            }
        return {
            "coordinates": coords,
            "address": self.address,
            "city": self.city,
            "country": self.country,
            "country_code": self.country_code,
            "timezone": self.timezone,
            "local_time": self.local_time,
            "geofence_zone": self.geofence_zone,
            "geofence_passed": self.geofence_passed,
            "geofence_reason": self.geofence_reason,
            "geocode_cached": self.geocode_cached,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Lightweight in-memory TTL cache (for reverse geocode results)
# ---------------------------------------------------------------------------
class _TTLCache:
    def __init__(self, maxsize: int, ttl: int):
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store[key] = (time.monotonic(), value)
                self._store.move_to_end(key)
                return
            if len(self._store) >= self._maxsize:
                self._store.popitem(last=False)
            self._store[key] = (time.monotonic(), value)


_geocode_cache = _TTLCache(maxsize=GEOCODE_CACHE_MAX, ttl=GEOCODE_CACHE_TTL)
_tz_finder = None
_tz_finder_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Timezone finder singleton bootstrap
# ---------------------------------------------------------------------------
def _get_tz_finder():
    global _tz_finder
    if _tz_finder is not None:
        return _tz_finder
    with _tz_finder_lock:
        if _tz_finder is None:
            try:
                _tz_finder = _import_timezonefinder()()
            except Exception as exc:
                logger.warning("[GEO] timezonefinder unavailable: %s", exc)
                _tz_finder = False
    return _tz_finder if _tz_finder else None


# ---------------------------------------------------------------------------
# EXIF GPS extraction
# ---------------------------------------------------------------------------
def extract_exif_gps(jpeg_bytes: bytes) -> Optional[Coordinates]:
    try:
        piexif = _import_piexif()
        exif_dict = piexif.load(jpeg_bytes)
        gps = exif_dict.get("GPS", {})
        if not gps:
            return None

        def _to_float(v) -> float:
            if isinstance(v, tuple) and len(v) == 2 and v[1] != 0:
                return float(v[0]) / float(v[1])
            return float(v)

        def _dms_to_dd(dms_val, ref_val) -> Optional[float]:
            if not dms_val or len(dms_val) < 3:
                return None
            try:
                deg = _to_float(dms_val[0])
                mins = _to_float(dms_val[1])
                sec = _to_float(dms_val[2])
                dd = deg + mins / 60.0 + sec / 3600.0
                ref = ref_val.decode("ascii", errors="ignore").upper() if isinstance(ref_val, bytes) else str(ref_val).upper()
                if ref[:1] in ("S", "W"):
                    dd = -dd
                return dd
            except Exception:
                return None

        lat = _dms_to_dd(gps.get(piexif.GPSIFD.GPSLatitude), gps.get(piexif.GPSIFD.GPSLatitudeRef))
        lon = _dms_to_dd(gps.get(piexif.GPSIFD.GPSLongitude), gps.get(piexif.GPSIFD.GPSLongitudeRef))
        if lat is None or lon is None:
            return None

        coords = Coordinates(lat=lat, lon=lon, source="exif")
        return coords if coords.is_valid() else None
    except Exception as exc:
        logger.debug("[GEO] EXIF parse skipped: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Geofence helpers (circle and polygon)
# ---------------------------------------------------------------------------
def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _point_in_polygon_zone(lat: float, lon: float, zone: dict) -> bool:
    point_cls, polygon_cls = _import_shapely()
    polygon_points = zone.get("polygon", [])
    if len(polygon_points) < 3:
        return False
    poly = polygon_cls([(p[1], p[0]) for p in polygon_points])
    return poly.contains(point_cls(lon, lat))


def check_geofence(lat: float, lon: float, zone_id: Optional[str] = None) -> dict:
    if not GEOFENCE_ZONES:
        return {
            "passed": True,
            "zone_id": None,
            "zone_name": None,
            "distance_m": None,
            "reason": "no_zone_configured",
        }

    zones = [z for z in GEOFENCE_ZONES if z.get("id") == zone_id] if zone_id else list(GEOFENCE_ZONES)
    if zone_id and not zones:
        return {
            "passed": False,
            "zone_id": None,
            "zone_name": None,
            "distance_m": None,
            "reason": "zone_not_found",
        }

    for zone in zones:
        ztype = zone.get("type", "circle")
        if ztype == "circle":
            center = zone.get("center") or zone.get("centre")
            if not center or len(center) != 2:
                continue
            dist = _haversine_m(lat, lon, float(center[0]), float(center[1]))
            radius = float(zone.get("radius_m", 100.0))
            if dist <= radius:
                return {
                    "passed": True,
                    "zone_id": zone.get("id"),
                    "zone_name": zone.get("name"),
                    "distance_m": round(dist, 1),
                    "reason": "inside_zone",
                }
        elif ztype == "polygon":
            try:
                if _point_in_polygon_zone(lat, lon, zone):
                    return {
                        "passed": True,
                        "zone_id": zone.get("id"),
                        "zone_name": zone.get("name"),
                        "distance_m": None,
                        "reason": "inside_zone",
                    }
            except ImportError:
                logger.warning("[GEO] shapely not installed, polygon geofence skipped")
                return {
                    "passed": True,
                    "zone_id": None,
                    "zone_name": None,
                    "distance_m": None,
                    "reason": "shapely_missing",
                }

    return {
        "passed": False,
        "zone_id": None,
        "zone_name": None,
        "distance_m": None,
        "reason": "outside_all_zones",
    }


# ---------------------------------------------------------------------------
# Reverse geocoding backends + cache wrapper
# ---------------------------------------------------------------------------
def _empty_geocode(error: str = "") -> dict:
    return {
        "address": "",
        "city": "",
        "country": "",
        "country_code": "",
        "cached": False,
        "error": error,
    }


async def reverse_geocode(lat: float, lon: float) -> dict:
    coords = Coordinates(lat=lat, lon=lon)
    if not coords.is_valid():
        return _empty_geocode(error="invalid_coordinates")

    key = coords.cache_key()
    cached = _geocode_cache.get(key)
    if cached is not None:
        return {**cached, "cached": True}

    try:
        if GEOCODER_BACKEND == "nominatim":
            result = await _nominatim_reverse(lat, lon)
        elif GEOCODER_BACKEND == "google":
            result = await _google_reverse(lat, lon)
        elif GEOCODER_BACKEND == "here":
            result = await _here_reverse(lat, lon)
        else:
            return _empty_geocode(error=f"unknown_backend:{GEOCODER_BACKEND}")
    except asyncio.TimeoutError:
        logger.warning("[GEO] reverse geocode timeout for %.5f, %.5f", lat, lon)
        return _empty_geocode(error="timeout")
    except Exception as exc:
        logger.warning("[GEO] reverse geocode error: %s", exc)
        return _empty_geocode(error=str(exc))

    result = {**result, "cached": False}
    _geocode_cache.set(key, result)
    return result


# Backend-specific reverse geocode implementations
async def _nominatim_reverse(lat: float, lon: float) -> dict:
    nominatim_cls, aio_adapter = _import_geopy_async()
    async with nominatim_cls(
        user_agent=NOMINATIM_USER_AGENT,
        adapter_factory=aio_adapter,
        timeout=GEOCODE_TIMEOUT,
    ) as geolocator:
        location = await geolocator.reverse(f"{lat}, {lon}", language="en", exactly_one=True)
    if location is None:
        return _empty_geocode(error="no_result")
    addr = location.raw.get("address", {})
    return {
        "address": location.address or "",
        "city": addr.get("city") or addr.get("town") or addr.get("village") or "",
        "country": addr.get("country", ""),
        "country_code": addr.get("country_code", "").upper(),
    }


async def _google_reverse(lat: float, lon: float) -> dict:
    from geopy.adapters import AioHTTPAdapter
    from geopy.geocoders import GoogleV3

    async with GoogleV3(
        api_key=GEOCODER_API_KEY,
        adapter_factory=AioHTTPAdapter,
        timeout=GEOCODE_TIMEOUT,
    ) as geolocator:
        location = await geolocator.reverse(f"{lat}, {lon}", exactly_one=True)

    if location is None:
        return _empty_geocode(error="no_result")

    components = location.raw.get("address_components", [])

    def _component(target_type: str, field: str = "long_name") -> str:
        for item in components:
            if target_type in item.get("types", []):
                return item.get(field, "")
        return ""

    return {
        "address": location.address or "",
        "city": _component("locality") or _component("administrative_area_level_2"),
        "country": _component("country"),
        "country_code": _component("country", "short_name").upper(),
    }


async def _here_reverse(lat: float, lon: float) -> dict:
    from geopy.adapters import AioHTTPAdapter
    from geopy.geocoders import Here

    async with Here(
        apikey=GEOCODER_API_KEY,
        adapter_factory=AioHTTPAdapter,
        timeout=GEOCODE_TIMEOUT,
    ) as geolocator:
        location = await geolocator.reverse(f"{lat}, {lon}", exactly_one=True)

    if location is None:
        return _empty_geocode(error="no_result")
    raw = location.raw.get("address", {})
    return {
        "address": location.address or "",
        "city": raw.get("city", ""),
        "country": raw.get("countryName", ""),
        "country_code": raw.get("countryCode", "").upper(),
    }


# ---------------------------------------------------------------------------
# Timezone + local timestamp helpers
# ---------------------------------------------------------------------------
def resolve_timezone(lat: float, lon: float) -> Optional[str]:
    tf = _get_tz_finder()
    if tf is None:
        return None
    try:
        return tf.timezone_at(lat=lat, lng=lon)
    except Exception as exc:
        logger.debug("[GEO] timezone lookup failed: %s", exc)
        return None


def local_iso_timestamp(lat: float, lon: float) -> str:
    now_utc = datetime.now(timezone.utc)
    tz_name = resolve_timezone(lat, lon)
    if not tz_name:
        return now_utc.isoformat()

    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        try:
            from backports.zoneinfo import ZoneInfo  # type: ignore
        except ImportError:
            return now_utc.isoformat()

    try:
        return now_utc.astimezone(ZoneInfo(tz_name)).isoformat()
    except Exception:
        return now_utc.isoformat()


# ---------------------------------------------------------------------------
# Main pipeline entrypoint
# ---------------------------------------------------------------------------
async def geotag_event(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    source: str = "browser",
    accuracy_m: Optional[float] = None,
    image_bytes: Optional[bytes] = None,
    zone_id: Optional[str] = None,
    skip_geocode: bool = False,
) -> GeoInfo:
    if not GEOTAGGING_ENABLED:
        return GeoInfo(geofence_passed=True, geofence_reason="geotagging_disabled")

    coords: Optional[Coordinates] = None
    if lat is not None and lon is not None:
        coords = Coordinates(lat=lat, lon=lon, source=source, accuracy_m=accuracy_m)
        if not coords.is_valid():
            return GeoInfo(error="invalid_coordinates", geofence_passed=not GEOFENCE_ENFORCE)

    if coords is None and image_bytes:
        coords = extract_exif_gps(image_bytes)

    if coords is None:
        if GEOFENCE_ENFORCE:
            return GeoInfo(
                geofence_passed=False,
                geofence_reason="no_coordinates_provided",
                error="no_coordinates",
            )
        return GeoInfo(geofence_passed=True, geofence_reason="no_coordinates")

    fence = check_geofence(coords.lat, coords.lon, zone_id=zone_id)
    geo = GeoInfo(
        coordinates=coords,
        geofence_zone=fence.get("zone_id"),
        geofence_passed=fence["passed"] or not GEOFENCE_ENFORCE,
        geofence_reason=fence["reason"],
    )

    if not fence["passed"] and GEOFENCE_ENFORCE:
        logger.warning("[GEO] geofence reject for %.5f, %.5f: %s", coords.lat, coords.lon, fence["reason"])
        return geo

    geo.timezone = resolve_timezone(coords.lat, coords.lon)
    geo.local_time = local_iso_timestamp(coords.lat, coords.lon)

    if not skip_geocode:
        gc = await asyncio.wait_for(reverse_geocode(coords.lat, coords.lon), timeout=GEOCODE_TIMEOUT + 1)
        geo.address = gc.get("address", "")
        geo.city = gc.get("city", "")
        geo.country = gc.get("country", "")
        geo.country_code = gc.get("country_code", "")
        geo.geocode_cached = bool(gc.get("cached", False))
        if gc.get("error"):
            geo.error = str(gc["error"])

    return geo


# ---------------------------------------------------------------------------
# Compatibility helper for FastAPI form defaults
# ---------------------------------------------------------------------------
def make_geo_form_fields():
    return None, None, None


# ---------------------------------------------------------------------------
# Local self-test (run: python -m modules.geotagging)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_lat = 12.9716
    test_lon = 77.5946
    print("[GEO SELF-TEST] import_ok")
    print(f"[GEO SELF-TEST] geofence={check_geofence(test_lat, test_lon)}")
    print(f"[GEO SELF-TEST] timezone={resolve_timezone(test_lat, test_lon)}")
    result = asyncio.run(geotag_event(lat=test_lat, lon=test_lon, skip_geocode=True))
    print(f"[GEO SELF-TEST] geotag_event={result.to_dict()}")
