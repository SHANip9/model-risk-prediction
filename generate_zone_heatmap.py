"""
OmniSight AI — Zone Risk Heatmap Generator  (Live API version)
===============================================================
DevTrails 2026 · Gig Worker Safety

Changes from the original generate_zone_heatmap.py
---------------------------------------------------
1. Fetches zone scores from GET /zones/risk/heatmap (live API) instead
   of reading zone_risk_heatmap_latest.csv
2. Falls back to static CSV if the API is unreachable (local/CI usage)
3. Injects a <script> block into the output HTML so the page
   auto-refreshes its data every 15 minutes without a full page reload —
   this means the iframe in AdminDashboard / ClientDashboard always
   shows current risk without touching the React components

Run locally
-----------
    python scripts/generate_zone_heatmap.py

    # Override API base URL (default: http://127.0.0.1:8000)
    OMNISIGHT_API=https://your-backend.cleverapps.io python scripts/generate_zone_heatmap.py

GitHub Actions
--------------
The workflow (update_heatmap.yml) calls this script on a schedule.
The generated heatmap.html is committed to omnisight-frontend/public/
and Vercel auto-redeploys.  The self-refresh JS inside the HTML then
keeps data current between Action runs.
"""

from __future__ import annotations

import json
import os
import sys
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import requests

try:
    import folium
    from folium.plugins import HeatMap
    import pandas as pd
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
OUTPUT_HTML = ROOT / "omnisight-frontend" / "public" / "heatmap.html"
CSV_FALLBACK = (
    ROOT
    / "omnisight-backend"
    / "model-risk-prediction"
    / "data"
    / "processed"
    / "zone_risk_heatmap_latest.csv"
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

API_BASE      = os.getenv("OMNISIGHT_API", "http://127.0.0.1:8000")
HEATMAP_URL   = f"{API_BASE}/zones/risk/heatmap"
USER_LAT      = 19.0760
USER_LON      = 72.8777
RADIUS_KM     = 25.0

# Zone geo fallback (used when API AND csv are both unavailable)
ZONE_GEO: dict[str, dict] = {
    "zone_1": {"display_name": "Dharavi",      "city": "Mumbai", "latitude": 19.0422, "longitude": 72.8553},
    "zone_2": {"display_name": "Kurla West",   "city": "Mumbai", "latitude": 19.0728, "longitude": 72.8826},
    "zone_3": {"display_name": "Andheri East", "city": "Mumbai", "latitude": 19.1136, "longitude": 72.8697},
    "zone_4": {"display_name": "Bandra Kurla", "city": "Mumbai", "latitude": 19.0596, "longitude": 72.8656},
    "zone_5": {"display_name": "Thane West",   "city": "Thane",  "latitude": 19.1852, "longitude": 72.9710},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def risk_label(score: float) -> tuple[str, str]:
    """(label, hex_color)"""
    if score >= 75:
        return "DANGER",  "#dc2626"
    if score >= 50:
        return "CAUTION", "#f59e0b"
    return "SAFE",    "#16a34a"

# ---------------------------------------------------------------------------
# Data loading — live API → CSV fallback → hardcoded fallback
# ---------------------------------------------------------------------------

def load_from_api() -> list[dict] | None:
    """Fetch zone scores from the live backend API."""
    try:
        resp = requests.get(HEATMAP_URL, timeout=8)
        resp.raise_for_status()
        payload = resp.json()
        zones = payload.get("zones", [])
        if not zones:
            return None
        print(f"[OmniSight] ✅ Loaded {len(zones)} zones from live API ({HEATMAP_URL})")
        return zones
    except Exception as exc:
        print(f"[OmniSight] ⚠️  Live API unavailable ({exc}) — trying CSV fallback")
        return None


def load_from_csv() -> list[dict] | None:
    """Fall back to the static CSV produced by the weekly ML pipeline."""
    if not CSV_FALLBACK.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(CSV_FALLBACK)
        df["week_start"] = pd.to_datetime(df["week_start"])
        df = df.sort_values("week_start").groupby("zone_id").last().reset_index()

        zones = []
        for _, row in df.iterrows():
            zid = row["zone_id"]
            geo = ZONE_GEO.get(zid, {})
            score = float(row["risk_score_0_100"])
            label, color = risk_label(score)
            zones.append({
                "id":           zid,
                "display_name": geo.get("display_name", zid),
                "city":         geo.get("city", ""),
                "center":       [geo.get("latitude", 19.07), geo.get("longitude", 72.88)],
                "risk_score":   score,
                "risk_bin":     str(row.get("risk_bin", "medium")),
                "risk_label":   label,
                "color":        color,
                "condition":    "Historical data",
                "precip_mm":    0.0,
                "wind_kph":     0.0,
            })
        print(f"[OmniSight] ✅ Loaded {len(zones)} zones from CSV fallback")
        return zones
    except Exception as exc:
        print(f"[OmniSight] ⚠️  CSV fallback failed ({exc}) — using hardcoded data")
        return None


def load_hardcoded_fallback() -> list[dict]:
    """Last resort — hardcoded representative scores."""
    print("[OmniSight] ⚠️  Using hardcoded fallback data")
    fallback_scores = {
        "zone_1": 72.0, "zone_2": 65.0,
        "zone_3": 41.0, "zone_4": 68.0, "zone_5": 38.0,
    }
    zones = []
    for zid, geo in ZONE_GEO.items():
        score = fallback_scores.get(zid, 50.0)
        label, color = risk_label(score)
        zones.append({
            "id":           zid,
            "display_name": geo["display_name"],
            "city":         geo["city"],
            "center":       [geo["latitude"], geo["longitude"]],
            "risk_score":   score,
            "risk_bin":     "high" if score >= 66 else ("medium" if score >= 33 else "low"),
            "risk_label":   label,
            "color":        color,
            "condition":    "Fallback data",
            "precip_mm":    0.0,
            "wind_kph":     0.0,
        })
    return zones


def load_zones() -> list[dict]:
    return load_from_api() or load_from_csv() or load_hardcoded_fallback()

# ---------------------------------------------------------------------------
# Self-refresh JavaScript
# Injected into the HTML so the iframe auto-updates every 15 minutes.
# Calls GET /zones/risk/heatmap, then updates marker colors + popups.
# ---------------------------------------------------------------------------

SELF_REFRESH_JS = f"""
<script>
(function () {{
  const API_BASE       = "{API_BASE}";
  const REFRESH_MS     = 15 * 60 * 1000;   // 15 minutes
  const HEATMAP_ENDPOINT = API_BASE + "/zones/risk/heatmap";

  function labelFromScore(score) {{
    if (score >= 75) return {{ label: "DANGER",  color: "#dc2626" }};
    if (score >= 50) return {{ label: "CAUTION", color: "#f59e0b" }};
    return             {{ label: "SAFE",    color: "#16a34a" }};
  }}

  function buildPopup(z) {{
    const {{ label, color }} = labelFromScore(z.risk_score);
    return (
      "<div style='font-family:Inter,sans-serif;font-size:12px;min-width:240px'>" +
      "<div style='background:" + color + ";color:#fff;padding:6px 10px;" +
        "border-radius:8px 8px 0 0;font-weight:700'>" +
        z.id.toUpperCase().replace("_", " ") + " — " + label + "</div>" +
      "<div style='padding:8px 10px;border:1px solid #e5e7eb;" +
        "border-top:none;border-radius:0 0 8px 8px'>" +
        "<b>Area:</b> " + z.display_name + " | <b>City:</b> " + z.city + "<br/>" +
        "<b>Risk Score:</b> <span style='color:" + color + ";font-weight:800;font-size:14px'>" +
        z.risk_score.toFixed(1) + "</span>/100<br/>" +
        "<b>Condition:</b> " + z.condition + "<br/>" +
        "<b>Rain:</b> " + z.precip_mm + " mm | <b>Wind:</b> " + z.wind_kph + " kph" +
      "</div></div>"
    );
  }}

  // Store references to Leaflet circle markers keyed by zone id
  const markerMap = {{}};
  // The map object is created by Folium as window._leaflet_map or similar.
  // We poll until the map is available, then register markers.
  function waitForMap(cb, attempts) {{
    attempts = attempts || 0;
    const maps = Object.values(window).filter(v => v && v._leaflet_id && v._container);
    if (maps.length > 0) {{ cb(maps[0]); return; }}
    if (attempts < 30) setTimeout(() => waitForMap(cb, attempts + 1), 500);
  }}

  function refreshScores() {{
    fetch(HEATMAP_ENDPOINT)
      .then(r => r.json())
      .then(data => {{
        const zones = data.zones || [];
        zones.forEach(z => {{
          const m = markerMap[z.id];
          if (!m) return;
          const {{ color }} = labelFromScore(z.risk_score);
          // Update circle color
          if (m.circle) {{
            m.circle.setStyle({{ color: color, fillColor: color }});
            m.circle.setRadius(500 + z.risk_score * 11);
          }}
          // Update popup
          if (m.marker) {{
            m.marker.getPopup().setContent(buildPopup(z));
          }}
        }});

        // Update banner timestamp
        const banner = document.getElementById("omni-refresh-banner");
        if (banner) {{
          const now = new Date().toLocaleString("en-IN", {{timeZone:"Asia/Kolkata",hour12:true}});
          banner.textContent = "⟳ OmniSight AI — Risk scores refreshed at " + now + " IST";
        }}
      }})
      .catch(err => console.warn("[OmniSight] Refresh failed:", err));
  }}

  // Kick off
  waitForMap(function (map) {{
    map.eachLayer(function (layer) {{
      // Tag Folium CircleMarkers with their zone id via tooltip text
      if (layer.options && layer.options.radius && layer.getTooltip) {{
        const tip = layer.getTooltip();
        if (!tip) return;
        const text = tip._content || "";
        const match = text.match(/zone_\\d/i);
        if (match) {{
          const zid = match[0].toLowerCase();
          if (!markerMap[zid]) markerMap[zid] = {{}};
          if (layer.options.radius > 20) markerMap[zid].circle = layer;
          else markerMap[zid].marker = layer;
        }}
      }}
    }});
  }});

  setInterval(refreshScores, REFRESH_MS);
  console.log("[OmniSight] Self-refresh armed — updating every 15 min from", HEATMAP_ENDPOINT);
}})();
</script>
"""

REFRESH_BANNER = """
<div id="omni-refresh-banner"
  style="position:fixed;bottom:0;left:0;right:0;z-index:9998;
         background:rgba(10,15,26,0.92);color:#6b7280;
         font-family:Inter,sans-serif;font-size:10px;
         text-align:center;padding:4px 0;border-top:1px solid rgba(255,255,255,0.05)">
  ⟳ OmniSight AI — Loading live risk data...
</div>
"""

# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def build_map(zones: list[dict]) -> "folium.Map":
    uc = [USER_LAT, USER_LON]
    m = folium.Map(
        location=uc,
        zoom_start=12,
        tiles="cartodbpositron",
        control_scale=True,
    )

    # Zone layers by risk level
    layer_danger  = folium.FeatureGroup(name="🔴 DANGER  ≥75",   show=True)
    layer_caution = folium.FeatureGroup(name="🟡 CAUTION 50–74", show=True)
    layer_safe    = folium.FeatureGroup(name="🟢 SAFE    <50",   show=True)

    layer_map = {"DANGER": layer_danger, "CAUTION": layer_caution, "SAFE": layer_safe}

    marker_layer = folium.FeatureGroup(name="📍 Zone Markers", show=True)
    heatmap_pts: list = []
    nearest_layer = folium.FeatureGroup(name="🔗 Nearest Zone Links", show=True)

    for z in zones:
        score = z["risk_score"]
        lbl, color = risk_label(score)
        lat, lon = z["center"][0], z["center"][1]
        dist = haversine(USER_LAT, USER_LON, lat, lon)

        # Background shaded circle (tagged with zone id for JS refresh)
        folium.Circle(
            [lat, lon],
            radius=500 + score * 11,
            color=color,
            fill=True,
            fill_opacity=0.13,
            weight=2.5,
            tooltip=f"{z['id']} [{lbl}] {z['display_name']}  {score:.1f}/100",
        ).add_to(layer_map[lbl])

        # Popup card
        popup_html = (
            f"<div style='font-family:Inter,sans-serif;font-size:12px;min-width:240px'>"
            f"<div style='background:{color};color:#fff;padding:6px 10px;"
            f"border-radius:8px 8px 0 0;font-weight:700'>"
            f"{z['id'].upper().replace('_',' ')} — {lbl}</div>"
            f"<div style='padding:8px 10px;border:1px solid #e5e7eb;"
            f"border-top:none;border-radius:0 0 8px 8px'>"
            f"<b>Area:</b> {z['display_name']} | <b>City:</b> {z['city']}<br/>"
            f"<b>Risk Score:</b> <span style='color:{color};font-weight:800;font-size:14px'>"
            f"{score:.2f}</span>/100<br/>"
            f"<b>Condition:</b> {z['condition']}<br/>"
            f"<b>Rain:</b> {z['precip_mm']} mm | <b>Wind:</b> {z['wind_kph']} kph<br/>"
            f"<b>Distance:</b> {dist:.2f} km"
            f"</div></div>"
        )

        # Circle marker
        folium.CircleMarker(
            [lat, lon],
            radius=8 + max(0, min(12, score / 10)),
            color=color,
            fill=True,
            fill_opacity=0.88,
            tooltip=f"{z['id']} | {z['display_name']} | {score:.1f}/100 | {dist:.1f}km",
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(marker_layer)

        # Score label
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=(
                    f"<div style='background:{color};color:#fff;padding:2px 7px;"
                    f"border-radius:8px;font-size:10px;font-weight:700;"
                    f"white-space:nowrap;box-shadow:0 1px 5px rgba(0,0,0,.3);"
                    f"margin-top:18px'>"
                    f"{z['id'].upper().replace('_',' ')} · {score:.0f}</div>"
                ),
                icon_size=(110, 22),
                icon_anchor=(55, 0),
            ),
        ).add_to(marker_layer)

        heatmap_pts.append([lat, lon, max(0.0, score / 100.0)])

    # User location pin
    folium.Marker(
        uc,
        tooltip="📍 Gig Worker Location",
        icon=folium.Icon(color="blue", icon="motorcycle", prefix="fa"),
    ).add_to(m)

    # Scan radius ring
    folium.Circle(
        uc,
        radius=RADIUS_KM * 1000,
        color="#0ea5e9",
        fill=False,
        weight=2,
        dash_array="6",
        tooltip=f"OmniSight {RADIUS_KM:.0f}km scan radius",
    ).add_to(m)

    # Nearest zone link lines
    sorted_zones = sorted(
        zones,
        key=lambda z: haversine(USER_LAT, USER_LON, z["center"][0], z["center"][1]),
    )
    for z in sorted_zones[:3]:
        _, color = risk_label(z["risk_score"])
        dist = haversine(USER_LAT, USER_LON, z["center"][0], z["center"][1])
        folium.PolyLine(
            [uc, z["center"]],
            color=color,
            weight=2.5,
            opacity=0.55,
            tooltip=f"{z['display_name']} → {dist:.2f} km",
        ).add_to(nearest_layer)

    # Add all layers
    for lg in [layer_danger, layer_caution, layer_safe]:
        lg.add_to(m)
    marker_layer.add_to(m)
    nearest_layer.add_to(m)

    # Heatmap overlay
    HeatMap(
        heatmap_pts,
        min_opacity=0.3,
        radius=40,
        blur=26,
        max_zoom=13,
        name="🌡️ Risk Heatmap",
    ).add_to(m)

    # Danger alert banner
    danger_zones = [z["display_name"] for z in zones if z["risk_score"] >= 75]
    if danger_zones:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);"'
            f'style="z-index:9999;background:#dc2626;color:#fff;padding:7px 20px;'
            f'border-radius:24px;font-family:Inter,sans-serif;font-size:11px;'
            f'font-weight:700;box-shadow:0 4px 14px rgba(220,38,38,0.45);white-space:nowrap">'
            f'⚠️ DANGER ALERT — {" · ".join(danger_zones)} — High risk for gig workers</div>'
        ))

    # Legend
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:30px;left:20px;z-index:9999;'
        'background:#0a0f1a;color:#fff;padding:12px 15px;border-radius:12px;'
        'font-family:Inter,sans-serif;font-size:11px;min-width:210px;'
        'border:1px solid rgba(255,255,255,0.1)">'
        '<div style="font-size:13px;font-weight:800;margin-bottom:6px">🛵 OmniSight AI</div>'
        '<div style="color:#6b7280;font-size:10px;margin-bottom:8px">Live · Updates every 15 min</div>'
        '<div style="margin:3px 0"><span style="background:#dc2626;color:#fff;'
        'padding:1px 8px;border-radius:20px;font-weight:700;font-size:10px">DANGER</span> ≥75</div>'
        '<div style="margin:3px 0"><span style="background:#f59e0b;color:#fff;'
        'padding:1px 8px;border-radius:20px;font-weight:700;font-size:10px">CAUTION</span> 50–74</div>'
        '<div style="margin:3px 0"><span style="background:#16a34a;color:#fff;'
        'padding:1px 8px;border-radius:20px;font-weight:700;font-size:10px">SAFE</span> &lt;50</div>'
        '<div style="color:#6b7280;font-size:9px;margin-top:6px">⬤ 25km scan · Click markers for detail</div>'
        '</div>'
    ))

    folium.LayerControl(collapsed=False).add_to(m)

    # Self-refresh JS + status banner
    m.get_root().html.add_child(folium.Element(REFRESH_BANNER))
    m.get_root().html.add_child(folium.Element(SELF_REFRESH_JS))

    return m

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not HAS_FOLIUM:
        print("[OmniSight] ❌ folium not installed. Run: pip install folium pandas")
        sys.exit(1)

    print(f"[OmniSight] Fetching live zone scores from {HEATMAP_URL} ...")
    zones = load_zones()

    print(f"[OmniSight] Building map for {len(zones)} zones ...")
    m = build_map(zones)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUTPUT_HTML))

    print(f"[OmniSight] ✅ Heatmap saved → {OUTPUT_HTML}")
    print(f"[OmniSight]    Self-refresh JS injected — iframe will poll {HEATMAP_URL} every 15 min")

    # Print score summary
    print("\n  Zone Risk Summary:")
    print(f"  {'Zone':<10} {'Name':<16} {'Score':>6}  {'Level'}")
    print("  " + "-" * 46)
    for z in zones:
        lbl, _ = risk_label(z["risk_score"])
        print(f"  {z['id']:<10} {z['display_name']:<16} {z['risk_score']:>6.1f}  {lbl}")


if __name__ == "__main__":
    main()
