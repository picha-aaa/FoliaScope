import os
import io
import json
import time
import base64
from typing import Optional, Dict, Any
import requests
from PIL import Image

from google.adk import Agent
import google.auth
from google.auth.transport.requests import Request as GRequest

DEBUG_LOG = True

def log(*args):
    if DEBUG_LOG:
        print(*args, flush=True)

def _preview_json(label: str, obj: Any, limit: int = 1200):
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    if len(s) > limit:
        s = s[:limit] + "…"
    log(label, s)

def predict_foliage_duration(location: str, species: str, peak_status_percent: int) -> Dict[str, Any]:
    log("[TOOL] predict_foliage_duration() called with:",
        {"location": location, "species": species, "peak_status_percent": peak_status_percent})

    species_lower = (species or "").strip().lower()
    baseline = 7 
    if species_lower in {"ginkgo", "ginkgo biloba"}:
        baseline = 4
    elif species_lower in {"maple", "sugar maple", "red maple"}:
        baseline = 6
    elif species_lower in {"oak", "red oak", "white oak"}:
        baseline = 8
    elif species_lower in {"sweetgum"}:
        baseline = 7
    elif species_lower in {"birch", "cherry"}:
        baseline = 5

    pct = max(0, min(int(peak_status_percent or 0), 100))
    remaining = max(1, int(round(baseline * (100 - pct) / 100.0)))

    result = {
        "location": location,
        "species": species or "unknown",
        "peak_status_percent": pct,
        "estimated_days_remaining": remaining,
        "note": "Heuristic baseline; adjust with weather."
    }
    _preview_json("[TOOL] predict_foliage_duration →", result)
    return result

def get_weather_forecast(location: str) -> Dict[str, Any]:
    api_key = os.environ.get("WEATHER_API_KEY")
    if not api_key:
        return {"status": "noop", "message": "WEATHER_API_KEY not set", "location": location}

    def _fmt(days):
        bullets, critical = [], "none noted"
        for item in days:
            desc = (item.get("weather", [{}])[0].get("description") or "").strip()
            temp = item.get("main", {}).get("temp", 0)
            try:
                temp_disp = f"{float(temp):.0f}°C"
            except Exception:
                temp_disp = f"{temp}°C"
            bullets.append(f"{desc}, {temp_disp}")
            if any(w in desc.lower() for w in ("rain", "storm", "snow", "wind", "gale")):
                critical = desc
        return bullets, critical

    def _forecast_q(q: str) -> Dict[str, Any]:
        url = (
            "https://api.openweathermap.org/data/2.5/forecast"
            f"?q={requests.utils.quote(q)}&appid={api_key}&units=metric&cnt=24"
        )
        log("[TOOL] weather URL:", url)
        r = requests.get(url, timeout=20)
        data = r.json()
        if r.status_code == 200 and "list" in data:
            bullets, critical = _fmt(data["list"][::8][:3])
            city_name = (data.get("city") or {}).get("name") or q
            return {"status": "success", "days": bullets, "critical_event": critical, "location": city_name}
        return {"status": "error", "message": data.get("message", "weather API error"), "location": q}

    def _forecast_latlon(q: str) -> Dict[str, Any]:
        geo = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": q, "limit": 1, "appid": api_key},
            timeout=20,
        ).json()
        if not geo:
            return {"status": "error", "message": "geocoding_failed", "location": q}
        lat, lon = geo[0].get("lat"), geo[0].get("lon")
        if lat is None or lon is None:
            return {"status": "error", "message": "no_lat_lon", "location": q}

        url = (
            "https://api.openweathermap.org/data/2.5/forecast"
            f"?lat={lat}&lon={lon}&appid={api_key}&units=metric&cnt=24"
        )
        log("[TOOL] weather URL:", url)
        r = requests.get(url, timeout=20)
        data = r.json()
        if r.status_code == 200 and "list" in data:
            bullets, critical = _fmt(data["list"][::8][:3])
            city_name = (data.get("city") or {}).get("name") or q
            return {"status": "success", "days": bullets, "critical_event": critical, "location": city_name}
        return {"status": "error", "message": data.get("message", "weather API error"), "location": q}

    r1 = _forecast_q(location)
    if r1["status"] == "success":
        _preview_json("[TOOL] get_weather_forecast →", r1)
        return r1

    if "," in location:
        parts = [p.strip() for p in location.split(",")]
        if len(parts) >= 2 and len(parts[1]) in (2, 3):
            r2 = _forecast_q(f"{parts[0]},{parts[1]},US")
            if r2["status"] == "success":
                _preview_json("[TOOL] get_weather_forecast →", r2)
                return r2

    r3 = _forecast_latlon(location)
    _preview_json("[TOOL] get_weather_forecast →", r3)
    return r3


def search_google_for_context(query: str) -> Dict[str, Any]:
    log("[TOOL] search_google_for_context() called with:", {"query": query})
    api_key = os.environ.get("GOOGLE_API_KEY")
    cse_id = os.environ.get("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        result = {"status": "noop", "message": "GOOGLE_API_KEY/GOOGLE_CSE_ID not set", "query": query}
        _preview_json("[TOOL] search_google_for_context →", result)
        return result

    try:
        url = (
            "https://www.googleapis.com/customsearch/v1"
            f"?key={api_key}&cx={cse_id}&q={requests.utils.quote(query)}"
        )
        log("[TOOL] CSE URL:", url)
        resp = requests.get(url, timeout=20).json()
        items = resp.get("items", [])[:3]
        results = [{"title": it.get("title"), "link": it.get("link"), "snippet": it.get("snippet")} for it in items]
        result = {"status": "success", "results": results, "query": query}
        _preview_json("[TOOL] search_google_for_context →", result)
        return result
    except Exception as e:
        result = {"status": "error", "message": str(e), "query": query}
        _preview_json("[TOOL] search_google_for_context →", result)
        return result

folia_agent = Agent(
    name="folia_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are 'Folia', a foliage advisor.\n"
        "You MUST call tools in this order when needed:\n"
        "  1) predict_foliage_duration(location, species, peak_status_percent)\n"
        "  2) get_weather_forecast(location)\n"
        "\n"
        "From predict_foliage_duration, COPY the numeric peak value into your output key "
        "'peak_status_percent'.\n"
        "\n"
        "The weather tool returns a JSON like:\n"
        '  {\"status\":\"success\",\"days\":[\"few clouds, 16°C\",\"light rain, 11°C\",\"light rain, 9°C\"],'
        ' \"critical_event\":\"light rain\",\"location\":\"Atlanta\"}\n'
        "You MUST copy the array under key 'days' EXACTLY into your output as 'weather_days'. "
        "These are plain strings (no 'Day i:' prefix); do NOT add labels to them. "
        "Copy 'critical_event' EXACTLY into 'weather_critical'.\n"
        "\n"
        "If the weather tool is not success or has no 'days', set:\n"
        "  'weather_days': [],\n"
        "  'weather_critical': 'weather_api_failed',\n"
        "  'estimated_days_color_will_last': null.\n"
        "\n"
        "After copying weather, adjust the foliage duration downward if critical_event mentions "
        "rain, storm, snow, or wind.\n"
        "\n"
        "Return EXACTLY ONE JSON object with keys:\n"
        '  \"tree_species\": string (best effort; use \"unknown\" if unsure),\n'
        '  \"peak_status_percent\": integer|null,\n'
        '  \"estimated_days_color_will_last\": integer|null,\n'
        '  \"estimated_color_change_date\": string|null,\n'
        '  \"weather_days\": array of strings,\n'
        '  \"weather_critical\": string,\n'
        '  \"video_prompt\": string|null\n'
        "Output ONLY the JSON. No prose."
    ),
    tools=[predict_foliage_duration, get_weather_forecast, search_google_for_context],
)


def generate_video_forecast(prompt: str, image: Image.Image, max_retries: int = 2) -> Optional[str]:
    log("[VIDEO] generate_video_forecast() prompt:", (prompt or "")[:400])

    img_b64 = None
    try:
        buf = io.BytesIO()
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        image.save(buf, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        log("[VIDEO] WARN: could not encode image:", e)

    project_id = (
        os.environ.get("GOOGLE_PROJECT_ID")
        or os.environ.get("GCP_PROJECT")
        or os.environ.get("PROJECT_ID")
    )
    location = os.environ.get("GOOGLE_LOCATION") or os.environ.get("GCP_LOCATION") or "us-central1"
    if not project_id:
        log("[VIDEO] ERROR: no project id in env.")
        return None

    try:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(GRequest())
        token = creds.token
    except Exception as e:
        log("[VIDEO] ERROR getting token:", e)
        return None

    start_url = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{location}/publishers/google/models/veo-3.1-fast-generate-preview:predictLongRunning"
    )

    full_prompt = (
        "Time-lapse of the uploaded fall tree over the predicted number of days, "
        "keep the same trunk/branch structure and camera angle. "
        "Apply the specified weather (rain/wind) on the forecast day so leaves fall faster. "
        f"{prompt or ''}"
    )

    instance: Dict[str, Any] = {"prompt": full_prompt}
    if img_b64:
        instance["image"] = {"bytesBase64Encoded": img_b64, "mimeType": "image/jpeg"}

    start_body = {"instances": [instance], "parameters": {"duration": 8, "sampleCount": 1}}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    for attempt in range(max_retries + 1):
        try:
            if attempt:
                wait = 5 * (2 ** (attempt - 1))
                log(f"[VIDEO] Retry {attempt}/{max_retries} after {wait}s…")
                time.sleep(wait)
            start_resp = requests.post(start_url, headers=headers, json=start_body, timeout=60)
            start_resp.raise_for_status()
            op_name = start_resp.json()["name"]
            log("[VIDEO] operationName:", op_name)
            break
        except requests.HTTPError as e:
            log(f"[VIDEO] start error (attempt {attempt+1}):", e, getattr(start_resp, "text", ""))
            if attempt >= max_retries:
                return None
        except Exception as e:
            log(f"[VIDEO] start error (attempt {attempt+1}):", e)
            if attempt >= max_retries:
                return None
    else:
        return None

    poll_url = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{location}/publishers/google/models/veo-3.1-fast-generate-preview:fetchPredictOperation"
    )
    for _ in range(24):
        r = requests.post(poll_url, headers=headers, json={"operationName": op_name}, timeout=60)
        if r.status_code >= 400:
            log("[VIDEO] polling error:", r.text)
            return None
        data = r.json()
        if data.get("done"):
            videos = data.get("response", {}).get("videos", [])
            if not videos:
                log("[VIDEO] done but no videos[]")
                _preview_json("[VIDEO] full response →", data)
                return None
            vid = videos[0]
            b64 = vid.get("bytesBase64Encoded")
            if b64:
                return f"data:video/mp4;base64,{b64}"
            uri = vid.get("gcsUri") or vid.get("uri")
            return uri or None
        log("[VIDEO] still running…")
        time.sleep(5)

    log("[VIDEO] timed out waiting for video")
    return None
