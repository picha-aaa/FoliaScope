import os
import io
import json
import asyncio
import base64
import re
from types import SimpleNamespace
from pathlib import Path

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

import vertexai
from google.api_core.exceptions import ResourceExhausted, PermissionDenied, NotFound
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import google.generativeai as genai

from folia_agent.agent import (
    folia_agent,
    generate_video_forecast as _agent_generate_video_forecast,
    get_weather_forecast,
)

load_dotenv()


def get_config(name: str, default: str | None = None):
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)


GOOGLE_API_KEY = get_config("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = get_config("OPENWEATHER_API_KEY")
GCP_PROJECT_ID = get_config("GCP_PROJECT_ID")
GCP_LOCATION = get_config("GCP_LOCATION", "us-central1")

APP_NAME = "Folia Forecaster"
USER_ID = "user-1"
SESSION_ID = "session-1"


try:
    if GCP_PROJECT_ID:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        print(f"[app] Vertex AI initialized for project={GCP_PROJECT_ID}, location={GCP_LOCATION}")
    else:
        print("[app] WARN: GCP_PROJECT_ID not set, skipping vertexai.init()")
except Exception as e:
    print("[app] WARN: could not init Vertex AI:", e)


SESSION_SERVICE = InMemorySessionService()
RUNNER = Runner(agent=folia_agent, session_service=SESSION_SERVICE, app_name=APP_NAME)


async def _ensure_session():
    try:
        await SESSION_SERVICE.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID, state={}
        )
    except Exception:
        pass


asyncio.run(_ensure_session())

GENAI_MODEL = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        GENAI_MODEL = genai.GenerativeModel("gemini-2.5-flash")
        print("[app] Direct GENAI_MODEL initialized.")
    except Exception as e:
        print("[app] ERROR initializing direct GEMINI:", e)
        GENAI_MODEL = None
else:
    print("[app] WARN: no GOOGLE_API_KEY found in secrets/env")

SPECIES_LABELS = [
    "maple", "oak", "ginkgo", "sweetgum", "birch",
    "beech", "cherry", "poplar", "aspen", "hickory", "elm",
]


def _to_text_from_adk_response(resp) -> str | None:
    try:
        if hasattr(resp, "text") and isinstance(resp.text, str) and resp.text.strip():
            return resp.text
        candidates = getattr(resp, "candidates", None)
        if candidates:
            out = []
            for c in candidates:
                content = getattr(c, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    for p in parts:
                        t = getattr(p, "text", None)
                        if isinstance(t, str) and t.strip():
                            out.append(t)
            if out:
                return "\n".join(out)
        if isinstance(resp, dict):
            return json.dumps(resp)
        return str(resp) if resp is not None else None
    except Exception:
        return None


def _image_to_inline_part(image: Image.Image) -> dict:
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    image.thumbnail((1024, 1024))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return {"inline_data": {"mime_type": "image/jpeg", "data": buf.getvalue()}}


def _classify_species_with_gemini(image: Image.Image) -> str | None:
    if GENAI_MODEL is None:
        return None
    try:
        part = _image_to_inline_part(image)
        prompt = (
            "Identify the dominant tree species in this photo. "
            "Reply ONLY with one word from this set: "
            + ", ".join(SPECIES_LABELS)
            + ". If you truly cannot tell, reply 'unknown'."
        )
        resp = GENAI_MODEL.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}, part]}],
            generation_config={
                "response_mime_type": "text/plain",
                "temperature": 0.2,
            },
        )
        guess = (getattr(resp, "text", "") or "").strip().lower()
        guess = re.sub(r"[^a-z]", "", guess)
        if guess in SPECIES_LABELS:
            return guess
        return None
    except Exception:
        return None


def _strip_dup_prefix(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    parts = s.split(":", 1)
    if len(parts) == 2 and parts[0].strip().lower().startswith("day "):
        rest = parts[1].lstrip()
        if rest.lower().startswith(parts[0].lower() + ":"):
            second = rest.split(":", 1)
            if len(second) == 2:
                return f"{parts[0]}:{second[1]}"
    return s


def _format_weather_lines(weather_days: list[str]) -> list[str]:
    lines = []
    for idx, day in enumerate(weather_days, start=1):
        clean = _strip_dup_prefix(str(day))
        if clean.lower().startswith(f"day {idx}:"):
            desc = clean.split(":", 1)[1].strip()
        else:
            desc = clean
        if idx == 1:
            lines.append(f"Day 1 (Tomorrow): {desc}")
        else:
            lines.append(f"Day {idx}: {desc}")
    return lines


def _format_final_answer(parsed: dict) -> dict:
    tree = (parsed.get("tree_species") or "fall").strip().lower()
    days_left = parsed.get("estimated_days_color_will_last")
    changed_date = parsed.get("estimated_color_change_date")
    weather_days = parsed.get("weather_days") or []
    critical = parsed.get("weather_critical")

    lines = []
    lines.append(f"That's a beautiful {tree.capitalize()} tree at its absolute peak!")

    if isinstance(days_left, int):
        if critical and critical not in ("none noted", "weather_api_failed"):
            lines.append(
                f"Since {critical} is expected, which can speed up leaf drop, the AI predicts the color will last for {days_left} day{'s' if days_left != 1 else ''}."
            )
        else:
            lines.append(
                f"Based on the forecast and tree type, the AI predicts the color will last for {days_left} day{'s' if days_left != 1 else ''}."
            )
    else:
        lines.append("I could not compute how long the color will last because the weather data was incomplete.")

    if weather_days:
        lines.append("Weather forecast:")
        lines.extend(_format_weather_lines(weather_days))

    if critical and critical not in ("none noted", "weather_api_failed"):
        lines.append("I strongly recommend you take your photos tomorrow. After the weather event, the tree will likely be bare.")
    else:
        lines.append("I strongly recommend you take your photos soon while the color is still strong.")

    if changed_date:
        lines.append(f"(It likely changed color around {changed_date}.)")

    final_answer = "\n\n".join(lines)

    vp = parsed.get("video_prompt")
    if not vp:
        dur = days_left or 3
        vp = (
            f"Time-lapse of the uploaded fall tree over {dur} days, "
            f"color fade following the above 3-day weather, same camera angle, "
            f"apply '{critical or 'calm weather'}' on the specified day."
        )

    parsed["final_answer"] = final_answer
    parsed["video_prompt"] = vp
    return parsed


def _to_valid_payload(raw_text: str | None) -> dict | None:
    if not raw_text:
        return None

    def _extract_json_block_local(s: str) -> dict | None:
        if not s:
            return None
        if "```" in s:
            start = s.find("```json")
            if start == -1:
                start = s.find("```")
            if start != -1:
                end = s.find("```", start + 3)
                if end != -1:
                    candidate = s[start:end].replace("```json", "").replace("```", "").strip()
                    try:
                        return json.loads(candidate)
                    except Exception:
                        pass
        try:
            i, j = s.find("{"), s.rfind("}")
            if i != -1 and j != -1 and j > i:
                return json.loads(s[i:j+1])
        except Exception:
            pass
        try:
            return json.loads(s)
        except Exception:
            return None

    parsed = _extract_json_block_local(raw_text)
    if not isinstance(parsed, dict):
        return {
            "final_answer": "Model did not return JSON, so I cannot confirm it used the weather API.",
            "video_prompt": "Time-lapse of the uploaded fall tree.",
        }

    tree = parsed.get("tree_species")
    days_left = parsed.get("estimated_days_color_will_last")
    changed_date = parsed.get("estimated_color_change_date")
    weather_days = parsed.get("weather_days")
    critical = parsed.get("weather_critical")
    video_prompt = parsed.get("video_prompt")

    if not isinstance(weather_days, list) or len(weather_days) == 0:
        return {
            "final_answer": "Weather API data was not available in the model response, so I cannot give a reliable foliage estimate.",
            "video_prompt": "Time-lapse of the uploaded fall tree.",
        }

    parsed.setdefault("tree_species", (tree or "unknown"))
    parsed.setdefault("estimated_days_color_will_last", days_left if isinstance(days_left, int) else None)
    parsed.setdefault("estimated_color_change_date", changed_date if isinstance(changed_date, str) else None)
    parsed.setdefault("weather_critical", critical or "none noted")
    parsed.setdefault("video_prompt", video_prompt or None)

    return _format_final_answer(parsed)


def _inject_real_weather_and_finalize(result: dict, location: str, image: Image.Image | None = None) -> dict:
    if not result or not isinstance(result, dict):
        return result

    species = (result.get("tree_species") or "").strip().lower()
    if (not species or species == "unknown") and image is not None:
        species_guess = _classify_species_with_gemini(image)
        if species_guess:
            result["tree_species"] = species_guess

    if not result.get("weather_days"):
        wf = get_weather_forecast(location)
        if wf and wf.get("status") == "success":
            result["weather_days"] = wf.get("days", [])
            result["weather_critical"] = wf.get("critical_event", "none noted")
        else:
            result.setdefault("weather_days", [])
            result.setdefault("weather_critical", "weather_api_failed")

    if result.get("estimated_days_color_will_last") is None and result.get("weather_days"):
        base = 7
        crit = (result.get("weather_critical") or "").lower()
        if any(k in crit for k in ("storm", "heavy rain", "snow", "high wind", "strong wind", "gale")):
            base = max(1, base - 3)
        elif any(k in crit for k in ("rain", "wind")):
            base = max(1, base - 2)
        result["estimated_days_color_will_last"] = base

    return _format_final_answer(result)


def direct_gemini_forecast(question: str, location: str, image: Image.Image) -> dict | None:
    if GENAI_MODEL is None:
        return {"final_answer": "Error: Direct Gemini model is not initialized.", "video_prompt": None}

    image_part = _image_to_inline_part(image)
    schema_prompt = (
        "You are 'Folia'. Look at the tree image and location.\n"
        "Return ONLY one JSON object with keys:\n"
        "{\n"
        '  "tree_species": one of ["maple","oak","ginkgo","sweetgum","birch","beech","cherry","poplar","aspen","hickory","elm"] or "unknown",\n'
        '  "estimated_days_color_will_last": integer|null,\n'
        '  "estimated_color_change_date": string|null,\n'
        '  "weather_days": [],\n'
        '  "weather_critical": "weather_api_failed",\n'
        '  "video_prompt": string|null\n'
        "}\n"
        "Do NOT include extra text. Do not fabricate weather."
    )
    contents = [
        {
            "role": "user",
            "parts": [
                {
                    "text": schema_prompt + f"\n\nUser question: {question}\nLocation: {location}"
                },
                image_part,
            ],
        }
    ]

    try:
        resp = GENAI_MODEL.generate_content(
            contents=contents,
            generation_config={"response_mime_type": "application/json", "temperature": 0.2},
        )
        text = getattr(resp, "text", "") or ""
        return _to_valid_payload(text)
    except Exception as e:
        return {"final_answer": f"Error during Gemini fallback: {e}", "video_prompt": None}


async def run_agent_main(question: str, location: str, image: Image.Image):
    image_part = _image_to_inline_part(image)
    new_message = SimpleNamespace(
        role="user",
        parts=[{"text": f"User question: {question}\nThe location is: {location}"}, image_part],
    )

    raw_text, last_raw_obj = None, None
    debug_events, errors = [], []

    async for event in RUNNER.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=new_message):
        delta = getattr(event, "delta", None)
        if delta:
            piece = _to_text_from_adk_response(delta)
            if piece:
                raw_text = (raw_text or "") + piece

        err = getattr(event, "error", None)
        if err:
            errors.append(str(err))

        if getattr(event, "is_final_response", lambda: False)():
            last_raw_obj = getattr(event, "response", None)
            final_text = _to_text_from_adk_response(last_raw_obj)
            if final_text:
                raw_text = final_text

    if not raw_text and last_raw_obj is not None:
        raw_text = _to_text_from_adk_response(last_raw_obj) or str(last_raw_obj)

    payload = _to_valid_payload(raw_text)
    debug = {"raw_text": raw_text, "errors": errors, "events": debug_events[:10]}
    return payload, debug


def safe_generate_video_forecast(prompt: str, image: Image.Image):
    try:
        return _agent_generate_video_forecast(prompt, image)
    except (ResourceExhausted, PermissionDenied, NotFound) as e:
        print("[VIDEO] model unavailable:", e)
        return None
    except Exception as e:
        print("[VIDEO] unexpected error:", e)
        return None



# STREAMLIT UI
st.set_page_config(page_title=APP_NAME, page_icon="🍁", layout="wide")

st.title("🍁 FoliaScope")
st.markdown(
    """
**What this app does**

- Analyze a photo of fall trees (**upload a file or use your camera**) plus your location.  
- Combine **Gemini-2.5-flash** with a 3-day weather forecast from **Open Weather API** to estimate how long peak color will last and when it may fade.  
- Create a short **time-lapse** preview video using **VEO-3.1-fast** showing how the tree will look over the next three days.
"""
)

st.divider()
st.markdown("Upload a photo **or use your camera**")

question = "How long will this color last?"
location = st.text_input("Where did you take this photo?", "Atlanta, Georgia", key="location_val")


def _find_sample_dir() -> Path | None:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "folia_agent" / "img",
        here / "img",
        Path.cwd() / "folia_agent" / "img",
        Path.cwd() / "img",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return None


SAMPLE_DIR = _find_sample_dir()
exts = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

sample_img: dict[str, str | None] = {"<None>": None}
if SAMPLE_DIR:
    for f in sorted(SAMPLE_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in exts:
            sample_img[f.name] = str(f)

st.markdown("### Choose a photo source")
tab_upload, tab_camera, tab_samples = st.tabs(["Upload file", "Use camera", "Sample images"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload your photo:", type=["jpg", "jpeg", "png", "heic"], key="uploader"
    )

with tab_camera:
    camera_file = st.camera_input("Take a photo with your device camera", key="camera")

with tab_samples:
    sample = st.selectbox(
        label="Select image for classification here",
        options=list(sample_img.keys()),
        index=0,
        label_visibility="hidden",
    )

    with st.expander("Browse sample images"):
        if len(sample_img) > 1:
            keys = [k for k in sample_img.keys() if k != "<None>"]
            st.write("**Preview**")
            for key in keys:
                try:
                    st.image(sample_img[key], caption=key, use_column_width=True)
                except Exception:
                    st.caption(f"(unable to preview {key})")
        else:
            st.info("No sample images found in `folia_agent/img` or `img`.")

image, src_label = None, None
try:
    if camera_file is not None:
        image = Image.open(camera_file)
        src_label = "Camera photo"
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
        src_label = "Your upload"
    elif sample and sample != "<None>" and sample_img.get(sample):
        image = Image.open(sample_img[sample])
        src_label = f"Sample: {sample}"
except Exception:
    st.error("Could not open the image. If it's HEIC, install pillow-heif or upload a JPG/PNG.")
    st.stop()

if image is not None and question and location:
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption=src_label, use_column_width=True)
    with col2:
        if st.button("Generate Forecast", type="primary"):
            try:
                with st.spinner("Analyzing and forecasting…"):
                    try:
                        payload, debug = asyncio.run(run_agent_main(question, location, image))
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        payload, debug = loop.run_until_complete(run_agent_main(question, location, image))
                        loop.close()

                    result = _inject_real_weather_and_finalize(payload, location, image)

                needs_fallback = (
                    not result
                    or not isinstance(result, dict)
                    or not result.get("final_answer")
                )
                if needs_fallback:
                    with st.spinner("ADK was incomplete — trying direct Gemini fallback…"):
                        result = direct_gemini_forecast(question, location, image)
                        result = _inject_real_weather_and_finalize(result, location, image)

                if not result or not isinstance(result, dict):
                    st.error("No textual forecast. See debug below.")
                else:
                    st.success("**Forecast**")
                    st.markdown(result.get("final_answer") or "No answer.")

                    vp = result.get("video_prompt")
                    if vp:
                        st.success("**Time-lapse video** preview")
                        with st.spinner("🎬 Generating video (It will take upto 1-2 minutes.)"):
                            video_val = safe_generate_video_forecast(vp, image)
                        if video_val:
                            if isinstance(video_val, str) and video_val.startswith("data:video/mp4;base64,"):
                                st.video(base64.b64decode(video_val.split(",", 1)[1]))
                            else:
                                st.video(video_val)
                        else:
                            st.info("No video produced (preview / quota issue).")
                    else:
                        st.info("No video prompt produced by the agent.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)
