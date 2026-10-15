# 🍁 FoliaScope

**Snap a photo of a fall tree and find out how long its color will last.**

FoliaScope is an AI agent that identifies the tree in your photo, checks the real 3-day weather forecast for your location, estimates how many days of peak color are left, and renders a short time-lapse of how the tree will change.

Built at **AI ATL Hackathon 2025** · [Devpost](https://devpost.com/software/foliascope)

[![FoliaScope demo video](https://img.youtube.com/vi/7XeyjeyqIEA/maxresdefault.jpg)](https://youtu.be/7XeyjeyqIEA)

▶️ [Watch the demo](https://youtu.be/7XeyjeyqIEA)

---

## What it does

1. **You give it a photo and a location.** Upload a file, take one with your camera, or pick a sample image.
2. **The agent identifies the tree** (maple, oak, ginkgo, sweetgum, birch, and more) and how far past peak color it is.
3. **It pulls a real weather forecast** for the next 3 days from OpenWeather and flags rain, wind, storms, or snow, which strip leaves faster.
4. **It estimates how many days the color will last** and tells you when to go take your photos.
5. **It renders an 8-second time-lapse** with Veo 3.1 Fast, starting from your photo and following the forecast.

## How it works

```
photo + location
      │
      ▼
┌──────────────────────────────┐
│  "Folia" agent (Google ADK,   │
│   Gemini 2.5 Flash)           │
│                               │
│  1. predict_foliage_duration  │  species baseline × how far past peak
│  2. get_weather_forecast      │  OpenWeather, 3 days, flags critical events
│  3. search_google_for_context │  optional, Google Custom Search
└──────────────┬───────────────┘
               │ strict JSON (species, days left, weather, video prompt)
               ▼
   Streamlit app ── checks + fallbacks ──► forecast text
               │
               ▼
   Veo 3.1 Fast (Vertex AI, image-to-video) ──► 3-day time-lapse
```

**Design choices**

- **The weather comes from a tool, never from the model.** The agent must copy the forecast exactly from the OpenWeather tool. If the agent leaves it out, the app fetches it from OpenWeather itself, and if the weather service is down, the app says it can't estimate instead of guessing.
- **Species-specific baselines.** Each species has its own peak-color window (ginkgo drops fast, oaks hold longer). The remaining days scale with how far past peak the tree already is, then shrink when rain, wind, storms, or snow are coming.
- **Structured output.** The agent returns one JSON object, which the app validates before showing anything.
- **Fallbacks at every step.** If the agent's answer is incomplete, the app retries with a direct Gemini call. If the species is unclear, a second Gemini call picks from a fixed list. If the weather lookup by city fails, it retries with a US state suffix, then by latitude and longitude.
- **Video is optional.** Veo runs as a long-running job with retries and backoff. If it is unavailable or out of quota, you still get the forecast.

## Tech stack

| Layer | Tools |
|---|---|
| Agent | Google Agent Development Kit (ADK), Gemini 2.5 Flash, function tools |
| Video | Veo 3.1 Fast on Vertex AI (image-to-video) |
| Data | OpenWeather forecast and geocoding APIs, Google Custom Search (optional) |
| App | Streamlit, Python, Pillow |


## Limitations

- The days-remaining estimate is a heuristic (species baseline plus weather adjustment), not a trained model.
- The forecast looks 3 days ahead, so the estimate is most useful for trees at or near peak.
- Species recognition covers 11 common fall trees: maple, oak, ginkgo, sweetgum, birch, beech, cherry, poplar, aspen, hickory, and elm.
