# Faultline — LinkedIn demo kit

Assets and a recording guide for a ~60-second LinkedIn post aimed at **industrial / reliability engineering** audiences.

**Repo:** https://github.com/yashpadhye17/Faultline

---

## Assets in this repo

| File | Use |
|------|-----|
| [`docs/linkedin/faultline_architecture_linkedin.jpg`](linkedin/faultline_architecture_linkedin.jpg) | Architecture poster (end card or carousel slide 1) |
| [`docs/linkedin/00_four_copilots.jpg`](linkedin/00_four_copilots.jpg) | Four copilots overview slide |
| [`docs/linkedin/01_home.png`](linkedin/01_home.png) | Landing / hero screenshot |
| [`docs/linkedin/02_rca.png`](linkedin/02_rca.png) | Root cause analysis workspace |
| [`docs/linkedin/03_maintenance.png`](linkedin/03_maintenance.png) | Maintenance troubleshooting |
| [`docs/linkedin/04_metrics.png`](linkedin/04_metrics.png) | Evaluation / metrics dashboard |

If a screenshot file is missing, capture it yourself while the app runs on `http://localhost:8000` (see shot list below).

---

## LinkedIn caption (copy/paste)

Built **Faultline** — evidence-backed root-cause and maintenance copilots over NASA technical failure/repair reports.

Four workflows: RCA · maintenance troubleshooting · engineering comparison · research — with citations on every answer.

Stack: FastAPI, FAISS, cross-encoder reranking, Groq LLMs, evaluation dashboard.

Repo: https://github.com/yashpadhye17/Faultline

#ReliabilityEngineering #RootCauseAnalysis #RAG #IndustrialEngineering #MachineLearning

**First comment tip:** Paste the GitHub link again + one-line invite: “Happy to walk through the retrieval + eval pipeline.”

---

## Voiceover script (~60s)

Tone: calm, industrial, confident. Lead with reliability outcomes; mention RAG once.

> Faultline helps reliability engineers investigate failures with evidence, not guesswork.
>
> Ask a root-cause question over curated NASA technical failure and repair reports.
>
> Every answer comes back with citations you can open and verify.
>
> The same platform covers maintenance troubleshooting and engineering comparisons.
>
> Under the hood: retrieval, reranking, and grounded generation—plus an evaluation dashboard to measure answer quality.
>
> Faultline. Evidence-backed copilots for industrial reliability work.

### TTS option

Paste the script into CapCut, ElevenLabs, or macOS Spoken Content. Export voiceover as WAV/MP3 and lay it under a muted screen recording.

---

## Shot list (7 beats, 45–75s)

Prereqs: app running (`uvicorn` or Docker), Chrome fullscreen, bookmarks bar hidden, no personal tabs.

| Time | Screen | Action | VO line |
|------|--------|--------|---------|
| 0–8s | `/` home hero | Slow scroll or hold on headline + CTAs | Opening two sentences |
| 8–18s | `/rca` | Click a sample question chip (or paste demo question below) and submit | “Ask a root-cause question…” |
| 18–35s | `/rca` answer | Scroll to analysis + **citation / source pills**; pause 2–3s | “Every answer comes back with citations…” |
| 35–48s | `/maintenance` | One sample question → short answer | “The same platform covers maintenance…” |
| 48–58s | Metrics UI (`/metrics` or `/metrics-dashboard`) | Brief pan of dashboard / experiment strip | “…evaluation dashboard…” |
| 58–70s | Architecture JPG full screen | Show `faultline_architecture_linkedin.jpg` | Closing line |
| End | Black or architecture | On-screen text: GitHub URL | Silence or soft end |

### Demo questions (paste if chips are empty)

**RCA**

```text
Capacitors failed repeatedly during NEXT PPU testing. What root causes and recovery actions are reported?
```

**Maintenance**

```text
What early-warning signs indicate impending bearing failure, and what checks should technicians run first?
```

**Comparison (optional cut)**

```text
Compare top-down FMEA versus exhaustive component-level FMEA for complex electromechanical systems.
```

---

## How to record (Mac)

1. Start the app:

   ```bash
   # local
   export PYTHONPATH=.
   uvicorn main:app --host 0.0.0.0 --port 8000

   # or Docker (vectorstore mount recommended)
   docker run --rm -p 8000:8000 --env-file .env \
     -v "$(pwd)/data/vectorstore:/app/data/vectorstore" \
     faultline:local
   ```

2. Confirm `http://localhost:8000/healthz` returns ok. Prefer `readyz` with `vectorstore_loaded: true` so live answers work on camera.

3. Record with **Cmd+Shift+5** (macOS), **Loom**, or **OBS** — 1080p, browser window only.

4. Either speak the VO live, or mute mic and add TTS in CapCut / Descript.

5. Export **MP4**, ideally under ~100MB. Trim dead air; keep under 75s.

6. Post video **or** upload the **carousel** (architecture → home → RCA → maintenance → metrics) with the caption above.

---

## Carousel order (interactive alternative)

1. Architecture — `faultline_architecture_linkedin.jpg`
2. Four copilots — `00_four_copilots.jpg`
3. Home — `01_home.png`
4. RCA — `02_rca.png`
5. Maintenance — `03_maintenance.png`
6. Metrics — `04_metrics.png`

CTA in first comment: GitHub link.

---

## Attribution (keep accurate)

Data provided by NASA Scientific and Technical Information Program via the public NTRS API. This is **not** a NASA partnership or official NASA product.
