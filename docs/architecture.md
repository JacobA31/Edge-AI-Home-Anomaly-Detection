# Architecture

See proposal Figure 1.7.1. How it maps to this repo:

```
 Microphone (ReSpeaker XVF3800)
        │ USB audio, 16 kHz
        ▼
 Raspberry Pi 5 ── edge/ ─────────────────────────────┐
   capture → features → fall model + baseline        │
   → event record → SQLite outbox                    │  raw audio never leaves this box
        │ HTTPS POST /api/events (function key)       │
        ▼                                             ┘
 Azure Functions ── backend/ ── PostgreSQL (backend/db/migrations)
        │ GET /api/events
        ▼
 Static Web App ── frontend/ ── caregiver browser

 ml/      trains the models offline → .tflite → GitHub Release → Pi
 shared/  event.schema.json, the contract between edge, backend and frontend
```
