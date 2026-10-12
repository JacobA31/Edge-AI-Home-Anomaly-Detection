# frontend/

Caregiver dashboard: React + TypeScript, built with Vite, hosted on **Azure Static Web Apps (Free)**.
Deployed by `.github/workflows/frontend.yml` on every merge to `main` that touches `frontend/` or `shared/`.

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, /api/* is proxied to the backend on :7071
npm run lint
npm run build      # output in dist/
```

- Event types in `src/types/event.ts` mirror `shared/event.schema.json`; keep them in sync.
- `public/staticwebapp.config.json` is copied into the build and tells Static Web Apps how to route.
- The API is the separate Function App, so its URL comes from `VITE_API_BASE_URL` at build time
  (set from the `AZURE_FUNCTIONAPP_URL` repository variable in CI).
