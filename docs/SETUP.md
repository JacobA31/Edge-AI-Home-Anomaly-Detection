# First-time setup

Do this once. It takes about an hour. One person (the "repo admin") does Parts A to C; everyone does Part D.

The order matters a bit: push the code first, then create Azure resources, then give GitHub the
secrets. The first deploy runs **will fail (red ✗)** until Part C is done. That's expected.

---

## Part A – GitHub repository

### A1. Decide who owns the repo

Branch protection on a **private** repo needs GitHub Pro or Team. Pick one:

| Option | How |
|--------|-----|
| **Recommended:** private repo on one person's account | That person claims the free [GitHub Student Developer Pack](https://education.github.com/pack) (includes Pro), creates the repo, and adds the other four as collaborators. |
| Organization, public repo | Create a free organization (e.g. `sysc4907-e3`). Public repos get branch protection for free. Fine as long as no secrets or data are committed, which the rules already forbid. |
| Organization, private repo | Works, but branch protection won't be enforced on the free plan. Use only if the team trusts itself not to push to `main`. |

### A2. Create an empty repository

GitHub → **New repository**:
- Name: e.g. `e3-edge-ai-monitor`
- Private or public (see A1)
- **Do not** add a README, .gitignore or license (the template already has them)

### A3. Push the template

Unzip the template, then from inside the folder:

```bash
cd e3-edge-ai-monitor
git init -b main
git add .
git commit -m "chore: initial monorepo template"
git remote add origin https://github.com/<owner>/e3-edge-ai-monitor.git
git push -u origin main
```

Open the **Actions** tab. You'll see the four workflows start. Test/lint jobs should go green;
the two **deploy** jobs will fail because Azure isn't set up yet.

### A4. Add the team

- **Settings → Collaborators** (or the organization's **People**): invite the other four with *Write* access.
- Edit `.github/CODEOWNERS`: replace `@MEMBER1_USERNAME` etc. with real GitHub usernames.
  Commit this through a PR, as practice.

### A5. Protect `main`

**Settings → Branches → Add branch ruleset** (or *Add classic branch protection rule*) for `main`:
- ✅ Require a pull request before merging, with **1** approval
- ✅ Require review from Code Owners (optional, but useful)
- ✅ Block force pushes
- ❌ **Don't** add "Require status checks to pass". Our workflows only run when their folder
  changes. A required check that never starts blocks the PR forever. Reviewers check the
  ✓/✗ by eye instead.

### A6. Merge settings

**Settings → General → Pull Requests**:
- Allow **squash merging** only (untick merge commits and rebase)
- ✅ Automatically delete head branches

---

## Part B – Azure resources (Azure for Students subscription)

Create everything in the [Azure portal](https://portal.azure.com). Use **one resource group**
(e.g. `e3-rg`) and one region that your student subscription allows (e.g. Canada Central).
Deleting the resource group at the end of the course deletes everything at once.

### B1. PostgreSQL database

**Create → Azure Database for PostgreSQL – Flexible server**
- Workload type: *Development*. Compute: **Burstable, B1ms**. Storage: **32 GB**.
  This is the configuration covered by the 12-month free offer.
- Authentication: PostgreSQL authentication. Pick an admin username and password and save them in the team's password manager.
- Networking: **Public access**, ✅ *Allow public access from any Azure service within Azure*,
  and **+ Add current client IP address** (so you can connect from your laptop).
- After it's created: **Databases → + Add** → name `e3`.

Your connection string (keep it secret):

```
postgresql://<admin>:<password>@<server-name>.postgres.database.azure.com:5432/e3?sslmode=require
```

### B2. Function App (backend)

**Create → Function App → Flex Consumption**
- Name: e.g. `e3-api` (must be unique)
- Runtime: **Python**, version **3.11**
- Same resource group and region
- Leave the other settings on their defaults

After it's created:
1. **Settings → Environment variables → + Add**: `DATABASE_URL` = the connection string from B1. Apply.
2. **Settings → Configuration → General settings**: turn **SCM Basic Auth Publishing Credentials** to **On**. Save.
   (GitHub uses a publish profile to deploy, and that needs this setting.)
3. **Overview → Get publish profile** (called *Download publish profile* in some portal versions). You'll paste the contents of that file into GitHub in Part C.
4. **Overview → Default domain**: copy the full URL, e.g. `https://e3-api-xxxx.canadacentral-01.azurewebsites.net`.

> Why Flex Consumption and not "Consumption"? Microsoft now labels the Linux Consumption
> plan legacy, retiring in Sept 2028. Python only runs on Linux, and Microsoft recommends
> Flex Consumption for new apps. It still scales to zero.

### B3. Static Web App (frontend)

**Create → Static Web App**
- Plan: **Free**
- Deployment source: **Other** (important: choosing GitHub here makes Azure commit its own
  workflow file into the repo, which we don't want)

After it's created:
1. **Overview → Manage deployment token**: copy the token.
2. Copy the site URL, e.g. `https://happy-river-0abc.azurestaticapps.net`.
3. Back in the **Function App → API → CORS**, add that site URL. Save.

---

## Part C – Connect GitHub to Azure

In the repo: **Settings → Secrets and variables → Actions**.

**Secrets tab → New repository secret:**

| Name | Value |
|------|-------|
| `AZURE_FUNCTIONAPP_PUBLISH_PROFILE` | Entire contents of the publish profile file from B2 (open it in a text editor and copy all of it) |
| `AZURE_STATIC_WEB_APPS_API_TOKEN` | Deployment token from B3 |
| `AZURE_FUNCTIONAPP_MIGRATE_KEY` | Leave for now (step C3) |

**Variables tab → New repository variable:**

| Name | Value |
|------|-------|
| `AZURE_FUNCTIONAPP_NAME` | e.g. `e3-api` |
| `AZURE_FUNCTIONAPP_URL` | Full URL from B2, **no trailing slash** |

### C1. Deploy

**Actions → Backend (Azure Functions) → Run workflow**, then
**Actions → Frontend (Static Web Apps) → Run workflow**.
Both should go green. The backend shows a warning that migrations were skipped, which is fine for now.

### C2. Check it works

```bash
curl https://<your-function-app-url>/api/health      # {"status": "ok"}
```

Open the Static Web App URL. You should see "E3 Caregiver Dashboard – No events yet."

### C3. Turn on automatic database migrations

1. Azure portal → Function App → **Functions → migrate → Function Keys → + New key**, name `github`. Copy the value.
2. GitHub → add it as secret `AZURE_FUNCTIONAPP_MIGRATE_KEY`.
3. Re-run the backend workflow. The last step should print `{"applied": ["001_init"]}`.

From now on, every merged `backend/db/migrations/NNN_*.sql` file is applied automatically after deploy.

### C4. Key for the Raspberry Pi

Function App → **Functions → post_event → Function Keys → + New key**, name `pi-livingroom-01`.
Put it in `edge/config.yaml` on the Pi (see `edge/README.md`). One key per device, so a lost
device can be revoked without affecting the others.

---

## Part D – Everyone: clone and start working

```bash
git clone https://github.com/<owner>/e3-edge-ai-monitor.git
cd e3-edge-ai-monitor
```

Then follow the README in the folder you work on (`backend/`, `frontend/`, `edge/`, `ml/`),
and [CONTRIBUTING.md](../CONTRIBUTING.md) for branches and PRs. VS Code will suggest the
recommended extensions when you open the folder.

---

## Security notes before real data goes in

- `GET /api/events` is currently **anonymous** so the dashboard works out of the box. Protect it
  before real data goes in. The simplest route is **Authentication** (Easy Auth) on the Function
  App with Microsoft or GitHub login. The Static Web Apps Free plan's built-in login only
  protects the dashboard pages, not the separate Function App.
- A publish profile is a password. If someone leaves the team, **reset it** in the portal and
  update the GitHub secret. If your account is allowed to create *App registrations* in
  Microsoft Entra ID, you can switch to OIDC login later (no stored password). See the
  [Azure/functions-action README](https://github.com/Azure/functions-action).
- The PostgreSQL free offer lasts 12 months. **Delete the resource group** after the poster fair.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Backend deploy fails with **401** | SCM Basic Auth is off (B2 step 2), or the publish profile was reset. Download it again and update the secret. |
| Workflow didn't run after my merge | It only runs when its folder changes. Use **Run workflow** to start it by hand. |
| Dashboard shows "Can't reach the backend" | Check `AZURE_FUNCTIONAPP_URL` (no trailing slash) and the Function App's CORS list (B3 step 3). Rebuild the frontend after changing the variable. |
| Migration step fails | Check `DATABASE_URL` in the Function App and that "Allow Azure services" is on in the DB networking. Logs: Function App → Log stream. |
| PR stuck on "Expected – waiting for status" | A required status check was added in A5. Remove it. |
