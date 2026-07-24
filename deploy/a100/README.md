# A100 / KTM cluster deploy assets

Operator handoff, Dockerfiles, and conda envs for A100/KTM runners live here so the repo root stays constitution + product README only.

| File | Role |
| --- | --- |
| `README_RUN.md` | Human run book (flattened to runner root on package) |
| `README_AGENT_DEPLOY.md` | Deployment-agent contract |
| `README_HANDOFF.md.in` | Template → `README_HANDOFF.md` |
| `AGENT_RUNBOOK.md` / `README_KTM_*` | KTM synthetic-only runner SOP |
| `Dockerfile.a100` / `Dockerfile.a100.runtime` | Dependency / runtime images |
| `environment-*.yml` | Conda env specs |

Build from repo root:

```bash
docker build -f deploy/a100/Dockerfile.a100 -t neurotwin-a100-runner:local .
docker build -f deploy/a100/Dockerfile.a100.runtime -t neurotwin-a100-runtime:local .
```

Package:

```bash
bash scripts/package_runner_bundle.sh
bash scripts/package_ktm_runner_bundle.sh
```
