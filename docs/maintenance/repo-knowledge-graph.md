# Repo Knowledge Graph

This repo is wired for Graphify plus GitHub/Copilot instructions so future agents can recover the repository shape before editing.

Current graph:

- Report: `graphify-out/GRAPH_REPORT.md`
- Graph JSON: `graphify-out/graph.json`
- Browser view: `graphify-out/graph.html`
- Agent constitution (canonical): `AGENTS.md`
- Conductor workspace entry: `CONDUCTOR.md`
- Cursor rule: `.cursor/rules/kahlus-constitution.mdc`
- GitHub/Copilot instructions: `.github/copilot-instructions.md`

Local hook setup:

```bash
git config core.hooksPath .githooks
graphify update .
```

The tracked hooks run `graphify update .` after commits and branch switches. In this Codex worktree, Graphify's built-in `graphify hook install` does not work because `.git` is a pointer file, so the hook scripts live in `.githooks/`.

Agent rule:

Before architecture/codebase questions, read `graphify-out/GRAPH_REPORT.md` first. After code changes, run:

```bash
graphify update .
```
