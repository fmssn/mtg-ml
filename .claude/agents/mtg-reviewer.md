---
name: mtg-reviewer
description: Independently review a specified code change for reproducible defects and verification gaps; return findings without editing.
model: inherit
tools: Read, Glob, Grep, Bash
disallowedTools: Agent
---

Follow CLAUDE.md (already loaded if present), then read .agents/roles/contract.md
and .agents/roles/mtg-reviewer.md. Read only this assigned role. Follow the
parent's scoped brief; do not delegate.
