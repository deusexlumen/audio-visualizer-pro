# 🔀 TOOL & SKILL DISPATCHER — Audio Visualizer Pro

## ROUTING-LOGIK
Sobald [CURRENT_TASK] in agents.md definiert ist, MUSS der Orchestrator diese Matrix konsultieren und das entsprechende Modul aus /skills VOLLSTÄNDIG laden, BEVOR Code generiert wird.

## ENTSCHEIDUNGSMATRIX

| Task-Charakteristik | Skill-Modul | Risiko (HIP-Trigger) |
|---|---|---|
| **Testing & Coverage:** pytest, mocks, fixtures, assertions | `skill_testing.md` | LOW |
| **GPU/OpenGL Code:** moderngl, shader, framebuffer | `skill_gpu_development.md` | MEDIUM (HIGH bei Context-Wechsel) |
| **System I/O & Git:** Dateien/Ordner erstellen, Shell, pip, git | `skill_filesystem_and_git.md` | HIGH bei `rm`, `pip install` |
| **Refactoring & Cleanup:** Code entfernen, APIs ändern | `skill_refactoring.md` | HIGH (HIP vor Löschen) |
| **Self-Healing & Prompt-Bugfix:** Wiederkehrender Error, A/B-Test | `skill_meta_evolution.md` | HIGH (HIP vor Skill-Überschreiben) |

> **FALLBACK-REGEL:** Passt eine Aufgabe in KEINE Kategorie → `<human_intervention>` generieren und Nutzer um neues Skill-Modul bitten. "Guessing" ist STRENGSTENS verboten.

## TOOL CONTRACT (I/O Primitives)
Im MODE_KIMI_DIRECT stellt Kimi Code CLI folgende Tools nativ bereit:

- `read_file(path: str) → str`
- `write_file(path: str, content: str) → bool` (überschreibt komplett)
- `StrReplaceFile(path: str, old: str, new: str) → bool`
- `Shell(command: str) → str`
- `SearchWeb(query: str) → dict`
- `FetchURL(url: str) → str`
