# 🧬 SKILL SPECIFICATION: Prompt Evolution & Self-Healing

## 1. KOGNITIVE ANALYSE (Root Cause Analysis)
Im Meta-Zustand schreibst du KEINEN Ausführungscode. Aufgabe: Framework-Analyse.

1. Lies Fehler-Stacktrace aus `memory/temp.md`
2. Lies verantwortliche Skill-Datei (z.B. `skill_gpu_development.md`)
3. Identifiziere Ambiguitäten: Spielraum für Interpretation? Fehlt Negative Constraint?

## 2. MUTATION PROTOCOL (A/B Test Draft)
NIEMALS Original sofort überschreiben (Korruptionsgefahr).

1. Generiere VERBESSERTE, präzisere Version der skill_*.md
2. Speichere Entwurf in `<mutation_draft>` und schreibe nach `memory/temp_mutant.md`
3. Evaluiere Mutant gegen Fehlerursache

## 3. ADOPTION PROTOCOL
Wenn Mutant erfolgreich:
1. `OVERWRITE(target="skill_*.md", source="temp_mutant.md")`
2. Update `[CONTEXT MEMORY]` in agents.md mit Evolution-Grund
3. Generiere `[META]` Commit

## 4. MODE_KIMI_DIRECT NOTE
Im Kimi Direct Mode wird A/B-Testing durch manuelle Validierung ersetzt:
1. Mutant wird in `memory/temp_mutant.md` geschrieben
2. Kimi evaluiert den Mutant gegen die Fehlerursache
3. Bei Erfolg: StrReplaceFile auf das Original-Skill
4. Kein automatischer Sandbox-Execution-Loop nötig
