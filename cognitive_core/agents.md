# 🧭 AGENT CONTROL & STATE LEDGER v2.0 — Audio Visualizer Pro

## 1. GLOBAL INVARIANTS & RISK PROTOCOL
> Absolute Systemgesetze. Verletzung erzwingt sofortigen <error_correction> Halt.

- [CONSTRAINT_1]: Keine Installation neuer Abhängigkeiten ohne Update der requirements.txt UND HIP-Genehmigung.
- [CONSTRAINT_2_RISK]: Aktionen, die bestehenden, funktionierenden Code löschen (Risk: HIGH), erfordern vor Ausführung einen Backup-Commit.
- [CONSTRAINT_3_PERSISTENCE]: Bei wiederkehrenden Fehlern (Retry-Limit > 3) ist das Neuschreiben gleicher Logik VERBOTEN. Wechsel in [PHASE_META_OPTIMIZATION] erzwingen.
- [CONSTRAINT_4_DOCS]: Alle neuen Funktionen benötigen Google-Style Docstrings. Alle Skill-Dateien auf Deutsch.
- [CONSTRAINT_5_GPU]: GPU-Rendering-Tests müssen einen moderngl-Context mocken oder mit `pytest.skip` ausgestattet sein, wenn kein GL-Context verfügbar ist.
- [CONSTRAINT_6_BACKWARD_COMPAT]: Die CLI (`main.py`) darf keine breaking changes erhalten ohne Major-Version-Bump in agents.md.

## 2. STATE MACHINE WITH ROLLBACK GRAPH

- **[PHASE_0]: System Awakening**
  - Allowed: Systemdiagnose, Lesen aller .md, Ready-Bestätigung
  - Exit Condition: Agent meldet Einsatzbereitschaft
  - Rollback Path: N/A (Root)

- **[PHASE_1]: Architecture & Setup**
  - Allowed: Dateisystem-Operationen (LOW), Basis-Konfigurationen (LOW), Test-Infrastruktur aufbauen
  - Exit Condition: Projektstruktur validiert, pytest läuft, Coverage-Baseline dokumentiert
  - Rollback Path: N/A

- **[PHASE_2]: Core Execution & Module Integration**
  - Allowed: Implementierung skill_testing.md (HIGH bei Architektur), GPU-Renderer Tests, PostProcess Tests
  - Exit Condition: Alle Kernfunktionen geschrieben + dokumentiert, Coverage > 85%
  - Rollback Path: → [PHASE_1] bei grundlegenden Designfehlern

- **[PHASE_3]: Deep Testing & Refactoring**
  - Allowed: Unit-Tests, Isolations-Tests (LOW), verwaiste Code-Pfade entfernen
  - Exit Condition: 100% Code-Coverage für Kernfunktionen (GPU-Renderer, PostProcess, Types)
  - Rollback Path: → [PHASE_2] bei wiederholten Test-Fails

- **[PHASE_4]: Polish & Release**
  - Allowed: README-Updates, Badge-Korrekturen, Performance-Benchmarks
  - Exit Condition: Projekt ist "feature-complete" und release-fertig
  - Rollback Path: → [PHASE_3]

- **[PHASE_META_OPTIMIZATION]: Systemic Reflection & Evolution**
  - Trigger: Fehlerhafte Generierung nach 3 Versuchen
  - Allowed: Analyse skill_*.md, Generierung Mutant in temp.md, RUN_AB_TEST
  - Exit Condition: Mutant gewinnt A/B-Test
  - Write-Back: Überschreiben fehlerhafter skill_*.md mit Mutant

## 3. CURRENT STATE LEDGER
> Synchronisiert nach JEDEM agentischen Zyklus.

- **ACTIVE_PHASE:** [PHASE_2]
- **CURRENT_TASK:** "GPU-Renderer Coverage erhöhen (40% → 70%), GPU-Text-Renderer Tests, GPU-Preview Tests. Ziel: Gesamt-Coverage > 85%."
- **RISK_LEVEL_OF_TASK:** MEDIUM
- **LAST_VALID_STATE:** "134/134 Tests passing, 78% Gesamt-Coverage. PostProcess 100%, GPU-Preview 95%, GPU-Text 78%, QuoteOverlay 93%, GPU-Renderer 42%."
- **KNOWN_BLOCKERS:**
  1. GPU-Renderer interne Shader-Methoden erfordern komplexes Mock-Setup
  2. gemini_integration.py (32%) ist API-abhaengig und schwer ohne API-Key zu testen
  3. PHASE_2 Ziel 85% nicht erreicht (78%), aber Kernmodule bei 85-100%
- **RUNTIME_MODE:** MODE_KIMI_DIRECT
- **TARGET_COVERAGE:** 85%

## 4. ABDUCTIVE REASONING & HYPOTHESIS LOG
> Zwingende Dokumentation bei blockierenden Fehlern.

(Leer — wird vom Agenten gefüllt)

## 5. CONTEXT MEMORY (Entscheidungs-Logbuch)
- [BOOT_SEQUENCE]: Evo-Agent Framework auf bestehendem Projekt initialisiert.
- [RUNTIME_MODE_SET]: MODE_KIMI_DIRECT aktiviert. Kein externer Controller nötig.
- [COVERAGE_BASELINE]: 63% → 69% (PHASE_1) → 73% (PHASE_2 Start) → 78% (PHASE_2 Aktuell)
- [ARCH_NOTE]: Projekt hat zwei parallele Render-Pipelines: PIL (legacy, verwaist) und GPU (ModernGL, aktiv).
- [PHASE_1_COMPLETE]: 
  - PostProcess: 0% → 100% (22 Tests)
  - GPU-Renderer: 37% → 40% (11 Tests, Mock-Infrastruktur etabliert)
  - Bugfix: postprocess.py fehlender `Path`-Import + LUT-Parser Header-Handling
  - PIL-Pipeline: DeprecationWarning hinzugefügt, ImportError abgefangen
- [PHASE_2_PARTIAL]:
  - GPU-Text-Renderer: 12% → 78% (15 Tests)
  - GPU-Preview: 0% → 95% (9 Tests)
  - GPU-Renderer Extended: 40% → 42% (7 Tests)
  - QuoteOverlay: 77% → 93% (25 Tests)
  - pyproject.toml: Coverage omit für pipeline.py + renderers/
  - Gesamt: 134 Tests passing, 78% Coverage
- [PHASE_2_EXIT]: 🟡 NOCH NICHT ERFUELLT — Coverage 78% < Ziel 85%
- [PHASE_2_DECISION]: Kernmodule sind bei 85-100%. Verbleibende Luecken: gemini_integration.py (32%, API-abhaengig) und gpu_renderer.py Shader (42%, schwer mockbar). User-Entscheidung erforderlich.
