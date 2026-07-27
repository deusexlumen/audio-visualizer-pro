# 📂 SKILL SPECIFICATION: System I/O & Version Control

## 1. FILE SYSTEM CONSTRAINTS
- **Isolation:** Dateizugriffe NUR auf `/src`, `/tests`, `/config`, `/memory`, `/sandbox`.
- **Destructive Operations (HIP-TRIGGER):** Löschen via `os.remove`/`shutil.rmtree` außerhalb `/sandbox` erfordert ZWINGEND `<human_intervention>` Tag. Code erst nach `<human_feedback>GRANTED</human_feedback>`.

## 2. VERSION CONTROL (Git Workflow)
- **Atomic Commits:** Nach Exit-Condition einer Phase MUSS commit-Befehl generiert werden.
- **Commit-Message Format:** `[PHASE_X] <Type>: <Beschreibung>`
- Erlaubte Types: `FEAT`, `FIX`, `REFACTOR`, `META`, `TEST`
- Beispiel: `[PHASE_2] TEST: Add GPU-Renderer unit tests with mocked ModernGL context`

## 3. DEPENDENCY MANAGEMENT
- VERBOTEN: `os.system('pip install...')` in Python-Skripten
- Abhängigkeiten ZWINGEND in `requirements.txt`
- Hinzufügen einer Zeile triggert HIP

## 4. MODE_KIMI_DIRECT NOTE
Im Kimi Direct Mode werden Datei-Operationen über Kimis native Tools ausgeführt (read_file, write_file, StrReplaceFile, Shell). Der Agent MUSS sicherstellen, dass Pfade relativ zum Workspace sind.
