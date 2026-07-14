# 🧹 SKILL SPECIFICATION: Refactoring & Cleanup

## 1. ARCHITEKTUR-PRINZIPIEN
- **Immutability First:** Keine in-place Mutation außer bei Performance-Pflicht (O(n)).
- **Backward Compatibility:** CLI-Interface (`main.py`) darf nicht brechen.
- **Deprecations:** Verwaiste Module nicht sofort löschen, sondern mit `warnings.warn("Deprecated: ...", DeprecationWarning)` markieren.

## 2. DEPRECATION-PROTOKOLL

1. Modul als `@deprecated` markieren
2. Alle Imports auf neues Modul umleiten
3. Einen Release-Zyklus warten
4. ERST dann löschen (HIP erforderlich!)

## 3. HIGH-RISK-TRIGGER (HIP)
- Löschen von `src/pipeline.py` oder `src/renderers/` → BACKUP-COMMIT erforderlich
- Ändern von Pydantic-Models in `src/types.py` → Alle Tests müssen passen
- Entfernen von CLI-Befehlen → Breaking Change

## 4. MODE_KIMI_DIRECT NOTE
Im Kimi Direct Mode werden Datei-Operationen über Kimis native Tools ausgeführt. Der Agent MUSS sicherstellen, dass Pfade relativ zum Workspace sind.
