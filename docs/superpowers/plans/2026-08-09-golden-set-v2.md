# Golden Set v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Das Golden-Set für die Studio-Schwellenkalibrierung von Ein-Audio/Speech-only/Konstruktions-Labels auf Multi-Audio (3 Musik + 3 Podcast), beide Modi und einen menschlichen Labeling-Workflow heben.

**Architecture:** Ein Korpus-Manifest (`tests/golden/corpus.json`, Schema `golden-corpus/1`) beschreibt die Referenz-Audios mit id/mode/description/source. `tools/build_golden_set.py` loopt über das Korpus (oder ein einzelnes `--audio` wie bisher), schreibt `labels.json` im Schema `golden-set/2` (mit `human_label: null`, `audio`, `mode`, `frame`) und erzeugt ein Contact Sheet pro Audio. `tools/calibrate_thresholds.py` bevorzugt menschliche Labels über Konstruktions-Labels. Ein neues `tools/label_golden.py` setzt Human-Labels per fnmatch-Muster.

**Tech Stack:** Python 3.11, pytest, Pillow, numpy, ffmpeg (system). Keine neuen Dependencies.

## Global Constraints

- Sprache: Code-Kommentare, Docstrings, Docs und Commit-Messages auf Deutsch (AGENTS.md).
- Keine neuen pip-Dependencies; nur stdlib + bereits vorhandene (PIL, numpy, click).
- Windows-Konsole: `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` in CLI-Mains beibehalten; keine "→"/"≤"-Zeichen in print-Ausgaben neuer Tools (cp1252).
- `tools/` ist ein Package (`tools/__init__.py` existiert) — Tests importieren via `from tools.xxx import ...`.
- **Keine `git commit`-Aufrufe.** Tasks enden mit einem grünen pytest-Checkpoint. Commits macht der Maintainer separat.
- Audio-Assets gehören nach `tests/golden/audio/` (dieser Pfad ist NICHT gitignored; `output/` und `assets/user_uploads/` sind es).
- labels.json-Schema `golden-set/2` muss rückwärtskompatibel lesbar bleiben: Top-Level `renders` mit `id`, `good`, `metrics` bleiben bestehen (calibrate_thresholds.py liest sie).
- Matrix bleibt: 6 Visualizer × ALPHA_CAPS {0.3, 0.6, 1.0} × MASK_VARIANTS {False, True} = 36 Varianten pro Audio, 216 gesamt.

---

### Task 1: Korpus-Manifest + Loader

**Files:**
- Create: `tools/golden_corpus.py`
- Create: `tests/golden/corpus.json`
- Test: `tests/test_golden_corpus.py`

**Interfaces:**
- Produces: `load_corpus(path) -> list[dict]` (Audio-Dicts mit absolut aufgelöstem `"path"`), `missing_audio_files(audios) -> list[str]`, `CorpusError(ValueError)`, `VALID_MODES = {"music", "podcast", "hybrid"}`. Task 3 konsumiert `load_corpus`/`missing_audio_files`; Task 2's Repo-Test konsumiert beide.
- Audio-Dict-Felder: `id` (str), `path` (str, absolut nach load), `mode` ("music"|"podcast"|"hybrid"), `description` (str), `source` (str).

- [ ] **Step 1: Failing Tests schreiben** — `tests/test_golden_corpus.py`:

```python
"""Tests für das Golden-Set-Korpus-Manifest (golden-corpus/1)."""

import json
from pathlib import Path

import pytest

from tools.golden_corpus import CorpusError, load_corpus, missing_audio_files


def _write_manifest(tmp_path, audios):
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps({"version": "golden-corpus/1", "audios": audios}),
                 encoding="utf-8")
    return p


def _entry(**kw):
    base = {"id": "a1", "path": "a.m4a", "mode": "music",
            "description": "d", "source": "s"}
    base.update(kw)
    return base


def test_load_corpus_resolves_paths(tmp_path):
    (tmp_path / "a.m4a").write_bytes(b"x")
    p = _write_manifest(tmp_path, [_entry()])
    audios = load_corpus(p)
    assert audios[0]["id"] == "a1"
    assert audios[0]["path"] == str((tmp_path / "a.m4a").resolve())


def test_load_corpus_rejects_bad_mode(tmp_path):
    p = _write_manifest(tmp_path, [_entry(mode="jazz")])
    with pytest.raises(CorpusError, match="mode"):
        load_corpus(p)


def test_load_corpus_rejects_duplicate_ids(tmp_path):
    p = _write_manifest(tmp_path, [_entry(), _entry()])
    with pytest.raises(CorpusError, match="oppel"):
        load_corpus(p)


def test_load_corpus_rejects_missing_field(tmp_path):
    p = _write_manifest(tmp_path, [_entry(description="")])
    with pytest.raises(CorpusError, match="description"):
        load_corpus(p)


def test_missing_audio_files(tmp_path):
    (tmp_path / "da.m4a").write_bytes(b"x")
    p = _write_manifest(tmp_path, [_entry(id="da", path="da.m4a"),
                                   _entry(id="fehlt", path="fehlt.m4a")])
    audios = load_corpus(p)
    assert missing_audio_files(audios) == ["fehlt"]
```

- [ ] **Step 2: Tests laufen lassen, FAIL prüfen**

Run: `pytest tests/test_golden_corpus.py -v`
Expected: FAIL (`ModuleNotFoundError: tools.golden_corpus`)

- [ ] **Step 3: Loader implementieren** — `tools/golden_corpus.py`:

```python
"""Golden-Set-Korpus: Manifest laden und validieren (Spec studio-spec/2.1 §3.5).

Das Manifest (Schema ``golden-corpus/1``) beschreibt die Referenz-Audios
für das Golden Set: id, Pfad (relativ zum Manifest), Modus und eine
menschliche Begründung (description/source), warum die Datei im Korpus ist.
"""

import json
from pathlib import Path

VALID_MODES = {"music", "podcast", "hybrid"}
REQUIRED_FIELDS = ("path", "description", "source")


class CorpusError(ValueError):
    """Ungültiges Korpus-Manifest."""


def load_corpus(path):
    """Lädt corpus.json und validiert das Schema.

    Rückgabe: Liste der Audio-Dicts; ``path`` ist dabei absolut aufgelöst
    (relativ zum Manifest-Verzeichnis).
    """
    manifest = Path(path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("version") != "golden-corpus/1":
        raise CorpusError(f"Unbekannte Version: {data.get('version')!r}")
    audios = data.get("audios")
    if not isinstance(audios, list) or not audios:
        raise CorpusError("Manifest braucht nicht-leere Liste 'audios'.")
    seen: set[str] = set()
    out: list[dict] = []
    for entry in audios:
        eid = entry.get("id")
        if not eid or not isinstance(eid, str):
            raise CorpusError(f"Audio ohne gueltige id: {entry!r}")
        if eid in seen:
            raise CorpusError(f"Doppelte Audio-id: {eid!r}")
        seen.add(eid)
        mode = entry.get("mode")
        if mode not in VALID_MODES:
            raise CorpusError(
                f"{eid}: ungueltiger mode {mode!r} (erlaubt: {sorted(VALID_MODES)})")
        for field in REQUIRED_FIELDS:
            if not entry.get(field):
                raise CorpusError(f"{eid}: Feld {field!r} fehlt oder leer.")
        resolved = (manifest.parent / entry["path"]).resolve()
        out.append({**entry, "path": str(resolved)})
    return out


def missing_audio_files(audios):
    """Liste der ids, deren Datei nicht existiert."""
    return [a["id"] for a in audios if not Path(a["path"]).is_file()]
```

- [ ] **Step 4: Manifest anlegen** — `tests/golden/corpus.json`:

```json
{
  "version": "golden-corpus/1",
  "audios": [
    {
      "id": "music_severance",
      "path": "audio/music_severance.m4a",
      "mode": "music",
      "description": "Aggressiver Electronica-Album-Opener, treibende Beats, viele Transienten.",
      "source": "output/01 - Severance.mp4, Audiospur 30-120s (eigene Produktion Deus ex Lumen)"
    },
    {
      "id": "music_dunkelheit",
      "path": "audio/music_dunkelheit.m4a",
      "mode": "music",
      "description": "Dunkler, ruhigerer Track — wenig Transienten, Flächen; Gegenpol zu Severance.",
      "source": "output/12 - Dunkelheit.mp4, Audiospur 30-120s"
    },
    {
      "id": "music_velvet",
      "path": "audio/music_velvet.m4a",
      "mode": "music",
      "description": "Midtempo-Electronica mit melodischen Flächen, längster Track.",
      "source": "output/13 - Velvet Pressure.mp4, Audiospur 30-120s"
    },
    {
      "id": "podcast_macy",
      "path": "audio/podcast_macy.m4a",
      "mode": "podcast",
      "description": "Deutschsprachiger Monolog/Vortrag, Studiomikro, ruhiges Sprechen (bisherige Golden-Set-Referenz).",
      "source": "assets/user_uploads/Die_Macy-Stiftung_als_Fundament_der_Kontrolle.m4a, 60-150s"
    },
    {
      "id": "podcast_gorilla",
      "path": "audio/podcast_gorilla.m4a",
      "mode": "podcast",
      "description": "Eigene Podcast-Episode (aus projects/podcast.json), Einstiegspassage.",
      "source": "Downloads/Digitale_Gorilla-Taktiken_gegen_den_KI-Müll.m4a, 60-150s"
    },
    {
      "id": "podcast_gorilla_mid",
      "path": "audio/podcast_gorilla_mid.m4a",
      "mode": "podcast",
      "description": "Mittlerer Abschnitt derselben Episode — andere Passage/Dynamik. Limitation: gleiche Stimmen wie podcast_gorilla; Korpus soll mit künftigen Episoden wachsen.",
      "source": "Downloads/Digitale_Gorilla-Taktiken_gegen_den_KI-Müll.m4a, Mitte der Episode, 90s"
    }
  ]
}
```

- [ ] **Step 5: Tests grün**

Run: `pytest tests/test_golden_corpus.py -v`
Expected: 5 PASS

- [ ] **Step 6: Checkpoint** — `pytest tests/test_golden_corpus.py tests/test_studio_calibration.py -q` muss grün sein. Kein Commit (Maintainer-Sache).

---

### Task 2: Audio-Exzerpte erzeugen

**Files:**
- Create: `tests/golden/audio/*.m4a` (6 Dateien, via ffmpeg)
- Test: `tests/test_golden_corpus.py` (Repo-Manifest-Test ergänzen)

**Interfaces:**
- Consumes: `load_corpus`, `missing_audio_files` aus Task 1; Pfade aus `tests/golden/corpus.json`.
- Produces: die 6 Audiodateien, auf die das Manifest zeigt; Task 3 und 6 brauchen sie.

- [ ] **Step 1: Failing Repo-Test ergänzen** — ans Ende von `tests/test_golden_corpus.py`:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_repo_manifest_complete():
    """Das eingecheckte Manifest ist valide und alle Audiodateien existieren."""
    audios = load_corpus(REPO_ROOT / "tests" / "golden" / "corpus.json")
    assert len(audios) == 6
    modes = [a["mode"] for a in audios]
    assert modes.count("music") == 3
    assert modes.count("podcast") == 3
    assert missing_audio_files(audios) == []
```

Run: `pytest tests/test_golden_corpus.py::test_repo_manifest_complete -v`
Expected: FAIL (`missing_audio_files` == alle 6 ids)

- [ ] **Step 2: Exzerpte schneiden** (ffmpeg `-c:a copy`, kein Re-Encode):

```bash
mkdir -p tests/golden/audio
ffmpeg -y -v error -ss 30 -t 90 -i "output/01 - Severance.mp4" -vn -c:a copy tests/golden/audio/music_severance.m4a
ffmpeg -y -v error -ss 30 -t 90 -i "output/12 - Dunkelheit.mp4" -vn -c:a copy tests/golden/audio/music_dunkelheit.m4a
ffmpeg -y -v error -ss 30 -t 90 -i "output/13 - Velvet Pressure.mp4" -vn -c:a copy tests/golden/audio/music_velvet.m4a
ffmpeg -y -v error -ss 60 -t 90 -i "assets/user_uploads/Die_Macy-Stiftung_als_Fundament_der_Kontrolle.m4a_33410687_Die_Macy-Stiftung_als_Fundament_der_Kontrolle.m4a" -c:a copy tests/golden/audio/podcast_macy.m4a
ffmpeg -y -v error -ss 60 -t 90 -i "/c/Users/Buxe/Downloads/Digitale_Gorilla-Taktiken_gegen_den_KI-Müll.m4a" -c:a copy tests/golden/audio/podcast_gorilla.m4a
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "/c/Users/Buxe/Downloads/Digitale_Gorilla-Taktiken_gegen_den_KI-Müll.m4a")
MID=$(python -c "print(int(float('$D') / 2))")
ffmpeg -y -v error -ss "$MID" -t 90 -i "/c/Users/Buxe/Downloads/Digitale_Gorilla-Taktiken_gegen_den_KI-Müll.m4a" -c:a copy tests/golden/audio/podcast_gorilla_mid.m4a
```

- [ ] **Step 3: Exzerpte verifizieren** — jede Datei muss ≥ 80s haben:

```bash
for f in tests/golden/audio/*.m4a; do
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
  echo "$f: $d"
done
```
Expected: 6 Zeilen, alle ≥ 80.0

- [ ] **Step 4: Tests grün**

Run: `pytest tests/test_golden_corpus.py -v`
Expected: 6 PASS

---

### Task 3: build_golden_set.py Multi-Audio + labels.json v2

**Files:**
- Modify: `tools/build_golden_set.py` (komplettes main()-Refactoring + 2 neue Helfer + geändertes build_contact_sheet)
- Test: `tests/test_build_golden_set.py` (neu)

**Interfaces:**
- Consumes: `load_corpus`, `missing_audio_files` (Task 1, `tools.golden_corpus`).
- Produces: `variant_id(audio_id, viz_name, cap, use_mask) -> str`, `make_label_entry(vid, audio_entry, viz_name, cap, use_mask, metrics, good) -> dict`, geändertes `build_contact_sheet(frame_base, entries, out_path, cols=6, thumb_w=320)` (liest `entry["frame"]` relativ zu `frame_base`, zeigt HUMAN-Label falls gesetzt). labels.json v2: Top-Level `{"version": "golden-set/2", "construction_labels": true, "human_labels_pending": true, "renders": [...]}`; Render-Eintrag zusätzlich: `human_label` (null), `audio` (id), `mode`, `frame` (relativer Pfad). Task 4/5 konsumieren `human_label` und `audio`/`mode`.

- [ ] **Step 1: Failing Tests schreiben** — `tests/test_build_golden_set.py`:

```python
"""Tests für die Multi-Audio-Helfer des Golden-Set-Builders."""

from pathlib import Path

from PIL import Image

from tools.build_golden_set import (build_contact_sheet, make_label_entry,
                                    variant_id)


def test_variant_id():
    assert (variant_id("music_severance", "pulsing_core", 0.3, False)
            == "music_severance__pulsing_core_cap03")
    assert (variant_id("podcast_macy", "voice_flow", 1.0, True)
            == "podcast_macy__voice_flow_cap10_mask")


def test_make_label_entry_schema():
    audio = {"id": "a1", "mode": "music"}
    metrics = {"M1": 0.01, "M3": None, "M4": None, "M5": 0.05}
    e = make_label_entry("a1__viz_cap03", audio, "viz", 0.3, False,
                         metrics, good=True)
    assert e["id"] == "a1__viz_cap03"
    assert e["good"] is True
    assert e["human_label"] is None
    assert e["audio"] == "a1"
    assert e["mode"] == "music"
    assert e["metrics"]["M3"] is None
    assert "construction_note" in e


def test_contact_sheet_uses_frame_paths_and_human_tag(tmp_path):
    frame_dir = tmp_path / "frames" / "a1"
    frame_dir.mkdir(parents=True)
    entries = []
    for i in range(4):
        name = f"v{i}.png"
        Image.new("RGB", (854, 480), (i * 60, 0, 0)).save(frame_dir / name)
        entries.append({
            "id": f"a1__v{i}",
            "frame": f"frames/a1/{name}",
            "good": i % 2 == 0,
            "human_label": "good" if i == 0 else None,
        })
    out = tmp_path / "sheet.png"
    build_contact_sheet(tmp_path, entries, out, cols=2)
    assert out.is_file()
    assert Image.open(out).size[0] == 2 * 320
```

- [ ] **Step 2: Tests laufen lassen, FAIL prüfen**

Run: `pytest tests/test_build_golden_set.py -v`
Expected: FAIL (`ImportError: cannot import name 'variant_id'`)

- [ ] **Step 3: Implementieren** — in `tools/build_golden_set.py`:

a) Docstring oben ersetzen durch:

```python
"""Baut das Golden-Set für die Studio-Schwellenkalibrierung (Spec §3.5, §19).

Sweeped pro Referenz-Audio aus dem Korpus (``tests/golden/corpus.json``)
eine Matrix aus Visualizer × alpha_cap × Subjekt-Maske. Pro Variante werden
die Probe-Metriken (M1, M3, M4, M5) via ``evaluate_params`` gemessen, ein
Vorschau-Frame als PNG abgelegt und ein Konstruktions-Label vergeben
(good = alpha_cap ≤ 0.6). Ergebnis: ``tests/golden/labels.json`` (Schema
golden-set/2, mit leeren ``human_label``-Feldern) plus ein Contact Sheet
pro Audio zur visuellen Kontrolle und menschlichen Label-Vergabe.
"""
```

b) Import ergänzen (nach den bestehenden src-Imports):

```python
from tools.golden_corpus import load_corpus, missing_audio_files
```

c) Nach `_cap_token` die zwei neuen Helfer einfügen:

```python
def variant_id(audio_id: str, viz_name: str, cap: float,
               use_mask: bool) -> str:
    """Eindeutige Varianten-ID: '<audio>__<viz>_capXX[_mask]'."""
    vid = f"{audio_id}__{viz_name}_{_cap_token(cap)}"
    if use_mask:
        vid += "_mask"
    return vid


def make_label_entry(vid: str, audio_entry: dict, viz_name: str,
                     cap: float, use_mask: bool, metrics: dict,
                     good: bool) -> dict:
    """Baut einen labels.json-v2-Eintrag (Konstruktions-Label, Human-Feld leer)."""
    note = (
        f"Konstruktions-Label: alpha_cap={cap} "
        f"{'<=' if good else '>'} {GOOD_CAP_MAX} -> "
        f"{'good' if good else 'bad'}; "
        f"Maske {'Zentrums-Gauß' if use_mask else 'keine'}."
    )
    return {
        "id": vid,
        "good": good,
        "human_label": None,
        "audio": audio_entry["id"],
        "mode": audio_entry["mode"],
        "visualizer": viz_name,
        "alpha_cap": cap,
        "mask": use_mask,
        "metrics": {
            "M1": metrics["M1"],
            "M3": metrics["M3"],
            "M4": metrics["M4"],
            "M5": metrics["M5"],
        },
        "construction_note": note,
    }
```

d) `build_contact_sheet` komplett ersetzen durch:

```python
def build_contact_sheet(frame_base: Path, entries: list[dict],
                        out_path: Path, cols: int = 6,
                        thumb_w: int = 320) -> None:
    """Grid aller Vorschau-Frames mit ID + Label unter jedem Frame.

    ``entries[i]["frame"]`` ist der PNG-Pfad relativ zu ``frame_base``.
    Ein gesetztes ``human_label`` schlägt das Konstruktions-Label in der
    Anzeige (HUMAN: GOOD/BAD vs. CONSTR: GOOD/BAD).
    """
    label_h = 34
    thumb_h = int(thumb_w * PROBE_H / PROBE_W)
    rows = (len(entries) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)),
                      (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for i, entry in enumerate(entries):
        img = Image.open(frame_base / entry["frame"]).convert("RGB")
        img = img.resize((thumb_w, thumb_h), Image.LANCZOS)
        x, y = (i % cols) * thumb_w, (i // cols) * (thumb_h + label_h)
        sheet.paste(img, (x, y))
        human = entry.get("human_label")
        if human in ("good", "bad"):
            tag = f"HUMAN: {human.upper()}"
            color = (120, 220, 120) if human == "good" else (230, 110, 110)
        else:
            tag = "CONSTR: " + ("GOOD" if entry["good"] else "BAD")
            color = (120, 180, 220) if entry["good"] else (220, 160, 110)
        draw.text((x + 4, y + thumb_h + 2), entry["id"], fill=(220, 220, 220))
        draw.text((x + 4, y + thumb_h + 16), tag, fill=color)
    sheet.save(out_path)
```

e) `main()` komplett ersetzen durch (beachte: `DEFAULT_AUDIO`-Konstante entfällt, Einzel-Audio geht über `--audio` + `--mode`):

```python
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Windows-Konsole (cp1252) vertraegt kein "→"/"≤" — UTF-8 erzwingen
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser.add_argument("--corpus", default="tests/golden/corpus.json",
                        help="Korpus-Manifest (golden-corpus/1)")
    parser.add_argument("--audio", default=None,
                        help="Einzelnes Audio statt Korpus (Modus via --mode)")
    parser.add_argument("--mode", default="podcast",
                        choices=["music", "podcast", "hybrid"],
                        help="Modus fuer --audio")
    parser.add_argument("--out-dir", default="tests/golden")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.audio:
        audios = [{"id": Path(args.audio).stem, "path": args.audio,
                   "mode": args.mode, "description": "Einzelnes Audio",
                   "source": args.audio}]
    else:
        audios = load_corpus(args.corpus)
        missing = missing_audio_files(audios)
        if missing:
            raise SystemExit(
                f"Fehlende Audio-Dateien: {', '.join(missing)}")

    gauss_mask = _center_gauss(PROBE_W, PROBE_H)
    entries: list[dict] = []
    analyzer = AudioAnalyzer()
    total = (len(audios) * len(VISUALIZERS)
             * len(ALPHA_CAPS) * len(MASK_VARIANTS))
    idx = 0
    t0 = time.time()
    for audio in audios:
        print(f"[golden] Analysiere {audio['id']} ({audio['path']}) ...",
              flush=True)
        features = analyzer.analyze(audio["path"], fps=FPS)
        features_dict = build_features_dict(features, features.frame_count,
                                            FPS)
        duration = float(features_dict["duration"])
        timestamps = [duration * f for f in SAMPLE_FRACTIONS]
        t_mid = duration * 0.5

        frame_dir = out_dir / "frames" / audio["id"]
        frame_dir.mkdir(parents=True, exist_ok=True)
        audio_entries: list[dict] = []
        for viz_name in VISUALIZERS:
            viz_cls = get_visualizer(viz_name)
            for cap in ALPHA_CAPS:
                for use_mask in MASK_VARIANTS:
                    idx += 1
                    t_var = time.time()
                    vid = variant_id(audio["id"], viz_name, cap, use_mask)
                    local = vid.split("__", 1)[1]
                    mask = gauss_mask if use_mask else None
                    constraints = MeasureConstraints(
                        alpha_cap=cap, alpha_from_luma=True,
                        subject_strength=0.8)
                    probe = ProbeRenderer(width=PROBE_W, height=PROBE_H,
                                          fps=FPS)
                    try:
                        viz = viz_cls(probe.ctx, PROBE_W, PROBE_H)
                        metrics = evaluate_params(
                            probe, viz, features_dict, timestamps, {},
                            constraints, subject_mask=mask)
                        frame = probe.render_frame(
                            viz, features_dict, t_mid, None, {},
                            constraints, subject_mask=mask)
                    finally:
                        probe.release()

                    rel_frame = f"frames/{audio['id']}/{local}.png"
                    Image.fromarray(frame).save(out_dir / rel_frame)

                    good = cap <= GOOD_CAP_MAX
                    entry = make_label_entry(vid, audio, viz_name, cap,
                                             use_mask, metrics, good)
                    entry["frame"] = rel_frame
                    entries.append(entry)
                    audio_entries.append(entry)
                    m3_txt = (f"{metrics['M3']:.4f}"
                              if metrics["M3"] is not None else "n/a")
                    print(f"[golden] ({idx}/{total}) {vid}: "
                          f"M1={metrics['M1']:.4f} M3={m3_txt} "
                          f"M5={metrics['M5']:.4f} "
                          f"M6={metrics['M6_violations']} "
                          f"({time.time() - t_var:.1f}s)", flush=True)

        sheet_path = out_dir / f"contact_sheet_{audio['id']}.png"
        build_contact_sheet(out_dir, audio_entries, sheet_path)
        print(f"[golden] Contact Sheet -> {sheet_path}", flush=True)

    # 3) labels.json schreiben (Schema golden-set/2)
    labels = {
        "version": "golden-set/2",
        "construction_labels": True,
        "human_labels_pending": True,
        "renders": entries,
    }
    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"[golden] {len(entries)} Eintraege -> {labels_path}", flush=True)
    print(f"[golden] Gesamtlaufzeit: {time.time() - t0:.1f}s", flush=True)
```

f) Die Konstante `DEFAULT_AUDIO` (Zeilen 32-36 im Original) entfernen — sie wird nicht mehr referenziert.

- [ ] **Step 4: Tests grün**

Run: `pytest tests/test_build_golden_set.py tests/test_golden_corpus.py -v`
Expected: 9 PASS

- [ ] **Step 5: Rückwärts-Sanity** — `python tools/build_golden_set.py --help` zeigt `--corpus/--audio/--mode/--out-dir` ohne Fehler. Kein GPU-Lauf in diesem Task (das ist Task 6).

---

### Task 4: calibrate_thresholds.py — Human-Labels bevorzugen

**Files:**
- Modify: `tools/calibrate_thresholds.py`
- Test: `tests/test_studio_calibration.py` (ergänzen)

**Interfaces:**
- Consumes: labels.json v2 (`human_label`-Feld aus Task 3).
- Produces: `effective_label(render) -> tuple[bool, str]` — `(label, quelle)`, label True = gut, quelle `"human"` oder `"construction"`. Bestehende Funktionen `set_hash`, `sweep_threshold` bleiben unverändert.

- [ ] **Step 1: Failing Tests ergänzen** — an `tests/test_studio_calibration.py` anhängen:

```python
def test_effective_label_prefers_human():
    from tools.calibrate_thresholds import effective_label
    assert effective_label({"good": False, "human_label": "good"}) == (True, "human")
    assert effective_label({"good": True, "human_label": "bad"}) == (False, "human")
    assert effective_label({"good": True, "human_label": None}) == (True, "construction")
    assert effective_label({"good": False}) == (False, "construction")
```

Run: `pytest tests/test_studio_calibration.py::test_effective_label_prefers_human -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 2: Implementieren** — in `tools/calibrate_thresholds.py`:

a) Docstring-Zeile 3-5 ersetzen durch:

```python
Liest labels.json {"renders": [{"id": str, "good": bool,
"human_label": "good"|"bad"|null, "metrics": {...}}]}, sweept
Kandidaten-Schwellen je Metrik und gibt Sensitivität/Spezifität aus.
Menschliche Labels (human_label) schlagen Konstruktions-Labels (good).
```

b) Neue Funktion nach `set_hash` einfügen:

```python
def effective_label(render):
    """Menschliches Label schlägt Konstruktions-Label.

    Rückgabe (label, quelle): label True = gut; quelle 'human' oder
    'construction'.
    """
    human = render.get("human_label")
    if human in ("good", "bad"):
        return human == "good", "human"
    return bool(render["good"]), "construction"
```

c) `main()` ersetzen durch:

```python
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="tests/golden/labels.json")
    args = parser.parse_args()

    data = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    renders = data.get("renders", [])
    labeled = [effective_label(r) for r in renders]
    human_count = sum(1 for _, src in labeled if src == "human")
    if human_count < 20:
        print(f"WARNUNG: nur {human_count} menschliche Labels "
              f"(Minimum 20) — Schwellen bleiben 'assumed'.")
    for metric, higher_is_bad in [("M1", True), ("M3", True), ("M4", False)]:
        # None-Werte (z.B. M3 ohne Maske, M4 ohne Quotes) herausfiltern
        pairs = [(r["metrics"][metric], lbl)
                 for r, (lbl, _src) in zip(renders, labeled)
                 if r["metrics"].get(metric) is not None]
        if not pairs:
            print(f"{metric}: keine Werte — übersprungen.")
            continue
        values = [v for v, _ in pairs]
        labels = [g for _, g in pairs]
        best = sweep_threshold(values, labels, higher_is_bad)
        print(f"{metric}: t={best['threshold']:.3f} "
              f"sens={best['sensitivity']:.2f} spec={best['specificity']:.2f}")
```

- [ ] **Step 3: Tests grün**

Run: `pytest tests/test_studio_calibration.py tests/test_studio_thresholds.py -v`
Expected: alle PASS

---

### Task 5: Label-Tool + Labeling-Raster + README

**Files:**
- Create: `tools/label_golden.py`
- Create: `docs/internal/golden-set-labeling-raster.md`
- Modify: `tests/golden/README.md` (komplett ersetzen)
- Test: `tests/test_label_golden.py` (neu)

**Interfaces:**
- Consumes: labels.json v2 (Task 3).
- Produces: `set_labels(data, pattern, label) -> list[str]` (geänderte ids; mutiert `data`), `label_stats(data) -> tuple[int, int, int]` (good, bad, offen). CLI `python tools/label_golden.py [--labels PATH] [--list] [--stats] [--set MUSTER=good|bad ...] [--dry-run]`.

- [ ] **Step 1: Failing Tests schreiben** — `tests/test_label_golden.py`:

```python
"""Tests für das Human-Label-Tool des Golden Sets."""

import pytest

from tools.label_golden import label_stats, set_labels


def _data():
    return {"renders": [
        {"id": "music_severance__pulsing_core_cap03", "human_label": None},
        {"id": "music_severance__pulsing_core_cap10", "human_label": None},
        {"id": "podcast_macy__voice_flow_cap03", "human_label": "good"},
    ]}


def test_set_labels_exact_match():
    data = _data()
    changed = set_labels(data, "podcast_macy__voice_flow_cap03", "bad")
    assert changed == ["podcast_macy__voice_flow_cap03"]
    assert data["renders"][2]["human_label"] == "bad"


def test_set_labels_glob():
    data = _data()
    changed = set_labels(data, "music_severance__*", "good")
    assert len(changed) == 2


def test_set_labels_rejects_invalid_label():
    with pytest.raises(ValueError, match="Label"):
        set_labels(_data(), "*", "meh")


def test_label_stats():
    data = _data()
    assert label_stats(data) == (1, 0, 2)
    set_labels(data, "music_severance__pulsing_core_cap03", "bad")
    assert label_stats(data) == (1, 1, 1)
```

- [ ] **Step 2: FAIL prüfen** — `pytest tests/test_label_golden.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Tool implementieren** — `tools/label_golden.py`:

```python
"""Menschliche Labels für das Golden-Set setzen (Spec §3.5).

Workflow: Contact Sheets (tests/golden/contact_sheet_<audio>.png) ansehen,
dann Einträge einzeln oder per Muster labeln:

    python tools/label_golden.py --stats
    python tools/label_golden.py --list
    python tools/label_golden.py --set "music_severance__pulsing_core_cap03=good"
    python tools/label_golden.py --set "podcast_macy__*_cap10*=bad" --dry-run
"""

import argparse
import fnmatch
import json
import sys
from pathlib import Path

VALID_LABELS = {"good", "bad"}


def set_labels(data, pattern, label):
    """Setzt human_label für alle Render-ids, die auf das fnmatch-Muster passen.

    Rückgabe: Liste der geänderten ids.
    """
    if label not in VALID_LABELS:
        raise ValueError(f"Label muss einer von {sorted(VALID_LABELS)} sein.")
    changed = []
    for r in data["renders"]:
        if fnmatch.fnmatchcase(r["id"], pattern):
            r["human_label"] = label
            changed.append(r["id"])
    return changed


def label_stats(data):
    """Zählt human_label-Belegung: (good, bad, offen)."""
    good = bad = open_ = 0
    for r in data["renders"]:
        if r.get("human_label") == "good":
            good += 1
        elif r.get("human_label") == "bad":
            bad += 1
        else:
            open_ += 1
    return good, bad, open_


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser.add_argument("--labels", default="tests/golden/labels.json")
    parser.add_argument("--list", action="store_true",
                        help="Ungelabelte ids ausgeben")
    parser.add_argument("--stats", action="store_true",
                        help="Label-Statistik ausgeben")
    parser.add_argument("--set", dest="assignments", action="append",
                        default=[], metavar="MUSTER=good|bad",
                        help="human_label per fnmatch-Muster setzen")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur anzeigen, nicht schreiben")
    args = parser.parse_args()

    path = Path(args.labels)
    data = json.loads(path.read_text(encoding="utf-8"))

    changed_any = False
    for assignment in args.assignments:
        pattern, _, label = assignment.partition("=")
        changed = set_labels(data, pattern.strip(), label.strip())
        changed_any = changed_any or bool(changed)
        print(f"{pattern.strip()}: {len(changed)} Eintraege -> {label.strip()}")
        if args.dry_run:
            for cid in changed:
                print(f"  {cid}")

    good, bad, open_ = label_stats(data)
    data["human_labels_pending"] = open_ > 0
    if args.stats or args.assignments:
        print(f"Labels: {good} good / {bad} bad / {open_} offen")
    if args.list:
        for r in data["renders"]:
            if r.get("human_label") not in VALID_LABELS:
                print(r["id"])
    if changed_any and not args.dry_run:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"Gespeichert: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Tests grün** — `pytest tests/test_label_golden.py -v` → 4 PASS.

- [ ] **Step 5: Labeling-Raster schreiben** — `docs/internal/golden-set-labeling-raster.md`:

```markdown
# Golden-Set Labeling-Raster (golden-set/2)

Zweck: menschliche Qualitäts-Labels für die Studio-Schwellenkalibrierung.
Die Konstruktions-Labels (alpha_cap ≤ 0.6) sind nur Platzhalter; erst
menschliche Labels machen Schwellen "calibrated" statt "assumed".

## Workflow

1. `python tools/build_golden_set.py` (baut Frames + Contact Sheets)
2. Pro Audio das Contact Sheet ansehen: `tests/golden/contact_sheet_<audio>.png`
   (36 Frames: 6 Visualizer × 3 alpha_caps × 2 Masken)
3. Labels setzen: `python tools/label_golden.py --set "<muster>=good|bad"`
   (einzeln per id, oder Muster wie `podcast_macy__*_cap10*=bad`;
   vorher `--dry-run` zum Prüfen)
4. Fortschritt: `python tools/label_golden.py --stats`
5. Wenn alle 216 gelabelt: `python tools/calibrate_thresholds.py`

## Bewertungsfrage (einzige Frage pro Frame)

**"Würde ich diesen Frame so in einem fertigen Video veröffentlichen?"**

- good = ja, so freigebefähig
- bad = nein — im Zweifel immer bad (keine "meh"-Kategorie)

## Worauf achten (Brücke zu den Metriken)

- **M1 Overlay-Energie**: "Frisst der Visualizer das Bild?" — good, wenn
  Hintergrund/Motiv noch erkennbar ist und der Frame nicht zugekleistert wirkt.
- **M3 Subjekt-Störung**: Im Maskenzentrum (da, wo ein Sprecher/Gesicht läge)
  soll es ruhig bleiben. bad, wenn genau die Bildmitte unruhig/verdeckt ist.
- **M4 Kontrast**: Wäre Quote-Text auf diesem Frame lesbar? bad, wenn der
  Hintergrund zu unruhig/kontrastarm für Text ist.
- **M5 Vitalität (modusabhängig)**:
  - music: genug Bewegung/Energie — statisch-langweilig ist bad.
  - podcast: Ruhe — hektisches Flackern/Zucken ist bad.

## Regeln für konsistente Labels

- Immer ein Audio am Stück labeln (Kontextwechsel vermeiden).
- Nicht die Metrikwerte in labels.json ansehen, bevor das Label steht
  (sonst labelt man die Metrik, nicht das Bild).
- ~2-3 Sekunden Pro Frame; nicht grübeln — Bauchentscheid nach der
  Bewertungsfrage.
- Die id unter jedem Frame nennt Visualizer/alpha_cap/Maske; das
  Konstruktions-Tag (CONSTR: ...) ignorieren — es ist bewusst simpel.
```

- [ ] **Step 6: README ersetzen** — `tests/golden/README.md` komplett neu:

```markdown
# Golden-Set für die Studio-Schwellenkalibrierung

Schema golden-set/2: 6 Referenz-Audios (3 Musik, 3 Podcast — siehe
`corpus.json`) × 6 Visualizer × alpha_cap {0.3, 0.6, 1.0} × Maske
{keine, Zentrums-Gauß} = 216 Renders.

Je Eintrag in labels.json: id, good (Konstruktions-Label),
human_label (null|"good"|"bad"), audio, mode, visualizer, alpha_cap,
mask, metrics {M1, M3, M4, M5}, frame (PNG-Pfad).
Metrikwerte erzeugt der ProbeRenderer; **Labels vergibt der Mensch**
(Raster: `docs/internal/golden-set-labeling-raster.md`).
Ohne ≥ 20 menschliche Labels bleiben alle Schwellen "assumed".

## Workflow

    python tools/build_golden_set.py            # Frames + Sheets + labels.json
    python tools/label_golden.py --stats        # Label-Fortschritt
    python tools/calibrate_thresholds.py        # Trennschärfe prüfen

## Stand 2026-08-09 (v2)

Neuaufbau auf Multi-Audio-Korpus. v1 (36 Einträge, ein Speech-Audio,
reine Konstruktions-Labels) liegt als `labels.v1.backup.json`.
```

- [ ] **Step 7: Checkpoint** — `pytest tests/test_label_golden.py tests/test_build_golden_set.py tests/test_golden_corpus.py tests/test_studio_calibration.py tests/test_studio_thresholds.py -q` → alle grün.

---

### Task 6: Harness ausführen + Artefakte verifizieren

**Files:**
- Modify: `tests/golden/labels.json` (neu generiert), `tests/golden/frames/**`, `tests/golden/contact_sheet_*.png` (generiert)
- Create: `tests/golden/labels.v1.backup.json`

**Interfaces:**
- Consumes: alles aus Tasks 1–3 (Korpus, Exzerpte, Multi-Audio-Builder).
- Produces: 216 Render-Einträge + 6 Contact Sheets, bereit für die menschliche Labeling-Session (Maintainer-Aufgabe danach).

- [ ] **Step 1: v1 sichern**

```bash
cp tests/golden/labels.json tests/golden/labels.v1.backup.json
```

- [ ] **Step 2: Harness laufen lassen** (GPU-Lauf, 216 Varianten — als Hintergrund-Task, großzügiges Timeout):

```bash
python tools/build_golden_set.py --corpus tests/golden/corpus.json
```
Expected: 216 Log-Zeilen `(idx/216)`, 6 "Contact Sheet"-Zeilen, "216 Eintraege -> tests/golden/labels.json"

- [ ] **Step 3: Struktur verifizieren**

```bash
python -c "
import json
d = json.load(open('tests/golden/labels.json', encoding='utf-8'))
r = d['renders']
assert d['version'] == 'golden-set/2', d['version']
assert len(r) == 216, len(r)
assert all(e['human_label'] is None for e in r)
assert {e['mode'] for e in r} == {'music', 'podcast'}
assert len({e['audio'] for e in r}) == 6
print('labels.json OK:', len(r), 'renders')
"
ls tests/golden/contact_sheet_*.png | wc -l   # Expected: 6
```

- [ ] **Step 4: Visuelle Stichprobe** — zwei Contact Sheets (`contact_sheet_music_severance.png`, `contact_sheet_podcast_macy.png`) als Bild öffnen und prüfen: 36 Thumbnails, ids lesbar, keine schwarzen/leeren Frames.

- [ ] **Step 5: Label-Tool-Smoke-Test gegen echte Datei**

```bash
python tools/label_golden.py --stats
python tools/label_golden.py --set "podcast_macy__voice_flow_cap03=good" --dry-run
```
Expected: "0 good / 0 bad / 216 offen"; dry-run listet genau 1 Eintrag, Datei unverändert (danach `--stats` weiterhin 216 offen).

- [ ] **Step 6: Gesamt-Checkpoint**

Run: `pytest tests/ -q --ignore=tests/golden -x -q` (voller Lauf wie bisher üblich) plus gezielt `pytest tests/test_golden_corpus.py tests/test_build_golden_set.py tests/test_label_golden.py tests/test_studio_calibration.py tests/test_studio_thresholds.py -v`
Expected: keine neuen Fehler; die 5 golden-bezogenen Dateien komplett grün.

---

## Self-Review

**Spec coverage (Goal-Kriterien):**
1. Multi-Audio + beide Modi im Builder → Task 3 (corpus loop, `--mode`), Task 1 (mode-Feld). ✓
2. 3 Musik + 3 Podcast ausgewählt, dokumentiert, als Assets vorhanden → Task 1 Step 4 (Manifest mit description/source), Task 2 (Dateien), `test_repo_manifest_complete`. ✓
3. Labeling-Raster-Doc → Task 5 Step 5. ✓
4. Golden-Renders generiert → Task 6. ✓
5. Labeling-Workflow nutzbar → Task 3 (Sheets mit HUMAN/CONSTR-Tags), Task 5 (label_golden.py + Raster), Task 6 Step 5 (Smoke-Test). ✓
6. Relevante pytest-Tests grün → Checkpoints in jedem Task + Task 6 Step 6. ✓

**Placeholder-Scan:** keine TBD/TODO; alle Code-Schritte enthalten vollständigen Code. Die ffmpeg-Offsets für `podcast_gorilla_mid` werden zur Laufzeit aus der Dateidauer berechnet (konkretes Kommando in Task 2 Step 2) — das ist Absicht, kein Platzhalter.

**Type-Konsistenz:** `load_corpus`/`missing_audio_files` (Task 1) = Importe in Task 3. `variant_id`/`make_label_entry`-Signaturen in Task 3 Tests == Implementierung. `build_contact_sheet(frame_base, entries, out_path, cols, thumb_w)` konsistent zwischen Task 3 Implementierung und Test (`entry["frame"]` relativ zu `frame_base`). `effective_label -> (bool, str)` konsistent in Task 4 Test und main(). `set_labels`/`label_stats`-Signaturen in Task 5 Tests == Implementierung. Bekannte Abweichung: `tests/golden/labels.json` wird neu geschrieben — `config/studio_thresholds.v1.json` referenziert es unter `calibration_report.set`; der Report bleibt historisch korrekt (verweist zusätzlich auf v1-Backup via README).
