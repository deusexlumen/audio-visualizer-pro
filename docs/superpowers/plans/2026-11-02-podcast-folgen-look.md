# Podcast-Folgen-Look Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ein neuer Render-Modus „Folgen-Look“ macht aus Audio + einem ChatGPT-Bild ein fertiges Podcast-Video (Kino-Standbild A, Zitat-Bühne B, Kapitel-Karte C), vorbereitet von der KI und vor dem Render in einer Prüfliste korrigierbar.

**Architecture:** Reine Logik (Lesezeit, Wortzeiten, Kamerafahrt, Zustand pro Zeitpunkt, YouTube-Kapiteltext, KI-Plan-Validierung) liegt im neuen Paket `src/episode/` und ist ohne GPU testbar. Ein eigener GPU-Pass `src/episode/compositor.py` setzt **nach** Bloom/Tonemapping das Folgenbild (scharf + vorab unscharf, Kamerafahrt, Kartenverschiebung) zusammen und legt die getonemappte Visualizer-Ebene (Stimm-Lichtstrich) per Screen-Blend darüber; dadurch läuft das Bild nie durch Bloom, ACES, Aberration oder Grain. Text (Zitate, Bühne, Karte) zeichnet `src/episode/text.py` auf der CPU in `_emit_frame`, an der Stelle des alten Zitat-Overlays. Der alte Pfad bleibt unberührt, solange `episode=None`.

**Tech Stack:** Python 3.11, ModernGL (GLSL `#version 330`), NumPy, Pillow, Pydantic v2, PyQt6, pytest (+ `pytest-qt`), google-genai.

**Spec:** `docs/superpowers/specs/2026-11-02-podcast-folgen-look-design.md`

## Global Constraints

- Kommentare, Docstrings und Commit-Meldungen auf Deutsch; Commits enden mit den Attributionszeilen der Sitzung.
- Branch: `feature/podcast-folgen-look`. Niemals `git add -A` (im Arbeitsbaum liegen `Fraktal-Zoom/`, `.playwright-mcp/` und ein geänderter `CLAUDE.md`): nur die Dateien des Tasks stagen.
- Alte `.avproj`-Dateien und `projects/podcast.json` laden unverändert; alle neuen Felder haben Standardwerte.
- Bestehende Presets in `config/`, der alte Zitat-Look und der Musik-Pfad ändern ihr Verhalten nicht. Vollständige Suite (Stand 2026-11-02: 577 grün) und `tests/golden/` bleiben grün.
- Lesezeit: 15 Zeichen/s, Nachlauf mindestens 1,5 s nach Sprechende, Ausblenden spätestens 0,3 s vor dem nächsten Zitat oder der nächsten Karte.
- Bühnen-Blende etwa 0,8 s; Kapitel-Karte etwa 4 s; Kamera-Zoom höchstens 8 % (`max_zoom = 1.08`); Akzentfarbe `#F2B866`; Markentext `DEUS EX LUMEN`.
- Zeiten von Zitaten kommen immer aus den Segmenten (`locate_in_segments`), nie aus Zeitangaben der KI.
- YouTube-Kapitel: erste Zeile `0:00`, mindestens 3 Kapitel, jedes mindestens 10 s.
- Standardauflösung 1920×1080.
- ⚠️ Task 6 lädt eine Schriftdatei herunter. Vorher Dateiname, Quelle und Größe nennen und das Go des Nutzers abwarten.

## Review Focus

1. **Bild kleiner oder anders geformt als 16:9** (1024×1024, 1536×1024): Zuschnitt (cover) ohne Verzerrung, kein Rand sichtbar bei Kamerafahrt; Hinweis in der Prüfliste. → Tests in Task 5 (`test_cover_crop_*`, `test_image_too_small`) und Task 4 (`test_camera_stays_inside_image`).
2. **Zitate dicht hintereinander oder überlappend**: kein doppelter Text, kein negatives Fenster. → `test_display_window_overlapped_next` (Task 2), `test_state_never_two_texts` (Task 4).
3. **Folge ohne Kapitel oder KI fehlgeschlagen**: Render läuft trotzdem (eine Kamerafahrt über alles, keine Karte); Kapiteltext warnt statt falsch zu schreiben. → `test_timeline_without_chapters` (Task 4), `test_chapter_text_warns_too_few` (Task 3), `test_panel_shows_error_keeps_title` (Task 11).
4. **Intro-Video gesetzt**: Kapitelzeiten in der Enddatei verschoben, erste Zeile bleibt `0:00`. → `test_chapter_text_with_intro_offset` und `test_main_start_offset_matches_filter` (Task 3).
5. **Altes Projekt oder alter Preset nach dem Update**: lädt und rendert wie vorher. → `test_old_project_dict_loads_defaults` (Task 10), `test_old_quote_json_loads` (Task 1), volle Suite in Task 12.

---

### Task 1: Datenmodell und Schema

**Files:**
- Modify: `src/types.py` (`Quote`, neu `Chapter`, `ProjectConfig`)
- Modify: `config/schemas.py` (`QuoteSchema`, neu `ChapterSchema`, Projekt-Schema)
- Modify: `src/gui/quote_editor.py:374-380` (`result_quote`), `src/beat_sync.py:72`, `src/quote_refiner.py:111`: Zitate per `quote.model_copy(update={...})` statt `Quote(...)` neu bauen, damit `kind` und `accent_word` erhalten bleiben
- Test: `tests/test_episode_model.py`

**Interfaces:**
- Produces:
  - `Quote.kind: Literal["quote", "stage"] = "quote"`, `Quote.accent_word: Optional[str] = None`
  - `class Chapter(BaseModel): start: float; title: str`
  - `ProjectConfig.look: Literal["classic", "episode"] = "classic"`, `.chapters: List[Chapter] = []`, `.episode_title: Optional[str] = None`, `.episode_number: Optional[str] = None`
  - Dieselben Felder im Pydantic-Schema von `config/schemas.py` (`ChapterSchema` mit `start: float = Field(ge=0)`).

- [ ] **Step 1: Write the failing tests**

```python
def test_old_quote_json_loads():
    q = Quote(text="Hallo Welt heute", start_time=1.0, end_time=2.0)
    assert q.kind == "quote" and q.accent_word is None

def test_stage_quote_roundtrip():
    q = Quote(text="Der Mensch stört", start_time=1, end_time=2, kind="stage", accent_word="stört")
    assert Quote(**q.model_dump()) == q

def test_project_defaults_classic():
    cfg = load_and_validate_config("config/podcast_story.json")
    assert cfg.look == "classic" and cfg.chapters == [] and cfg.episode_title is None

def test_schema_accepts_episode_fields(tmp_path):
    # podcast_story.json + look/chapters/episode_title -> valid, Kapitel als Objekte
    ...
    assert cfg.look == "episode" and cfg.chapters[0].title == "Einstieg"

def test_schema_rejects_negative_chapter_start(tmp_path):
    # chapters=[{"start": -1, "title": "x"}] -> ValidationError

def test_stage_survives_editor_beat_sync_and_refiner(qtbot):
    # Bühnen-Zitat durch QuoteEditorDialog.result_quote, sync_quotes_to_beats, refine_quote_timestamps
    assert all(r.kind == "stage" and r.accent_word == "stört" for r in results)
```

- [ ] **Step 2: Run** `pytest tests/test_episode_model.py -v` → FAIL (`kind` unbekannt / `Chapter` fehlt).
- [ ] **Step 3: Implement** die Felder in `src/types.py` und `config/schemas.py`; das Schema reicht `look`, `chapters`, `episode_title`, `episode_number` durch.
- [ ] **Step 4: Run** `pytest tests/test_episode_model.py tests/test_cli.py tests/test_recipe_schema.py -v` → PASS.
- [ ] **Step 5: Commit** `feat(folge): Datenmodell fuer Kapitel, Buehnen-Saetze und Folgen-Look`

---

### Task 2: Look-Konfiguration und Lesezeit

**Files:**
- Create: `src/episode/__init__.py`, `src/episode/config.py`, `src/episode/timing.py`
- Test: `tests/test_episode_timing.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class EpisodeLookConfig` mit Feldern und Standardwerten: `chars_per_second=15.0`, `min_hold=1.5`, `gap_before_next=0.3`, `card_duration=4.0`, `card_fade=0.4`, `stage_fade=0.8`, `text_fade_out=0.35`, `word_fade=0.25`, `max_zoom=1.08`, `accent_color="#F2B866"`, `brand_text="DEUS EX LUMEN"`, `font_path: Optional[str]=None`, `quote_interval=90.0`, `min_chapters=3`, `max_chapters=8`.
  - `@dataclass class EpisodeRenderSpec: image_path: str; chapters: list[Chapter]; quotes: list[Quote]; title: str; number: Optional[str] = None; config: EpisodeLookConfig = field(default_factory=EpisodeLookConfig)` (in `config.py`).
  - `display_window(quote: Quote, next_event_start: Optional[float], cfg: EpisodeLookConfig) -> tuple[float, float]`
  - `word_reveal_times(quote: Quote) -> list[float]`
  - `visible_word_count(reveal_times: list[float], t: float) -> int`

`display_window`: `start = quote.start_time`; `wunsch = max(quote.end_time + min_hold, start + len(text) / chars_per_second)`. Ohne `next_event_start`: Ergebnis `(start, wunsch)`. Mit `next_event_start = n`: `ende = max(min(wunsch, n - gap_before_next), min(start + 0.5, n))`. So überlappt ein Fenster nie das nächste.
`word_reveal_times`: Wort i erscheint bei `start + (Zeichenposition des Wortanfangs / len(text)) * (end - start)`.

- [ ] **Step 1: Write the failing tests**

```python
CFG = EpisodeLookConfig()
def q(text, s, e): return Quote(text=text, start_time=s, end_time=e)

def test_display_window_short_uses_hold():
    assert display_window(q("Kurz und klar gesagt.", 10, 11.5), None, CFG) == pytest.approx((10, 13.0))

def test_display_window_long_uses_reading_time():
    assert display_window(q("x" * 150, 20, 28), None, CFG) == pytest.approx((20, 30.0))

def test_display_window_dense_next_caps():
    assert display_window(q("Kurz und klar gesagt.", 10, 11.5), 12.0, CFG) == pytest.approx((10, 11.7))

def test_display_window_overlapped_next():
    assert display_window(q("Kurz und klar gesagt.", 10, 11.5), 10.2, CFG) == pytest.approx((10, 10.2))

def test_word_reveal_times():
    assert word_reveal_times(q("eins zwei drei", 0, 1.4)) == pytest.approx([0.0, 0.5, 1.0])

def test_visible_word_count():
    assert [visible_word_count([0, 0.5, 1.0], t) for t in (-0.1, 0, 0.7, 2)] == [0, 1, 2, 3]
```

- [ ] **Step 2: Run** `pytest tests/test_episode_timing.py -v` → FAIL (Modul fehlt).
- [ ] **Step 3: Implement** `config.py` und `timing.py` wie oben.
- [ ] **Step 4: Run** `pytest tests/test_episode_timing.py -v` → PASS.
- [ ] **Step 5: Commit** `feat(folge): Lesezeit-Regel und Wortzeiten`

---

### Task 3: YouTube-Kapiteltext, Intro-Versatz, Folgentitel

**Files:**
- Create: `src/episode/chapters.py`
- Modify: `src/intro_renderer.py` (Versatzberechnung aus `_build_filter_complex` in eine Funktion ziehen, Verhalten gleich)
- Test: `tests/test_episode_chapters.py`

**Interfaces:**
- Produces:
  - `main_start_offset(intro_duration: float, fade_duration: float) -> float` in `src/intro_renderer.py`; `_build_filter_complex` nutzt sie.
  - `format_timestamp(seconds: float) -> str` → `M:SS` unter einer Stunde, sonst `H:MM:SS`.
  - `youtube_chapter_text(chapters: list[Chapter], total_duration: float, intro_offset: float = 0.0) -> tuple[str, list[str]]` → (Text, Warnungen). Erste Zeile immer `0:00 <Titel des ersten Kapitels>`; alle weiteren `start + intro_offset`. Warnungen: weniger als 3 Kapitel; ein Kapitel kürzer als 10 s (das letzte gemessen bis `total_duration + intro_offset`).
  - `write_chapter_file(video_path: str, chapters, total_duration, intro_offset=0.0) -> tuple[Path, list[str]]` → schreibt `<video>.kapitel.txt` (UTF-8).
  - `default_episode_title(audio_path: str) -> str`

- [ ] **Step 1: Write the failing tests**

```python
CH = [Chapter(start=0, title="Einstieg"), Chapter(start=95.4, title="Das System"), Chapter(start=3725, title="Ausweg")]

def test_format_timestamp():
    assert [format_timestamp(s) for s in (0, 95.4, 3725)] == ["0:00", "1:35", "1:02:05"]

def test_chapter_text_basic():
    text, warn = youtube_chapter_text(CH, total_duration=4000)
    assert text.splitlines() == ["0:00 Einstieg", "1:35 Das System", "1:02:05 Ausweg"] and warn == []

def test_chapter_text_with_intro_offset():
    text, _ = youtube_chapter_text(CH, total_duration=4000, intro_offset=4.0)
    assert text.splitlines()[:2] == ["0:00 Einstieg", "1:39 Das System"]

def test_chapter_text_warns_too_few():
    _, warn = youtube_chapter_text(CH[:2], total_duration=4000)
    assert any("3" in w for w in warn)

def test_chapter_text_warns_short_chapter():
    _, warn = youtube_chapter_text([CH[0], Chapter(start=5, title="kurz"), CH[1]], total_duration=4000)
    assert any("10 s" in w for w in warn)

def test_main_start_offset_matches_filter():
    assert main_start_offset(5.0, 1.0) == pytest.approx(4.0)
    assert main_start_offset(0.5, 1.0) == pytest.approx(0.1)   # Fade wird auf intro-0.1 gekuerzt

def test_default_episode_title():
    assert default_episode_title("x/Digitale_Gorilla-Taktiken_gegen_den_KI-Müll.m4a") == "Digitale Gorilla-Taktiken gegen den KI-Müll"
```

- [ ] **Step 2: Run** `pytest tests/test_episode_chapters.py -v` → FAIL.
- [ ] **Step 3: Implement** die Funktionen; `_build_filter_complex` ruft `main_start_offset` statt der Inline-Rechnung.
- [ ] **Step 4: Run** `pytest tests/test_episode_chapters.py tests/test_intro_renderer.py -v` → PASS.
- [ ] **Step 5: Commit** `feat(folge): YouTube-Kapiteltext mit Intro-Versatz`

---

### Task 4: Folgen-Zeitachse (Kamera, Bühne, Karte, aktiver Text)

**Files:**
- Create: `src/episode/timeline.py`
- Test: `tests/test_episode_timeline.py`

**Interfaces:**
- Consumes: `display_window`, `EpisodeLookConfig` (Task 2), `Chapter`, `Quote` (Task 1).
- Produces:
  - `@dataclass(frozen=True) class CameraPose: scale: float; offset_x: float; offset_y: float` (Versatz in UV-Einheiten)
  - `@dataclass(frozen=True) class EpisodeState: camera: CameraPose; stage_amount: float; card_amount: float; chapter_index: int; text_index: int; text_alpha: float` (`-1` = keins)
  - `class EpisodeTimeline(chapters, quotes, duration, cfg)` mit `items: list[Quote]` (nach Start sortiert), `windows: list[tuple[float, float]]`, `card_windows: list[tuple[float, float]]`, `state_at(t: float) -> EpisodeState`, `card_conflicts() -> list[int]`.

Regeln:
- Fenster je Zitat aus `display_window`, `next_event_start` = früherer von nächstem Zitat-Start und nächstem Karten-Start.
- Karte k: Fenster `[start - card_fade, start + card_duration]` (erste bei 0: `[0, card_duration]`), Rampe `card_fade` rein und raus; bei `t == chapter.start` ist `card_amount == 1`.
- Während `card_amount > 0` ist `text_index == -1` (Karte gewinnt). `card_conflicts()` liefert Indizes, deren Fenster ein Kartenfenster schneidet.
- `stage_amount`: für `kind == "stage"` von `start - stage_fade` linear auf 1 bis `start`, hält, fällt von `end` bis `end + stage_fade`.
- `text_alpha`: 1 im Fenster, fällt in den letzten `text_fade_out` Sekunden auf 0.
- Kamera: Schlüsselposen an jeder Kapitelgrenze (plus Anfang und Ende), Kapitel k fährt von Pose k zu Pose k+1 mit Smoothstep. Posen deterministisch aus einer festen Liste von 4 Bewegungen (rein, raus, links, rechts), `1.0 <= scale <= max_zoom`, `|offset| <= (scale - 1) / 2`. Ohne Kapitel: eine Fahrt über die ganze Dauer.

- [ ] **Step 1: Write the failing tests**

```python
def test_camera_stays_inside_image():  # für t in linspace(0, dur, 2000)
    assert 1.0 <= s.camera.scale <= 1.08 and abs(s.camera.offset_x) <= (s.camera.scale - 1) / 2 + 1e-9

def test_camera_continuous_at_chapter_boundary():
    a, b = tl.state_at(95.4 - 1/30).camera, tl.state_at(95.4).camera
    assert abs(a.scale - b.scale) < 1e-3 and abs(a.offset_x - b.offset_x) < 1e-3

def test_card_full_at_chapter_start():
    assert tl.state_at(95.4).card_amount == pytest.approx(1.0) and tl.state_at(95.4).chapter_index == 1

def test_card_hides_text_and_reports_conflict():
    # Zitat startet bei 96.0 (in der Karte von 95.4)
    assert tl.state_at(97.0).text_index == -1 and tl.card_conflicts() == [idx]

def test_stage_amount_ramps():
    # Bühne 30.0-33.0, Fenster-Ende E aus display_window
    assert tl.state_at(29.2).stage_amount == pytest.approx(0.0, abs=1e-6)
    assert tl.state_at(30.0).stage_amount == pytest.approx(1.0)
    assert tl.state_at(E + 0.8).stage_amount == pytest.approx(0.0, abs=1e-6)

def test_state_never_two_texts():  # dicht folgende Zitate: Fenster überlappen nie
    assert all(w0[1] <= w1[0] + 1e-9 for w0, w1 in zip(tl.windows, tl.windows[1:]))

def test_timeline_without_chapters():
    tl = EpisodeTimeline([], quotes, 600, CFG)
    assert tl.state_at(10).card_amount == 0 and tl.state_at(10).chapter_index == -1
```

- [ ] **Step 2: Run** `pytest tests/test_episode_timeline.py -v` → FAIL.
- [ ] **Step 3: Implement** `timeline.py` nach den Regeln oben.
- [ ] **Step 4: Run** `pytest tests/test_episode_timeline.py -v` → PASS.
- [ ] **Step 5: Commit** `feat(folge): Zeitachse fuer Kamerafahrt, Buehne und Kapitel-Karte`

---

### Task 5: GPU-Compositor für das Folgenbild

**Files:**
- Create: `src/episode/compositor.py`
- Test: `tests/test_episode_compositor.py` (`pytestmark = pytest.mark.gpu`, Kontext wie in `tests/test_gpu_spiral.py`)

**Interfaces:**
- Consumes: `EpisodeState`, `CameraPose` (Task 4).
- Produces:
  - `cover_crop(img: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image` (Mitte auf Zielformat zuschneiden, dann LANCZOS)
  - `image_too_small(image_size: tuple[int, int], width: int, height: int) -> bool` (True, wenn der 16:9-Zuschnitt schmaler als `width` ist)
  - `class EpisodeCompositor(ctx, width, height, image_path: str, stage_blur_px: float = 18.0)` mit `render(overlay_tex, state: EpisodeState, target_fbo) -> None` und `release()`.

Shader: UV nach Kamerapose (`uv' = 0.5 + (uv - 0.5) / scale + offset`), bei Karte um `0.34 * card_amount` nach rechts verschoben; scharfe und vorab unscharfe Textur (beide 8-Bit, wie `_load_background_texture`) mischen nach `stage_amount`, Helligkeit `mix(1, 0.35, stage_amount)`; Lesbarkeits-Verlauf nur unterhalb 55 % der Bildhöhe; danach `overlay_tex` (getonemappte Visualizer-Ebene) mal `mix(1, 0.5, max(stage_amount, card_amount))` (Lichtstrich in B und C gedimmt) per Screen-Blend `1 - (1 - bild) * (1 - overlay)`. Kein Tonemapping, kein Grain.

- [ ] **Step 1: Write the failing tests**

```python
def test_cover_crop_square_to_16_9():
    assert cover_crop(Image.new("RGB", (1024, 1024)), 1920, 1080).size == (1920, 1080)

def test_cover_crop_keeps_center():  # roter Mittelpunkt bleibt in der Mitte

def test_image_too_small():
    assert image_too_small((1536, 1024), 1920, 1080) and not image_too_small((2560, 1440), 1920, 1080)

def test_colour_fidelity_neutral_state(gl, tmp_path):
    # Zufallsbild 256x144, state: scale 1, offset 0, stage 0, card 0, overlay schwarz
    diff = np.abs(out[:72].astype(int) - ref[:72].astype(int)).mean()   # obere Hälfte
    assert diff <= 3.0

def test_black_overlay_changes_nothing_and_white_saturates(gl, tmp_path):

def test_stage_darkens(gl, tmp_path):
    assert out_stage[:72].mean() < 0.5 * out_neutral[:72].mean()

def test_light_line_dimmed_on_stage(gl, tmp_path):
    # schwarzes Bild, graues Overlay: Beitrag bei stage 1 etwa halb so hell wie bei stage 0
```

- [ ] **Step 2: Run** `pytest tests/test_episode_compositor.py -v` → FAIL.
- [ ] **Step 3: Implement** `compositor.py`.
- [ ] **Step 4: Run** `pytest tests/test_episode_compositor.py -v` → PASS.
- [ ] **Step 5: Commit** `feat(folge): GPU-Compositor fuer Folgenbild ohne Tonemapping`

---

### Task 6: Serifenschrift und Text-Renderer

**Files:**
- Create: `assets/fonts/SourceSerif4-Variable.ttf` (oder gleichwertige OFL-Serifenschrift) plus Lizenz `assets/fonts/OFL-SourceSerif4.txt`
- Create: `src/episode/text.py`
- Test: `tests/test_episode_text.py`

⚠️ Vor Step 1: Download-Quelle (offizielles GitHub-Release der Schrift), Dateiname und Größe nennen und das Go des Nutzers abwarten.

**Interfaces:**
- Consumes: `EpisodeTimeline`, `EpisodeState` (Task 4), `word_reveal_times`, `visible_word_count` (Task 2).
- Produces:
  - `SERIF_FONT_PATH: Path`
  - `wrap_balanced(words: list[str], measure: Callable[[str], int], max_width: int) -> list[str]` (keine letzte Zeile mit nur einem Wort, wenn ≥ 4 Wörter)
  - `class EpisodeTextRenderer(timeline: EpisodeTimeline, width, height, title: str, number: Optional[str], cfg)` mit `apply(frame: np.ndarray, t: float) -> np.ndarray` (holt den Zustand selbst über `timeline.state_at(t)`).

Layout: Marke klein oben links (Alpha 0,75, aus während einer Karte). Zitat: unten links ab 6 % Breite, max. 64 % Breite, Serif, Größe `0.042 * height`, Zeilenabstand 1,35, keine Box, weicher Schatten; das jeweils neueste Wort blendet über `word_fade` ein. Bühne: Mitte, Inter Bold (`Inter-Variable.ttf`, Achse Bold), Größe `0.062 * height`, max. 72 % Breite, `accent_word` in `accent_color`. Karte: dunkle Spalte links 40 % mit Verlauf nach rechts, Zeile `DEUS EX LUMEN · FOLGE <n>` (ohne Nummer nur Marke) in Akzentfarbe, Titel Serif fett, `Kapitel <k> · <Name>`. Gecacht wird pro (Element, Anzahl sichtbarer Wörter) bzw. pro Kapitel.

- [ ] **Step 1: Write the failing tests**

```python
def test_font_has_german_glyphs():
    cmap = TTFont(SERIF_FONT_PATH).getBestCmap()
    assert all(ord(c) in cmap for c in "äöüÄÖÜß„“")

def test_wrap_balanced_no_orphan():
    lines = wrap_balanced("Wir sind mitten in der Ära des Dead Internet".split(), len, 30)
    assert len(lines[-1].split()) >= 2

def test_quote_has_no_box():  # graues Bild, Zitat aktiv
    changed = np.abs(out.astype(int) - frame.astype(int)).max(axis=2) > 40
    y0, y1, x0, x1 = bbox(changed)
    assert changed[y0:y1, x0:x1].mean() < 0.5

def test_words_appear_over_time():
    assert diff_at(t_word2) < diff_at(t_word3) < diff_at(t_end)

def test_stage_accent_word_coloured():  # Pixel nahe #F2B866 vorhanden

def test_card_left_column():
    assert change[:, : int(0.35 * w)].mean() > 5 * change[:, int(0.6 * w):].mean()

def test_no_text_outside_windows_except_brand():
```

- [ ] **Step 2: Run** `pytest tests/test_episode_text.py -v` → FAIL.
- [ ] **Step 3: Implement** `text.py`; Schrift und Lizenz ablegen.
- [ ] **Step 4: Run** `pytest tests/test_episode_text.py -v` → PASS.
- [ ] **Step 5: Commit** `feat(folge): Text-Renderer fuer Zitate, Buehne und Kapitel-Karte`

---

### Task 7: Visualizer „voice_line“ (Stimm-Lichtstrich)

**Files:**
- Create: `src/gpu_visualizers/voice_line.py`
- Test: wird von `tests/test_visuals.py` automatisch erfasst; zusätzlich `tests/test_voice_line.py`

**Interfaces:**
- Produces: Visualizer `voice_line` (Skelett aus `AGENTS.md`), `PARAMS`: `y_position (0.07, 0.0, 0.5)`, `thickness_px (2.0, 1.0, 8.0)`, `width (0.88, 0.2, 1.0)`, `intensity (1.0, 0.0, 3.0)`. Farbe `colors["primary"]`, Standard `#F2B866`. Treiber `voice_band`, Rückfall `rms`. Horizontal zur Mitte hin am hellsten, an den Enden ausgeblendet; sonst schwarz.

`voice_line` darf nicht in den klassischen oder Musik-Pfad rutschen: neue Konstante `AUTO_EXCLUDED_VISUALIZERS = {"voice_line"}` in `src/gpu_visualizers/__init__.py`. Die KI-Kandidatenliste für `generate_scene_timeline` (`src/gui/workers.py:353`) filtert sie heraus. `list_visualizers()` bleibt unverändert (die Auswahlliste im Parameter-Reiter darf ihn zeigen).

- [ ] **Step 1: Write the failing tests** `test_voice_line_brighter_with_voice` (Frame mit `voice_band=0.9` heller als mit `0.0`), `test_voice_line_only_near_bottom` (Zeilen über 30 % Höhe bleiben schwarz), `test_voice_line_not_offered_to_auto` (SmartMatcher-Empfehlung für Sprach- und Musik-Features ist nie `voice_line`; die Liste an `generate_scene_timeline` enthält ihn nicht).
- [ ] **Step 2: Run** `pytest tests/test_voice_line.py -v` → FAIL.
- [ ] **Step 3: Implement** den Visualizer und den Filter. Vorher `grep -rn "list_visualizers\|VISUALIZER_MAP" src tools main.py` laufen lassen und jeden Verbraucher prüfen.
- [ ] **Step 4: Run** `pytest tests/test_voice_line.py tests/test_visuals.py -v` → PASS.
- [ ] **Step 5: Commit** `feat(folge): Visualizer voice_line als Stimm-Lichtstrich`

---

### Task 8: Renderer und Vorschau im Folgen-Modus

**Files:**
- Modify: `src/gpu_renderer.py` (`GPUBatchRenderer.render`, Frame-Schleife, `_emit_frame`, `release`)
- Modify: `src/gpu_preview.py` (`render_gpu_preview`)
- Test: `tests/test_episode_render.py` (`gpu`)

**Interfaces:**
- Consumes: `EpisodeRenderSpec` (Task 2), `EpisodeTimeline` (Task 4), `EpisodeCompositor` (Task 5), `EpisodeTextRenderer` (Task 6).
- Produces: `render(..., episode: Optional[EpisodeRenderSpec] = None)` und `render_gpu_preview(..., episode: Optional[EpisodeRenderSpec] = None)`.

Mit `episode`: Hintergrund wird **nicht** in `self.fbo` gezeichnet (Clear schwarz); Visualizer, Bloom, Postprocess wie gehabt; danach `compositor.render(self.post_fbo.color_attachments[0], timeline.state_at(time), episode_fbo)` und `target_fbo = episode_fbo` (RGBA8, Größe wie `post_fbo`); in `_emit_frame` ersetzt `EpisodeTextRenderer.apply` das alte Zitat-Overlay; `quotes`/`quote_config` werden ignoriert. Ohne `episode` bleibt jeder Codepfad unverändert.

- [ ] **Step 1: Write the failing tests**

```python
def test_episode_render_writes_video(tmp_path, dummy_audio):
    # 2 s, 320x180, voice_line, ein Kapitel bei 0, ein Zitat 0.5-1.5
    assert probe(out)["frames"] == 60

def test_episode_frame_keeps_image_colours(tmp_path, dummy_audio):
    # EpisodeLookConfig(card_duration=0.5); Frame bei 1.8 s (Karte und Zitat vorbei): obere Hälfte ~ Bildmittel
    assert abs(frame[:90].mean() - ref[:90].mean()) <= 6

def test_episode_preview_returns_image():
    assert render_gpu_preview(..., episode=spec).size == (480, 270)

def test_classic_render_ignores_episode_code(monkeypatch):
    # episode=None: EpisodeCompositor wird nie gebaut
```

- [ ] **Step 2: Run** `pytest tests/test_episode_render.py -v` → FAIL.
- [ ] **Step 3: Implement** die Verzweigung in Renderer und Vorschau; Compositor in `release()` freigeben.
- [ ] **Step 4: Run** `pytest tests/test_episode_render.py tests/test_gpu_renderer.py tests/test_gpu_preview.py tests/test_spiral_render_integration.py -v` → PASS.
- [ ] **Step 5: Commit** `feat(folge): Renderer und Vorschau im Folgen-Modus`

---

### Task 9: KI-Plan (Kapitel, Bühnen-Sätze, Zitate)

**Files:**
- Create: `src/episode/plan.py`
- Modify: `src/gemini_integration.py` (neu `plan_episode`, `plan_episode_async`, Schema `EPISODE_PLAN_SCHEMA`)
- Test: `tests/test_episode_plan.py`

**Interfaces:**
- Consumes: `locate_in_segments` (`src/quote_timing.py`), `GeminiIntegration.transcribe_segments`, `_call_gemini_with_retry`, `save_json_result`/`load_json_result`.
- Produces:
  - `@dataclass class EpisodePlan: chapters: list[Chapter]; quotes: list[Quote]` (enthält `stage` und `quote`)
  - `build_episode_plan(raw: dict, segments: list[dict], duration: Optional[float], cfg: EpisodeLookConfig = EpisodeLookConfig()) -> EpisodePlan`
  - `GeminiIntegration.plan_episode(audio_path: str, audio_duration: Optional[float] = None, use_cache: bool = True, progress_callback=None) -> EpisodePlan` und `plan_episode_async(...) -> Future`
  - Rohformat: `{"chapters": [{"segment_index": int, "title": str}], "stage": [{"text": str, "accent_word": str}], "quotes": [{"text": str, "confidence": float}]}`

`build_episode_plan`: Kapitelstart = `segments[segment_index]["start"]`, erstes Kapitel auf 0.0, sortieren, Kapitel näher als 10 s am vorigen verwerfen, höchstens `max_chapters`; Texte per `locate_in_segments` verorten, nicht gefundene verwerfen; höchstens 3 Bühnen-Sätze; `accent_word` muss ein Wort des Satzes sein (ohne Groß/Klein), sonst längstes Wort; normale Zitate, die ein Bühnen-Fenster schneiden, verwerfen. Cache-Schlüssel `f"{model}|{PROMPT_VERSION}|episode"`, Art `"episode_plan"`.

Prompt (Spec-Vorgabe, wörtlich übernehmen, Zahlen einsetzen): Transkript mit Segmentnummern; „Teile die Folge in {k_min}–{k_max} Kapitel nach Thema. Gib für jedes Kapitel die Nummer des Segments, mit dem es beginnt, und einen kurzen Namen (höchstens 5 Wörter).“ — „Wähle 2–3 Bühnen-Sätze: die kühnsten Behauptungen der Folge, die Sätze, die Leute zum Nachdenken bringen. Nicht die beste Zusammenfassung. Nenne je Satz ein Schlüsselwort aus dem Satz.“ — „Wähle etwa {n} weitere Zitate (prägnant, höchstens 25 Wörter).“ — dazu die harten Regeln aus `_select_quotes_from_text` (wörtlich, keine Umformulierung, Satzanfang bis Satzende). `n = max(1, round(duration / quote_interval))`, `temperature: 0.0`.

- [ ] **Step 1: Write the failing tests** (Segmente als Fixture, keine Netzwerkaufrufe)

```python
def test_plan_first_chapter_at_zero(): assert plan.chapters[0].start == 0.0
def test_plan_drops_chapter_closer_than_10s():
def test_plan_drops_unlocatable_text():
def test_plan_caps_stage_at_three():
def test_plan_accent_word_fallback_longest(): assert stage.accent_word == "Gesellschaft"
def test_plan_drops_quote_overlapping_stage():
def test_plan_episode_uses_cache(monkeypatch):  # zweiter Aufruf ruft Gemini nicht
def test_plan_prompt_names_boldest_claims(monkeypatch):
    assert "kühnsten Behauptungen" in captured_prompt
```

- [ ] **Step 2: Run** `pytest tests/test_episode_plan.py -v` → FAIL.
- [ ] **Step 3: Implement** `plan.py` und die Gemini-Methoden.
- [ ] **Step 4: Run** `pytest tests/test_episode_plan.py tests/test_gemini_integration.py -v` → PASS.
- [ ] **Step 5: Commit** `feat(folge): KI-Plan fuer Kapitel, Buehnen-Saetze und Zitate`

---

### Task 10: CLI, Preset und GUI-Zustand

**Files:**
- Create: `config/podcast_folge.json`
- Modify: `main.py` (Config-Auflösung und `render`-Aufruf), `src/gui/state.py`, `src/gui/workers.py` (`RenderWorker` reicht `episode` durch; neu `EpisodePlanWorker`)
- Test: `tests/test_episode_cli.py`, Ergänzungen in `tests/test_gui_state.py`, `tests/test_gui_workers.py`

**Interfaces:**
- Consumes: `EpisodeRenderSpec`, `default_episode_title`, `write_chapter_file`, `main_start_offset`, `GeminiIntegration.plan_episode_async`.
- Produces:
  - Preset `podcast_folge.json`: `visual.type = "voice_line"`, 1920×1080, 30 fps, `colors.primary = "#F2B866"`, `postprocess` mit `chromatic_aberration 0`, `film_grain 0`, `vignette 0`, `look = "episode"`.
  - CLI: bei `look == "episode"` ist `background_image` Pflicht (sonst `click.BadParameter`), `episode=EpisodeRenderSpec(...)` an `render`, danach `write_chapter_file` (mit Intro-Versatz, falls Intro gesetzt) und Warnungen ausgeben.
  - `AppState`: neue Schlüssel `look="classic"`, `chapters: list[Chapter]=[]`, `episode_title=""`, `episode_number=""`; `to_dict` schreibt `kind`/`accent_word` je Zitat und `chapters`; `apply_dict`/`from_dict` setzen fehlende Schlüssel auf Standard (Muster wie `SPIRAL_STATE_KEYS`).
  - `EpisodePlanWorker(gemini, audio_path, audio_duration, parent=None)` mit Signalen `plan_ready(object)`, `plan_error(str, str)`; Timeout 300 s.
  - `RenderWorker` reicht `config.get("episode")` als `episode=` durch.
  - `PreviewWorker` reicht ebenso `episode=` an `render_gpu_preview` durch, damit ▶ in der Prüfliste den Folgen-Look zeigt.

- [ ] **Step 1: Write the failing tests**

```python
def test_cli_episode_requires_image(tmp_path):  # look=episode ohne Bild -> Exit-Code != 0
def test_cli_episode_passes_spec_and_writes_chapters(monkeypatch, tmp_path):  # render gemockt
def test_old_project_dict_loads_defaults():  # dict ohne neue Schlüssel
    assert s.look == "classic" and s.chapters == [] and s.episode_title == ""
def test_state_roundtrip_episode():  # kind, accent_word, chapters bleiben erhalten
def test_episode_plan_worker_emits_plan(qtbot):  # gemini gemockt
def test_render_worker_passes_episode(monkeypatch):
def test_preview_worker_passes_episode(monkeypatch):
```

- [ ] **Step 2: Run** `pytest tests/test_episode_cli.py tests/test_gui_state.py tests/test_gui_workers.py -v` → FAIL.
- [ ] **Step 3: Implement** Preset, CLI-Zweig, Zustand, Worker.
- [ ] **Step 4: Run** dieselben Tests plus `tests/test_cli.py tests/test_app_state.py` → PASS.
- [ ] **Step 5: Commit** `feat(folge): Preset, CLI und GUI-Zustand fuer den Folgen-Look`

---

### Task 11: Prüfliste im GUI und Kapiteltext nach dem Render

**Files:**
- Create: `src/gui/episode_panel.py`
- Modify: `src/gui/main_window.py` (Reiter „Folge“, Verdrahtung, Render-Config, Kapiteltext nach `_finish_render`)
- Test: `tests/test_episode_panel.py`, Ergänzung in `tests/test_gui_main_window.py`

**Interfaces:**
- Consumes: `AppState`-Felder (Task 10), `EpisodePlanWorker` (Task 10), `QuoteEditorDialog` (`src/gui/quote_editor.py`), `image_too_small`, `cover_crop` (Task 5), `EpisodeTimeline.card_conflicts` (Task 4), `youtube_chapter_text`/`write_chapter_file` (Task 3).
- Produces: `class EpisodePanel(QWidget)` mit `__init__(state, parent=None)`, Signalen `plan_requested()` und `preview_requested(float)`, Methoden `set_plan_error(msg: str)` und `refresh()`.

Inhalt des Reiters: Schalter „Folgen-Look verwenden“ (↔ `state.look`); Felder Folgentitel (leer → `default_episode_title(state.audio_path)` beim Laden des Audios) und Folgennummer; Knopf „KI vorbereiten“; Knöpfe „+ Kapitel“ und „+ Zitat“ (neue Zeile an der aktuellen Abspielzeit, auch nach KI-Fehler nutzbar); Hinweiszeile (Bild zu klein, mit dem Tipp „In ChatGPT ein Querformat anfordern, z. B. 1536×1024; Wichtiges nicht an den oberen und unteren Rand“; KI-Fehler). Tabelle nach Zeit: Art (Kapitel / Bühne / Zitat; bei Zitat-Zeilen umschaltbar), Zeit, Text (direkt editierbar), Schlüsselwort (nur Bühne, Auswahl aus Wörtern des Satzes), ▶ (spielt Audio ab dieser Zeit und sendet `preview_requested(t)`), ✕ (löschen). Doppelklick auf die Zeit öffnet `QuoteEditorDialog` (Zitat/Bühne) bzw. ein Zahlenfeld (Kapitel). Konfliktzeilen markiert, Tooltip „Fällt in eine Kapitel-Karte“. Jede Änderung schreibt in `state`.

`main_window.py`: Reiter nach „Parameter“; `plan_requested` → `EpisodePlanWorker`; Ergebnis setzt `state.chapters`, `state.quotes`, `state.look = "episode"`; `preview_requested(t)` → bestehende Vorschau bei `t / duration`; Render-Config bekommt `episode` bei `state.look == "episode"` (ohne Bild: Fehlermeldung, kein Render); `spec.quotes` kommt immer aus `state.quotes`, unabhängig von `quotes_enabled` (Standard `False`); im Folgen-Modus sind „Extrahieren“ und „Alle einrasten“ im alten Reiter „Zitate“ gesperrt, weil sie `state.quotes` überschreiben und die Bühnen-Sätze löschen würden; nach `_finish_render` im Folgen-Modus `write_chapter_file` (Intro-Versatz über `get_media_info(intro)["duration"]` und `main_start_offset`, falls Intro aktiv) und ein Dialog mit dem Text, den Warnungen und einem Knopf „Kopieren“.

- [ ] **Step 1: Write the failing tests**

```python
def test_panel_rows_sorted_by_time(qtbot, state):
def test_panel_delete_removes_from_state(qtbot, state):
def test_panel_toggle_kind_to_stage(qtbot, state):  # Quote.kind == "stage", accent_word gesetzt
def test_panel_marks_card_conflict(qtbot, state):
def test_panel_title_defaults_from_audio(qtbot, state):
def test_panel_warns_small_image(qtbot, state, tmp_path):  # 1024x1024-Bild
def test_panel_shows_error_keeps_title(qtbot, state):
def test_panel_add_rows_by_hand(qtbot, state):  # „+ Kapitel“ und „+ Zitat“ legen Einträge in state an
def test_main_window_render_config_has_episode(qtbot, monkeypatch):
    # state.quotes_enabled = False, look = "episode"
    assert config["episode"].quotes  # nicht leer
def test_panel_edit_time_keeps_stage(qtbot, state):  # Zeit über den Dialog ändern -> kind/accent_word bleiben
def test_old_quotes_tab_locked_in_episode_mode(qtbot, state):
def test_main_window_writes_chapter_file_after_render(qtbot, monkeypatch, tmp_path):
```

- [ ] **Step 2: Run** `pytest tests/test_episode_panel.py tests/test_gui_main_window.py -v` → FAIL.
- [ ] **Step 3: Implement** Panel und Verdrahtung.
- [ ] **Step 4: Run** `pytest tests/test_episode_panel.py tests/test_gui_main_window.py tests/test_gui_smoke.py tests/test_quotes_panel.py -v` → PASS.
- [ ] **Step 5: Commit** `feat(folge): Pruefliste und Kapiteltext im GUI`

---

### Task 12: Gesamtprüfung

**Files:** keine neuen; ggf. `SESSION_NOTES.md` (Entscheidungen anhängen).

- [ ] **Step 1: Run** `pytest tests/ -v` → alles grün (mindestens die 577 alten Tests plus die neuen; GPU-Tests laufen auf dieser Maschine).
- [ ] **Step 2: Run** einen echten Render über 60 s mit einer Sprachaufnahme und einem 1536×1024-Bild über `python main.py render <audio> --config config/podcast_folge.json -o <scratchpad>/folge_test.mp4` (Kapitel/Zitate per Hand in einer Kopie des Presets). Erwartet: Datei vorhanden, `.kapitel.txt` daneben, Renderzeit pro Frame nicht über der eines heutigen `lumina_core`-Renders bei gleicher Auflösung (Referenz 2026-11-02: ~45 ms/Frame bei 720p).
- [ ] **Step 3: Check** Frames bei Karte, Zitat und Bühne als Bilder ansehen (ffmpeg-Auszug). Erwartet: keine Farbsäume, kein Bildrand bei der Kamerafahrt, Text nicht vor dem gesprochenen Satz.
- [ ] **Step 4: Commit** `test(folge): Gesamtpruefung Folgen-Look` (nur wenn Dateien geändert wurden).
