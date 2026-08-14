# Modus-Erkennung: Messwerte und Kalibrierung

Stand: 2026-08-10. Betrifft `AudioAnalyzer._detect_mode_advanced` und
`_estimate_tempo_simple` in `src/analyzer.py`.

## Befund vor der Korrektur

Alle sechs Golden-Audios wurden als `music` klassifiziert — auch die drei
Podcasts. Der `speech`- und der `hybrid`-Zweig waren toter Code.

Gemessene Eingangsgrößen (echte Neuberechnung mit Temp-Cache, fps=30):

| Datei | tempo | onset_std | cent_mean | voice_mean | rms_var |
|---|---:|---:|---:|---:|---:|
| music_dunkelheit.m4a | 120.0 | 0.344 | 7341 | 0.210 | 0.163 |
| music_severance.m4a | 120.0 | 0.311 | 6055 | 0.288 | 0.154 |
| music_velvet.m4a | 120.0 | 0.171 | 7068 | 0.240 | 0.161 |
| podcast_gorilla.m4a | 120.0 | 0.291 | 5300 | 0.380 | 0.108 |
| podcast_gorilla_mid.m4a | 120.0 | 0.928 | 5553 | 0.349 | 0.106 |
| podcast_macy.m4a | 120.0 | 0.329 | 5201 | 0.372 | 0.107 |

Die alte UND-Kette lautete:

```python
is_speech = (voice_mean > 0.45) and (rms_var < 0.15) and (cent_mean < 2000)
is_music  = (tempo > 60) and (onset_std > 0.08) and (cent_mean > 1200) and (voice_mean < 0.5)
```

Drei der vier Musik-Kriterien waren konstant erfüllt, zwei der drei
Sprach-Kriterien unerfüllbar:

- **`cent_mean < 2000` unerreichbar.** Der Pre-Emphasis-Filter in `analyze()`
  (`y[1:] -= 0.97 * y[:-1]`) läuft **vor** dem STFT und kippt das Spektrum um
  ~6 dB/Oktave nach oben. Sprache liegt roh bei 500–1500 Hz Centroid, nach
  Pre-Emphasis bei ~5200 Hz. Die Schwellen 1200/2000 stammen aus einer Zeit
  ohne Pre-Emphasis.
- **`voice_mean > 0.45` unerreichbar.** Gemessenes Maximum 0.380 — ebenfalls
  eine Folge der Höhenanhebung, die den Anteil des 80–3000-Hz-Bands drückt.
- **`onset_std > 0.08` immer wahr.** `onset_env` geht roh (nicht normiert) in
  die Erkennung, die Werte liegen bei 0.17–0.93.
- **`tempo > 60` immer wahr**, weil `tempo` immer 120.0 war (siehe unten).

## Kalibrierung

Drei Merkmale trennen den Korpus sauber:

| Merkmal | Musik | Podcast | Lücke |
|---|---|---|---|
| `rms_var` | 0.154–0.163 | 0.106–0.108 | 0.108 ↔ 0.154 |
| `voice_mean` | 0.210–0.288 | 0.349–0.380 | 0.288 ↔ 0.349 |
| `cent_mean` | 6055–7341 | 5201–5553 | 5553 ↔ 6055 |

Statt harter UND-Ketten wird jedes Merkmal einzeln auf einen Score in
[-1, +1] abgebildet (+1 = klar Sprache) und gemittelt. Die Kanten liegen
bewusst außerhalb der gemessenen Lücken, damit unbekanntes Material nicht an
einer haarscharfen Schwelle kippt:

| Merkmal | Sprache-Kante | Musik-Kante |
|---|---|---|
| `rms_var` | ≤ 0.10 | ≥ 0.17 |
| `voice_mean` | ≥ 0.35 | ≤ 0.28 |
| `cent_mean` | ≤ 5400 | ≥ 6300 |

Entscheidung ab `|score| > 0.30`, sonst `hybrid`. Der Wert liegt knapp unter
1/3: zwei voll ausgeschlagene Merkmale setzen sich damit gegen ein
gegenläufiges drittes durch, echte Mischfälle bleiben `hybrid`.

Erreichte Scores am Korpus: Musik −0.59 bis −0.93, Podcast +0.82 bis +0.93.

**Tempo und Onset-Streuung gehen nicht mehr ein.** Sprache bekommt vom
Tempo-Schätzer ebenso plausible BPM-Werte wie Musik, und `onset_std` trennt
am Korpus nicht — die Podcasts streuen dort sogar stärker als die Musik.

### Grenzen dieser Kalibrierung

Sechs Dateien sind eine dünne Grundlage. Die Kanten sind an drei Lücken
angepasst und nicht an unabhängigem Material validiert. Wenn Fehlklassi-
fikationen auftauchen: erst die Kennwerte der betroffenen Datei messen
(Skript-Muster siehe unten), dann Kanten anpassen — nicht raten.

Alle drei Merkmale hängen am Pre-Emphasis-Filter. Wird der geändert oder
entfernt, sind sämtliche Kanten hinfällig.

## Tempo-Schätzung

`_estimate_tempo_simple` lieferte ausnahmslos 120.0. Ursache: `argmax` über
den gemittelten Tempogram trifft Lag 0 (dort ist die Energie immer maximal),
und `librosa.tempo_frequencies` gibt an Index 0 `inf` zurück. `inf > 250`
griff den Fallback.

Jetzt: `librosa.feature.rhythm.tempo` (log-normaler Prior um 120 BPM, fängt
Oktav-Fehler ab), Fallback auf das Tempogram-Maximum **mit Maskierung**
ungültiger Lags, erst dann 120.0. Ergebnisse: 139.7 / 112.3 / 120.2 BPM für
die drei Musikstücke.

Für Sprache liefert der Schätzer weiterhin beliebige Werte (103–154 BPM) —
das ist erwartbar und ohne Bedeutung, `tempo` ist bei Sprache kein
sinnvolles Merkmal.

## Kennwerte selbst messen

`_detect_mode_advanced` monkeypatchen, um die Eingangsgrößen mitzuschreiben,
und den Analyzer mit einem Temp-Cache-Verzeichnis instanziieren, damit
wirklich neu gerechnet wird:

```python
an = AudioAnalyzer(cache_dir=tmpdir)   # umgeht den Feature-Cache
an.analyze(pfad, fps=30)
```

## Cache

Die Korrektur ändert `mode` und `tempo` in den gecachten Features.
`CACHE_VERSION` wurde deshalb von 8 auf 9 gezogen — alle bestehenden
Feature-Caches werden verworfen und bei der nächsten Nutzung neu berechnet.
