# Projekt-Export (ZIP-Bundle mit Asset-Check und Manifest)

Status: Entwurf, unverifiziert
Datum: 2026-08-09

> Hinweis zur Ablage: Der Skill verlangt `docs/design/<slug>/design.md`.
> Wegen der Session-Restriktion „Schreibzugriffe nur unter .skill-test/s1g/"
> liegt das Doc hier. Status bleibt deshalb "Entwurf, unverifiziert" —
> Beleglage wird laufend gepflegt, Repo-Schreiben findet nicht statt.

## 0. Engpass und Fundament

### a) Ist der Projekt-Export der Engpass?

Das Vorhaben ist ein explizit vom Nutzer bestelltes Feature mit konkretem
Anlass (Demo). Der Engpass ist hier nicht „wird das Feature benutzt", sondern
„kann ein Projekt vom Rechner des Erstellers auf einen anderen Rechner (oder
in ein Backup) übertragen werden, ohne dass Assets verloren gehen". Genau das
ist heute der Fall: die Projekt-JSON referenziert Assets über **absolute
Pfade** (`projects/podcast.json` Zeile 3: `audio_path: "C:\\Users\\Buxe\\
Downloads\\..."`, Zeile 4: `background_path` ebenso unter Downloads)
`[belegt]`. Eine solche Projektdatei ist außerhalb des eigenen Systems
wertlos. Der Export löst also ein reales, heute nachweisbar vorhandenes
Problem. Verdikt: ja, das ist ein echter Engpass.

### b) Ist die Schicht darunter intakt?

Der Export setzt voraus, dass es (1) einen verlässlichen Ort gibt, an dem der
komplette Projekt-Zustand als JSON vorliegt, und (2) dass alle Assets darin
auffindbar referenziert sind.

- (1) ist gegeben: `src/gui/main_window.py:230` `_save_project`,
  `:236` `_save_project_as`, `:270` `_load_project`, `self._project_path`
  in `:64` `[belegt]`. Das Projektformat ist flaches JSON, `version: 1`
  `[belegt, projects/podcast.json:2]`.
- (2) ist nur teilweise gegeben — und das ist die wackelige Stelle:
  - Audio: `audio_path`, absolut `[belegt]`
  - Hintergrund: `background_path`, absolut `[belegt]`
  - „Verwendete Configs": im Projekt-JSON gibt es **kein** Config-Feld;
    Parameter stehen inline in `viz_extra_params`, `quote_config` etc.
    `[belegt, projects/podcast.json]`. Welche Config-Preset-Datei
    ursprünglich geladen wurde, ist nicht persistiert `[vermutet, muss in
    Stufe 2 an main_window.py/state.py verifiziert werden]`.

Heißt: Das Fundament trägt das Feature, aber „verwendete Configs exportieren"
ist im aktuellen Projektformat möglicherweise gar nicht abbildbar, weil die
Config-Herkunft nicht gespeichert wird. Kein bekannter Defekt im Sinne von
„kaputter Code darunter" — aber eine echte Lücke, die als Annahme/Lücke in
Stufe 2 und im Report (Stufe 5) geführt wird. Keine P1-Blocker bekannt.

**Verdikt Stufe 0: Weitermachen.** Export ist der Engpass, Fundament intakt,
eine Format-Lücke (Config-Herkunft) wird explizit geführt statt still
grün gemalt.

── Checkpoint Stufe 0 ──
Entschieden: Export-Feature ist gerechtfertigt (Projekt-JSON nutzt absolute
Pfade und ist heute nicht portabel). Bauziel bestätigt.
Offen: Alles Produkt-, Architektur- und Design-Level; konkret: was „verwendete
Configs" bedeutet, wenn das Projektformat keine Config-Referenz kennt.
Wackelig: Die Annahme, dass Configs überhaupt exportierbar referenziert sind.
Wenn die Herkunft nirgends persistiert ist, kippt der Export-Umfang auf
„inline-Parameter gelten als Stand der Config" — das würde den Umfang des
Features sichtbar verkleinern und muss vor Stufe 3 geklärt sein.
