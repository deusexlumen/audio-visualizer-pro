import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from src.gui.main_window import MainWindow


def test_main_window_has_tabs(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.right_tabs.count() == 5
    assert window.right_tabs.tabText(0) == "Parameter"
    assert window.right_tabs.tabText(1) == "Folge"
    assert window.right_tabs.tabText(2) == "KI"
    assert window.right_tabs.tabText(3) == "Zitate"
    assert window.right_tabs.tabText(4) == "Studio"


def test_render_button_shows_error_without_audio(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    with patch.object(QMessageBox, "critical", return_value=None) as mock_crit:
        qtbot.mouseClick(window.btn_render, Qt.MouseButton.LeftButton)

    assert window._render_worker is None
    mock_crit.assert_called_once()
    assert "Audio" in mock_crit.call_args[0][2]


def test_render_button_shows_error_without_features(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)

    audio = tmp_path / "test.wav"
    audio.touch()
    window.state.audio_path = str(audio)

    with patch.object(QMessageBox, "critical", return_value=None) as mock_crit:
        qtbot.mouseClick(window.btn_render, Qt.MouseButton.LeftButton)

    assert window._render_worker is None
    mock_crit.assert_called_once()
    assert "analysiert" in mock_crit.call_args[0][2]


def test_render_button_starts_render_worker(qtbot, tmp_path, dummy_audio_features):
    window = MainWindow()
    qtbot.addWidget(window)

    audio = tmp_path / "test.wav"
    audio.touch()
    window.state.audio_path = str(audio)
    window.state.features = dummy_audio_features

    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = False

    with patch("src.gui.workers.RenderWorker", return_value=mock_worker) as mock_cls, \
         patch.object(QMessageBox, "information", return_value=None):
        qtbot.mouseClick(window.btn_render, Qt.MouseButton.LeftButton)

    assert mock_cls.called
    config = mock_cls.call_args.args[0]
    assert config["audio_path"] == str(audio)
    assert config["visualizer_type"] == window.state.visualizer_type
    assert config["width"] == window.state.resolution[0]
    assert config["height"] == window.state.resolution[1]
    assert config["fps"] == window.state.render_fps
    assert config["codec"] == window.state.codec
    assert config["quality"] == window.state.quality
    assert config["gpu_encode"] == window.state.gpu_encode
    assert config["background_color"] == window.state.background_color

    mock_worker.start.assert_called_once()
    # Waehrend des Renderings: Render-Button gesperrt, Abbrechen/Fortschritt sichtbar
    assert not window.btn_render.isEnabled()
    assert window.btn_cancel.isVisibleTo(window)
    assert window.progress_bar.isVisibleTo(window)
    assert window._render_worker is mock_worker


def test_render_error_reaktiviert_ui(qtbot, tmp_path, dummy_audio_features):
    """Nach einem Render-Fehler: Dialog, UI entsperrt, Traceback geloggt."""
    window = MainWindow()
    qtbot.addWidget(window)

    audio = tmp_path / "test.wav"
    audio.touch()
    window.state.audio_path = str(audio)
    window.state.features = dummy_audio_features

    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = False

    with patch("src.gui.workers.RenderWorker", return_value=mock_worker):
        qtbot.mouseClick(window.btn_render, Qt.MouseButton.LeftButton)
    assert not window.btn_render.isEnabled()

    with patch.object(QMessageBox, "critical", return_value=None) as mock_crit, \
         patch("src.gui.main_window.logger.error") as mock_log:
        # sender() muss der Worker sein — Handler direkt mit gepatchtem sender aufrufen
        with patch.object(window, "sender", return_value=mock_worker):
            window._on_render_error("Encoder kaputt", "Traceback: ...")

    mock_crit.assert_called_once()
    assert "Encoder kaputt" in mock_crit.call_args[0][2]
    assert mock_log.called
    assert window.btn_render.isEnabled()
    assert not window.btn_cancel.isVisibleTo(window)


def test_render_finished_reaktiviert_ui(qtbot, tmp_path, dummy_audio_features):
    """Nach erfolgreichem Render: UI entsperrt, Erfolgsdialog."""
    window = MainWindow()
    qtbot.addWidget(window)

    audio = tmp_path / "test.wav"
    audio.touch()
    window.state.audio_path = str(audio)
    window.state.features = dummy_audio_features

    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = False

    with patch("src.gui.workers.RenderWorker", return_value=mock_worker):
        qtbot.mouseClick(window.btn_render, Qt.MouseButton.LeftButton)
    assert not window.btn_render.isEnabled()

    with patch("src.gui.main_window.QMessageBox") as mock_box_cls:
        mock_box = MagicMock()
        mock_box_cls.return_value = mock_box
        window._finish_render("C:/out/video.mp4")

    assert window.btn_render.isEnabled()
    assert "video.mp4" in window.status_label.text()


def test_preview_button_starts_preview_worker(qtbot, tmp_path, dummy_audio_features):
    window = MainWindow()
    qtbot.addWidget(window)

    audio = tmp_path / "test.wav"
    audio.touch()
    window.state.audio_path = str(audio)
    window.state.features = dummy_audio_features

    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = False

    with patch("src.gui.workers.PreviewWorker", return_value=mock_worker) as mock_cls:
        qtbot.mouseClick(window.btn_preview, Qt.MouseButton.LeftButton)

    assert mock_cls.called
    config = mock_cls.call_args.kwargs
    assert config["audio_path"] == str(audio)
    assert config["visualizer_type"] == window.state.visualizer_type
    assert config["background_color"] == window.state.background_color
    assert config["width"] == window.state.preview_width
    mock_worker.start.assert_called_once()
    assert window._preview_worker is mock_worker


def test_preview_timer_triggers_after_parameter_change(qtbot, tmp_path, dummy_audio_features):
    window = MainWindow()
    qtbot.addWidget(window)

    audio = tmp_path / "test.wav"
    audio.touch()
    window.state.audio_path = str(audio)
    window.state.features = dummy_audio_features

    with patch.object(window._preview_timer, "start") as mock_timer_start:
        window.state.color_mode = "fixed"
        qtbot.wait(100)
        mock_timer_start.assert_called_once_with(150)


def test_project_save_and_load_roundtrip(qtbot, tmp_path):
    """Projekt speichern und in ein frisches Fenster laden."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.state.visualizer_type = "bass_temple"
    window.state.pp_bloom = 1.5

    project = tmp_path / "test.avproj"
    window._write_project(str(project))
    assert project.exists()
    assert not window._dirty

    window2 = MainWindow()
    qtbot.addWidget(window2)
    window2._load_project(str(project))
    assert window2.state.visualizer_type == "bass_temple"
    assert window2.state.pp_bloom == 1.5
    assert not window2._dirty


def test_state_change_sets_dirty_flag(qtbot):
    """Projekt-relevante Aenderungen setzen den *-Marker im Titel."""
    window = MainWindow()
    qtbot.addWidget(window)
    window._set_dirty(False)
    window.state.pp_bloom = 1.9
    qtbot.wait(50)
    assert window._dirty
    assert window.windowTitle().endswith("*")


def test_first_supported_drop_filters_extensions(qtbot, tmp_path):
    """Der Drop-Filter akzeptiert nur bekannte Datei-Endungen."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from PyQt6.QtCore import QUrl

    window = MainWindow()
    qtbot.addWidget(window)

    def make_event(paths):
        event = MagicMock()
        mime = MagicMock()
        mime.hasUrls.return_value = True
        mime.urls.return_value = [QUrl.fromLocalFile(p) for p in paths]
        event.mimeData.return_value = mime
        return event

    # Echte Pfade aus tmp_path: ein Windows-Literal wie "C:/musik/song.mp3"
    # kommt unter Linux als "/C:/musik/song.mp3" aus QUrl zurueck.
    song = str(tmp_path / "song.mp3")
    projekt = str(tmp_path / "projekt.avproj")
    readme = str(tmp_path / "readme.txt")

    # Ueber Path vergleichen: QUrl gibt unter Windows Vorwaerts-Schraegstriche
    # zurueck, der String waere dann trotz gleichem Pfad ungleich.
    assert Path(window._first_supported_drop(make_event([song]))) == Path(song)
    assert Path(window._first_supported_drop(make_event([projekt]))) == Path(projekt)
    assert window._first_supported_drop(make_event([readme])) is None


def test_spiral_keys_trigger_preview_and_dirty_marker():
    from src.gui.main_window import MainWindow
    from src.gui.state import SPIRAL_STATE_KEYS

    assert set(SPIRAL_STATE_KEYS) <= MainWindow._PREVIEW_KEYS
    assert set(SPIRAL_STATE_KEYS) <= MainWindow._PROJECT_KEYS


def _episode_window(qtbot, tmp_path, features, with_image=True):
    from PIL import Image
    from src.types import Chapter, Quote
    window = MainWindow()
    qtbot.addWidget(window)
    # Keine echte GPU-Vorschau im Hintergrund starten
    window._preview_timer.timeout.disconnect()
    audio = tmp_path / "Folge_eins.wav"
    audio.touch()
    window.state.audio_path = str(audio)
    window.state.features = features
    window.state.audio_duration = 200.0
    if with_image:
        img = tmp_path / "bild.png"
        Image.new("RGB", (2560, 1440)).save(img)
        window.state.background_path = str(img)
    window.state.quotes_enabled = False
    window.state.chapters = [Chapter(start=0.0, title="Einstieg"),
                             Chapter(start=60.0, title="Mitte"),
                             Chapter(start=120.0, title="Schluss")]
    window.state.quotes = [Quote(text="Der Mensch stört", start_time=30, end_time=32,
                                 kind="stage", accent_word="stört")]
    window.state.look = "episode"
    return window


def test_main_window_render_config_has_episode(qtbot, tmp_path, dummy_audio_features):
    window = _episode_window(qtbot, tmp_path, dummy_audio_features)
    mock_worker = MagicMock()
    mock_worker.isRunning.return_value = False
    with patch("src.gui.workers.RenderWorker", return_value=mock_worker) as mock_cls:
        qtbot.mouseClick(window.btn_render, Qt.MouseButton.LeftButton)
    config = mock_cls.call_args.args[0]
    spec = config["episode"]
    assert spec.quotes and spec.quotes[0].kind == "stage"
    assert spec.image_path == window.state.background_path
    assert spec.title == "Folge eins"
    assert [c.title for c in spec.chapters] == ["Einstieg", "Mitte", "Schluss"]


def test_main_window_episode_without_image_blocks_render(qtbot, tmp_path, dummy_audio_features):
    window = _episode_window(qtbot, tmp_path, dummy_audio_features, with_image=False)
    with patch("src.gui.workers.RenderWorker") as mock_cls,          patch.object(QMessageBox, "critical", return_value=None) as mock_crit:
        qtbot.mouseClick(window.btn_render, Qt.MouseButton.LeftButton)
    mock_cls.assert_not_called()
    assert "Bild" in mock_crit.call_args[0][2]


def test_old_quotes_tab_locked_in_episode_mode(qtbot, tmp_path, dummy_audio_features):
    window = _episode_window(qtbot, tmp_path, dummy_audio_features)
    window.quotes_panel.gemini = MagicMock()
    window.state.look = "episode"
    assert not window.quotes_panel.btn_extract.isEnabled()
    assert not window.quotes_panel.btn_snap_all.isEnabled()
    window.state.look = "classic"
    assert window.quotes_panel.btn_extract.isEnabled()
    assert window.quotes_panel.btn_snap_all.isEnabled()


def test_main_window_writes_chapter_file_after_render(qtbot, tmp_path, dummy_audio_features):
    window = _episode_window(qtbot, tmp_path, dummy_audio_features)
    video = tmp_path / "video.mp4"
    with patch("src.gui.main_window.QMessageBox"),          patch.object(window, "_show_chapter_text") as mock_show:
        window._finish_render(str(video))
    path = tmp_path / "video.kapitel.txt"
    assert path.read_text(encoding="utf-8").startswith("0:00 Einstieg")
    text, warnings = mock_show.call_args.args
    assert "1:00 Mitte" in text and warnings == []


def test_episode_plan_result_fills_state(qtbot, tmp_path, dummy_audio_features):
    from src.episode.plan import EpisodePlan
    from src.types import Chapter, Quote
    window = _episode_window(qtbot, tmp_path, dummy_audio_features)
    window.state.look = "classic"
    plan = EpisodePlan(chapters=[Chapter(start=0.0, title="Neu")],
                       quotes=[Quote(text="x y z", start_time=5, end_time=6)])
    window._on_episode_plan_ready(plan)
    assert window.state.look == "episode"
    assert [c.title for c in window.state.chapters] == ["Neu"]
    assert window.state.quotes[0].text == "x y z"
    assert window.state.visualizer_type == "voice_line"
