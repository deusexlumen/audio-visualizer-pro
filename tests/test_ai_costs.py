"""Tests fuer Kosten-Tracking und Result-Cache."""

import tempfile
from pathlib import Path

from src.ai_costs import CostLedger, get_cost_ledger
from src import quote_cache


def test_ledger_akkumuliert_tokens_und_kosten():
    ledger = CostLedger()
    c1 = ledger.record("gemini-2.5-flash-lite", 1_000_000, 1_000_000)
    # 0.10 (input) + 0.40 (output) pro Mio
    assert abs(c1 - 0.50) < 1e-6
    ledger.record("gemini-2.5-flash-lite", 500_000, 0)
    assert ledger.calls == 2
    assert ledger.total_tokens == 2_500_000
    assert ledger.cost_usd > 0.5


def test_ledger_unbekanntes_modell_nutzt_default():
    ledger = CostLedger()
    cost = ledger.record("voellig-neues-modell", 1_000_000, 0)
    # default input 0.15
    assert abs(cost - 0.15) < 1e-6


def test_ledger_summary_und_reset():
    ledger = CostLedger()
    assert "keine" in ledger.summary().lower()
    ledger.record("gemini-2.5-flash-lite", 1000, 1000)
    assert "$" in ledger.summary()
    ledger.reset()
    assert ledger.calls == 0


def test_get_cost_ledger_ist_singleton():
    assert get_cost_ledger() is get_cost_ledger()


def test_json_result_cache_roundtrip(tmp_path, monkeypatch):
    # Cache-Verzeichnis in tmp umlenken
    monkeypatch.setattr(quote_cache, "_get_cache_dir", lambda: tmp_path)

    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"dummy audio content")

    assert quote_cache.load_json_result(str(audio), "quotes", "sig-a") is None

    data = [{"text": "Hallo", "start_time": 1.0, "end_time": 2.0, "confidence": 0.9}]
    quote_cache.save_json_result(str(audio), "quotes", "sig-a", data)

    loaded = quote_cache.load_json_result(str(audio), "quotes", "sig-a")
    assert loaded == data
    # Andere Signatur -> kein Treffer
    assert quote_cache.load_json_result(str(audio), "quotes", "sig-b") is None
