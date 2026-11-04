"""Seuil de décision par le coût : la théorie doit se retrouver sur les données, et le test ne doit pas servir.

Données synthétiques fabriquées ici : ces tests tournent partout (CI comprise), sans fichier externe.
"""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import decision as DEC
from app.main import app
from test_validation import _session, _synthetic

SPEC = {"name": "logreg"}


def test_theoretical_threshold_values():
    assert DEC.theoretical_threshold(1, 1) == 0.5                      # erreurs de même prix -> seuil 50 %
    assert abs(DEC.theoretical_threshold(5, 1) - 1 / 6) < 1e-12        # manquer coûte 5x plus -> on refuse dès 16,7 %
    with pytest.raises(ValueError):
        DEC.theoretical_threshold(0, 1)


def test_cost_at_is_exact_on_a_tiny_example():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    r = DEC.cost_at(y, p, 0.5, cost_fn=10, cost_fp=1)
    assert (r["fn"], r["fp"]) == (1, 1)                                # le 2e défaut est manqué, le 3e sain refusé
    assert abs(r["cost"] - (10 + 1) / 5) < 1e-12
    assert r["refuse_rate"] == 0.4 and r["recall"] == 0.5 and r["precision"] == 0.5


def test_decision_never_reads_the_test_set():
    s = _session()
    s.X_test_raw, s.y_test, s.M_test, s.Z_test = None, None, None, None
    r = DEC.decision_report(s, SPEC, cost_fn=5, cost_fp=1, n_splits=3)
    assert r["test_set_used"] is False


def test_best_threshold_is_never_worse_than_the_habit_and_follows_theory():
    s = _session()
    r = DEC.decision_report(s, SPEC, cost_fn=5, cost_fp=1, n_splits=5)
    assert r["best"]["cost"] <= r["at_half"]["cost"] + 1e-12
    assert r["best"]["cost"] <= min(r["accept_all"], r["refuse_all"]) + 1e-12
    assert abs(r["best"]["threshold"] - r["theoretical_threshold"]) < 0.2   # probabilités fiables -> théorie ≈ données
    assert r["at_theoretical"]["cost"] - r["best"]["cost"] < 0.02            # la courbe est plate autour de l'optimum


def test_costlier_misses_lower_the_threshold():
    s = _session()
    cheap = DEC.decision_report(s, SPEC, cost_fn=2, cost_fp=1, n_splits=3)
    dear = DEC.decision_report(s, SPEC, cost_fn=10, cost_fp=1, n_splits=3)
    assert dear["theoretical_threshold"] < cheap["theoretical_threshold"]
    assert dear["best"]["refuse_rate"] >= cheap["best"]["refuse_rate"]


def test_threshold_endpoint_through_csv_upload():
    c = TestClient(app)
    H = {"X-Session-Id": "pytest-thr"}
    buf = io.StringIO()
    _synthetic().to_csv(buf, index=False)
    assert c.post("/api/data/upload", files={"file": ("synthetic.csv", buf.getvalue().encode(), "text/csv")},
                  data={"target": "y"}, headers=H).status_code == 200
    assert c.post("/api/prep", json={"features": ["x1", "x2", "cat"]}, headers=H).status_code == 200
    r = c.post("/api/models/threshold", json={"model": {"name": "logreg"}, "cost_fn": 5, "cost_fp": 1, "n_splits": 3},
               headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["test_set_used"] is False and len(r.json()["curve"]) == 99
    bad = c.post("/api/models/threshold", json={"model": {"name": "logreg"}, "cost_fn": 0}, headers=H)
    assert bad.status_code == 422                                      # coût nul refusé par la validation
    assert c.post("/api/models/threshold", json={"model": {"name": "nope"}}, headers=H).status_code == 400


def test_threshold_requires_prepared_data():
    c = TestClient(app)
    r = c.post("/api/models/threshold", json={"model": {"name": "logreg"}}, headers={"X-Session-Id": "vide-thr"})
    assert r.status_code == 400
