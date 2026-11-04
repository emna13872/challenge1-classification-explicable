"""Calibration : les probabilités annoncées doivent ressembler aux fréquences réelles.

Données synthétiques fabriquées ici : ces tests tournent partout (CI comprise), sans fichier externe.
"""
import io

import numpy as np
from fastapi.testclient import TestClient

from app import calibration as C
from app.main import app
from app.state import Session
from test_validation import _session, _synthetic

SPEC = {"name": "logreg"}


def test_brier_and_ece_on_known_cases():
    y = np.array([0, 0, 1, 1])
    assert C.brier(y, np.array([0, 0, 1, 1])) == 0.0                       # parfait
    assert abs(C.brier(y, np.full(4, 0.5)) - 0.25) < 1e-12                 # « je ne sais pas » = 0,25
    perfect = [{"n": 10, "mean_pred": 0.3, "frac_pos": 0.3}, {"n": 10, "mean_pred": 0.8, "frac_pos": 0.8}]
    assert C.ece(perfect) == 0.0
    assert abs(C.ece([{"n": 10, "mean_pred": 0.4, "frac_pos": 0.1}]) - 0.3) < 1e-12


def test_calibration_never_reads_the_test_set():
    s = _session()
    s.X_test_raw, s.y_test, s.M_test, s.Z_test = None, None, None, None
    r = C.calibration_report(s, SPEC, n_splits=3)
    assert r["test_set_used"] is False and set(r["methods"]) == {"raw", "sigmoid", "isotonic"}


def test_balanced_classes_inflate_probabilities_and_calibration_repairs_them():
    s = _session()
    r = C.calibration_report(s, SPEC, n_splits=5, balance=True)
    raw = r["methods"]["raw"]
    assert raw["mean_pred"] > r["base_rate"] + 0.05                        # le rééquilibrage gonfle les probabilités
    assert r["recommended"] in ("sigmoid", "isotonic")
    fixed = r["methods"][r["recommended"]]
    assert fixed["brier"] < raw["brier"]
    assert abs(fixed["mean_pred"] - r["base_rate"]) < abs(raw["mean_pred"] - r["base_rate"])


def test_well_specified_model_needs_no_correction():
    s = _session()
    r = C.calibration_report(s, SPEC, n_splits=5)
    assert r["methods"]["raw"]["ece"] < 0.05
    assert r["methods"]["raw"]["brier"] < r["base_brier"]                  # mieux que « toujours le taux moyen »


def test_bins_are_coherent():
    s = _session()
    r = C.calibration_report(s, SPEC, n_splits=3)
    for m in r["methods"].values():
        assert sum(b["n"] for b in m["bins"]) == r["settings"]["n_rows_used"]
        assert all(0 <= b["mean_pred"] <= 1 and 0 <= b["frac_pos"] <= 1 for b in m["bins"])


def test_calibration_endpoint_through_csv_upload():
    c = TestClient(app)
    H = {"X-Session-Id": "pytest-calib"}
    buf = io.StringIO()
    _synthetic().to_csv(buf, index=False)
    assert c.post("/api/data/upload", files={"file": ("synthetic.csv", buf.getvalue().encode(), "text/csv")},
                  data={"target": "y"}, headers=H).status_code == 200
    assert c.post("/api/prep", json={"features": ["x1", "x2", "cat"]}, headers=H).status_code == 200
    r = c.post("/api/models/calibration", json={"model": {"name": "logreg"}, "balance": True, "n_splits": 3}, headers=H)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["test_set_used"] is False and j["recommended"] in ("raw", "sigmoid", "isotonic")
    assert c.post("/api/models/calibration", json={"model": {"name": "nope"}}, headers=H).status_code == 400


def test_calibration_requires_prepared_data():
    c = TestClient(app)
    r = c.post("/api/models/calibration", json={"model": {"name": "logreg"}}, headers={"X-Session-Id": "vide-calib"})
    assert r.status_code == 400
