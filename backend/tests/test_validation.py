"""Validation croisée : le lot de test ne doit jamais être touché pendant la comparaison.

Données synthétiques fabriquées ici : ces tests tournent partout (CI comprise), sans fichier externe.
"""
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import prep as P
from app import validation as V
from app.main import app
from app.state import Session


def _synthetic(n=1500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    logit = 1.3 * x1 - 0.6 * x2 + 0.9 * (cat == "c") - 0.7
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    x1 = pd.Series(x1).mask(rng.random(n) < 0.05)              # 5 % de valeurs manquantes
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y.astype(str)})


def _session() -> Session:
    s = Session()
    s.df, s.target, s.positive, s.name = _synthetic(), "y", "1", "synthetic"
    P.build(s, {"features": ["x1", "x2", "cat"], "test_size": 0.25, "random_state": 42,
                "num_impute": "median", "cat_impute": "mode", "scaling": "standard", "stratify": True})
    return s


SPECS = [{"name": "logreg"}, {"name": "tree", "params": {"max_depth": 4}}]


def test_cross_validation_never_reads_the_test_set():
    """Preuve par l'absurde : on détruit le lot de test ; la comparaison doit marcher quand même."""
    s = _session()
    s.X_test_raw, s.y_test, s.M_test, s.Z_test = None, None, None, None
    r = V.cross_validate(s, SPECS, n_splits=3, n_repeats=1)
    assert r["test_set_used"] is False and set(r["models"]) == {"logreg", "tree"}


def test_cv_summary_is_coherent():
    s = _session()
    r = V.cross_validate(s, SPECS, n_splits=4, n_repeats=2)
    assert r["settings"]["n_fits_per_model"] == 8
    for m in r["models"].values():
        assert m["auc"]["n"] == 8 and 0.6 < m["auc"]["mean"] < 1.0
        assert m["auc"]["std"] >= 0 and m["auc"]["min"] <= m["auc"]["mean"] <= m["auc"]["max"]
    best = r["models"][r["best"]]
    assert best["best"] and best["tied_with_best"]            # le meilleur est toujours « à égalité avec lui-même »


def test_cv_is_reproducible():
    s = _session()
    a = V.cross_validate(s, SPECS, n_splits=3, n_repeats=1, seed=7)
    b = V.cross_validate(s, SPECS, n_splits=3, n_repeats=1, seed=7)
    assert a["models"]["logreg"]["auc"] == b["models"]["logreg"]["auc"]


def test_refit_and_shortcut_give_close_results():
    """Le raccourci (préparation unique) ne doit pas s'écarter beaucoup de la méthode correcte."""
    s = _session()
    ok = V.cross_validate(s, SPECS, n_splits=4, n_repeats=1, refit_prep=True)
    quick = V.cross_validate(s, SPECS, n_splits=4, n_repeats=1, refit_prep=False)
    for k in ok["models"]:
        assert abs(ok["models"][k]["auc"]["mean"] - quick["models"][k]["auc"]["mean"]) < 0.02


def test_compare_endpoint_through_csv_upload():
    c = TestClient(app)
    H = {"X-Session-Id": "pytest-cv"}
    buf = io.StringIO()
    _synthetic().to_csv(buf, index=False)
    r = c.post("/api/data/upload", files={"file": ("synthetic.csv", buf.getvalue().encode(), "text/csv")},
               data={"target": "y"}, headers=H)
    assert r.status_code == 200, r.text
    assert c.post("/api/prep", json={"features": ["x1", "x2", "cat"]}, headers=H).status_code == 200
    r = c.post("/api/models/compare", json={"models": [{"name": "logreg"}, {"name": "tree"}],
                                           "n_splits": 3, "n_repeats": 1}, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["test_set_used"] is False and r.json()["best"] in ("logreg", "tree")


def test_compare_requires_prepared_data():
    c = TestClient(app)
    r = c.post("/api/models/compare", json={"models": [{"name": "logreg"}]}, headers={"X-Session-Id": "vide-cv"})
    assert r.status_code == 400


def test_unknown_model_is_rejected():
    c = TestClient(app)
    H = {"X-Session-Id": "pytest-cv"}
    r = c.post("/api/models/compare", json={"models": [{"name": "nope"}]}, headers=H)
    assert r.status_code in (400, 422)
