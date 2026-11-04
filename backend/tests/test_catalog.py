"""Le catalogue est la source unique : le reste de l'application lit le type du modèle, jamais son nom."""
import pytest

from app import explain as X
from app import models as MD
from test_validation import _session

KINDS = {"linear", "tree", "knn"}


def test_every_model_declares_what_the_rest_of_the_app_needs():
    for name, spec in MD.CATALOG.items():
        assert spec["kind"] in KINDS, name
        assert spec["shap_space"] in ("probabilité", "log-odds"), name
        if spec["kind"] == "tree":
            assert spec["importance_label"], name


def test_a_new_tree_model_is_handled_without_touching_explain(monkeypatch):
    """On invente un modèle « extra » de type arbre : explain.py doit le traiter d'après le catalogue seul."""
    monkeypatch.setitem(MD.CATALOG, "extra", {**MD.CATALOG["tree"], "label": "Arbre extra",
                                              "shap_space": "log-odds", "importance_label": "Importance (extra)"})
    s = _session()
    est = MD.make("tree", {"max_depth": 3}, False, 1.0)
    est.fit(s.Z_train, s.y_train)
    s.models["extra"] = {"est": est, "info": {}, "p_test": est.predict_proba(s.Z_test)[:, 1], "label": "Arbre extra"}
    assert X.shap_space("extra") == "log-odds"
    assert X.native_importance(s, "extra")["label"] == "Importance (extra)"


def test_unknown_model_fails_loudly():
    with pytest.raises(KeyError):
        X.shap_space("inconnu")
