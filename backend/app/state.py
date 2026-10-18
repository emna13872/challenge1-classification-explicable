"""État de session côté serveur (un état par identifiant de session)."""
from collections import OrderedDict
import math

import numpy as np
import pandas as pd


class Session:
    def __init__(self):
        self.df = None            # jeu de données complet
        self.name = ""            # nom du jeu de données
        self.target = None        # colonne cible
        self.positive = None      # modalité d'intérêt (str)
        # préparation
        self.features = []
        self.prep_cfg = {}
        self.imp = None           # ColumnTransformer (imputation + encodage)
        self.scaler = None        # normalisation (sur les colonnes numériques)
        self.n_num = 0
        self.meta = []            # métadonnées par colonne transformée
        self.names = []           # noms des colonnes transformées
        self.X_train_raw = None
        self.X_test_raw = None
        self.y_train = None
        self.y_test = None
        self.M_train = None       # espace "affichage" (imputé + encodé, non normalisé)
        self.M_test = None
        self.Z_train = None       # espace "modèle" (normalisé)
        self.Z_test = None
        # modèles
        self.models = {}
        self.cache = {}


_SESSIONS: "OrderedDict[str, Session]" = OrderedDict()
MAX_SESSIONS = 20


def get_session(sid: str) -> Session:
    sid = sid or "default"
    if sid in _SESSIONS:
        _SESSIONS.move_to_end(sid)
        return _SESSIONS[sid]
    s = Session()
    _SESSIONS[sid] = s
    while len(_SESSIONS) > MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    return s


def clean(o):
    """Rend n'importe quel objet sérialisable en JSON strict (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if o is pd.NA:
        return None
    return o
