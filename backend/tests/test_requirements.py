"""Chaque bibliothèque importée par l'application doit être déclarée dans requirements.txt.

Né d'une vraie erreur (4 novembre) : `httpx` n'était déclaré que pour les tests (requirements-dev.txt). Tout
marchait ici, où les deux listes sont installées, mais le serveur ne démarrait plus sur le PC d'Emna, qui
n'installe que requirements.txt (comme l'image Docker de production).
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
# nom importé -> paquet à déclarer (ou paquet qui l'installe déjà : pydantic vient avec fastapi, scipy avec scikit-learn)
PACKAGE = {"sklearn": "scikit-learn", "pydantic": "fastapi", "starlette": "fastapi", "scipy": "scikit-learn",
           "multipart": "python-multipart"}


def _declared() -> set[str]:
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return {re.split(r"[<>=!~\[;\s]", l.strip())[0].lower() for l in lines if l.strip() and not l.startswith("#")}


def test_every_runtime_import_is_declared():
    declared, missing = _declared(), []
    for f in sorted((ROOT / "app").glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for n in names:
                top = n.split(".")[0]
                if top in sys.stdlib_module_names:
                    continue
                pkg = PACKAGE.get(top, top).lower()
                if pkg not in declared:
                    missing.append(f"{f.name} importe « {top} » : ajouter « {pkg} » à requirements.txt")
    assert not missing, "\n".join(missing)
