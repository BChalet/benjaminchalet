"""Préparer des téléchargements sans laisser de fichiers temporaires.

La logique financière appartient au moteur. Ce module réutilise les exports
de OptionProfiles pour proposer exactement les mêmes figures dans Streamlit
et dans les scripts. Chaque appel possède son propre dossier temporaire :
deux utilisateurs ne peuvent pas écraser les résultats l'un de l'autre.
"""

from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
import gc

from interface_calculs import Parameters, make_tree
from profils_option import OptionProfiles


def scenario_json(params: Parameters) -> str:
    """Les dates ISO rendent un scénario lisible et reproductible."""
    return json.dumps(asdict(params), default=str, ensure_ascii=False, indent=2)


def profile_archive(params: Parameters) -> bytes:
    """Retourner un ZIP en mémoire : 2 PNG, 2 SVG, données CSV et paramètres.

    Le calcul est volontairement explicite, déclenché par un bouton. Garder
    les octets du ZIP en cache ne conserve aucun nœud de l'arbre en mémoire.
    """
    params.validate()
    tree = None
    try:
        tree = make_tree(params)
        with TemporaryDirectory(prefix="profils_option_") as directory:
            files = OptionProfiles(tree).plot(directory, show=False)
            buffer = BytesIO()
            with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
                for path in files.values():
                    archive.writestr(Path(path).name, Path(path).read_bytes())
                archive.writestr("parametres.json", scenario_json(params))
            return buffer.getvalue()
    finally:
        del tree
        gc.collect()
