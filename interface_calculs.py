"""Couche de calcul de l'interface interactive, indépendante de Streamlit.

Ce fichier relie les contrôles de l'écran aux classes du projet existant.
Il ne contient ni widget, ni graphique, ni formule de pricing supplémentaire :
``Tree`` reste responsable des prix et des probabilités ; ``Convergence``
fournit la formule européenne de Black-Scholes déjà utilisée dans le projet.

Séparer les calculs de l'écran présente deux avantages pédagogiques : on peut
tester les résultats sans navigateur et comprendre les algorithmes sans lire
le code de mise en page. Les fonctions retournent seulement des nombres, des
dates, des chaînes ou des DataFrames de ces valeurs. Elles ne renvoient jamais
l'arbre et ne rangent jamais ses nœuds dans une collection. Un cache de l'écran
peut donc mémoriser ces résultats sans conserver un grand graphe de nœuds.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, replace
import datetime as dt
import gc
from io import StringIO
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal

import pandas as pd

from arbre_binomial import Market, Model, Option, Tree
from convergence import Convergence
from debug_arbre import DebugTree


# Ces limites protègent un lancement interactif contre un calcul involontaire
# de plusieurs millions de nœuds par scénario. Elles ne changent pas le moteur.
MAX_STEPS = 2_000
MAX_STUDY_POINTS = 20
MAX_STRIKES = 81
MAX_DEBUG_STEPS = 40


def _finite_number(name: str, value: float) -> float:
    """Refuser aussi bool, NaN et infini, puis rendre un float Python."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} doit être un nombre réel fini.")
    return float(value)


def _validate_steps(value: int, *, maximum: int = MAX_STEPS) -> int:
    """Une grille comporte un entier de pas, sans arrondir 20,5 vers 20."""
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= maximum:
        raise ValueError(f"Le nombre de pas doit être un entier entre 1 et {maximum}.")
    return int(value)


def _validate_threshold(value: float) -> float:
    value = _finite_number("Le seuil de pruning", value)
    if not 0 <= value <= 1:
        raise ValueError("Le seuil de pruning doit être compris entre 0 et 1.")
    return value


def _validate_date(name: str, value: dt.date) -> None:
    # datetime hérite de date en Python, mais le moteur demande une date seule.
    if not isinstance(value, dt.date) or isinstance(value, dt.datetime):
        raise ValueError(f"{name} doit être une date sans heure (datetime.date).")


@dataclass(frozen=True)
class Parameters:
    """Photographie immuable des paramètres saisis dans le formulaire.

    Les taux sont des décimaux : 0.03 signifie 3 %, et non 0,03 %.
    Le rendement continu ``dividend_yield`` et le cash ``dividend_amount``
    peuvent coexister. Le moteur gère un seul détachement cash : sa date doit
    être fournie dès que son montant est positif. Hors de la vie du contrat,
    ce cash ne modifie pas l'arbre ; voir ``has_active_cash`` ci-dessous.

    ``frozen=True`` interdit de modifier silencieusement un résultat déjà
    calculé. Pour changer un champ : ``replace(params, rate=0.0)`` produit
    une nouvelle photographie, sans altérer celle affichée auparavant.
    """

    pricing_date: dt.date
    maturity: dt.date
    spot: float
    strike: float
    rate: float
    vol: float
    dividend_yield: float
    dividend_amount: float
    ex_dividend_date: dt.date | None
    nb_steps: int
    pruning_threshold: float
    option_type: Literal["Call", "Put"]
    exercise_type: Literal["European", "American"]

    def validate(self) -> None:
        """Vérifier tous les champs avant de commencer un calcul coûteux."""
        _validate_date("La date de calcul", self.pricing_date)
        _validate_date("La maturité", self.maturity)
        if self.maturity <= self.pricing_date:
            raise ValueError("La maturité doit être postérieure à la date de calcul.")
        for name in ("spot", "strike", "rate", "vol", "dividend_yield", "dividend_amount"):
            _finite_number(name, getattr(self, name))
        if self.spot <= 0:
            raise ValueError("Le spot doit être strictement positif.")
        if min(self.strike, self.vol, self.dividend_yield, self.dividend_amount) < 0:
            raise ValueError("Strike, volatilité, rendement continu et cash doivent être positifs ou nuls.")
        if self.ex_dividend_date is not None:
            _validate_date("La date de détachement", self.ex_dividend_date)
        if self.dividend_amount > 0 and self.ex_dividend_date is None:
            raise ValueError("Un dividende cash positif nécessite sa date de détachement.")
        _validate_steps(self.nb_steps)
        _validate_threshold(self.pruning_threshold)
        if self.option_type not in ("Call", "Put"):
            raise ValueError("Le type d'option doit être 'Call' ou 'Put'.")
        if self.exercise_type not in ("European", "American"):
            raise ValueError("Le type d'exercice doit être 'European' ou 'American'.")


def has_active_cash(params: Parameters) -> bool:
    """Un cash à la date initiale est déjà détaché ; un cash après T est futur."""
    return bool(
        params.dividend_amount > 0
        and params.ex_dividend_date is not None
        and params.pricing_date < params.ex_dividend_date <= params.maturity
    )


def make_tree(
    params: Parameters,
    nb_steps: int | None = None,
    pruning_threshold: float | None = None,
) -> Tree:
    """Traduire les paramètres de l'écran dans l'API historique et construire.

    ``nb_steps`` et ``pruning_threshold`` sont des substitutions explicites :
    les études peuvent ainsi isoler N, le pruning ou la taille du SVG sans
    changer les paramètres de la valorisation principale.
    """
    params.validate()
    steps = _validate_steps(params.nb_steps if nb_steps is None else nb_steps)
    threshold = _validate_threshold(
        params.pruning_threshold if pruning_threshold is None else pruning_threshold
    )
    return Tree(
        model=Model(params.pricing_date, steps, threshold),
        option=Option(params.strike, params.maturity, params.option_type, params.exercise_type),
        market=Market(
            spot=params.spot,
            rate=params.rate,
            vol=params.vol,
            dividend_yield=params.dividend_yield,
            dividend_amount=params.dividend_amount,
            ex_dividend_date=params.ex_dividend_date,
        ),
    ).build()


def _price(tree: Tree) -> float:
    """Ne jamais afficher un prix NaN/inf comme s'il était un résultat valide."""
    value = float(tree.price())
    if not isfinite(value):
        raise ValueError("Le calcul produit un prix non fini ; réduire les paramètres extrêmes.")
    return value


def _four_prices(tree: Tree) -> dict[str, float]:
    """Revaloriser quatre contrats sur UN seul arbre du même sous-jacent.

    Strike, call/put et exercice ne changent pas la construction de l'arbre.
    Chaque appel de price() remplace cependant les valeurs et décisions des
    nœuds. On remet donc l'objet Option d'origine à la fin ; le demandeur qui
    désire son profil appelle ensuite exercise_profile(), qui le revalorise.
    """
    original = tree.option
    result: dict[str, float] = {}
    try:
        for kind in ("Call", "Put"):
            for exercise in ("European", "American"):
                tree.option = Option(original.strike, original.maturity, kind, exercise)
                result[f"{kind.lower()}_{exercise.lower()}"] = _price(tree)
    finally:
        tree.option = original
    return result


def _black_scholes(params: Parameters) -> Convergence:
    """Construire la référence européenne seulement lorsqu'elle est exacte.

    Le cours ne donne pas une formule Black-Scholes exacte pour un cash actif.
    En revanche un montant dont la date est hors contrat est sans effet : on
    le retire seulement de la référence pour respecter cette distinction.
    La référence ne dépend pas du pruning et garde le rendement continu q.
    """
    if has_active_cash(params):
        raise ValueError(
            "Black-Scholes n'est pas une référence exacte avec un dividende cash actif. "
            "Désactiver le cash pour cette étude."
        )
    return Convergence(
        Market(params.spot, params.rate, params.vol, params.dividend_yield),
        Option(params.strike, params.maturity, params.option_type, "European"),
        params.pricing_date,
    )


def calculate_summary(params: Parameters) -> dict:
    """Prix, comparaisons et profils correspondant au contrat choisi à l'écran.

    L'arbre est construit une seule fois. Après les quatre prix de comparaison,
    le profil recalcule le contrat initial : on ne doit surtout pas afficher
    ses probabilités avec les décisions du dernier put américain comparé.

    ``mass_error`` mesure max|survie + exercice cumulé − 1| ; ``value_error``
    mesure max|valeur vivante actualisée + flux passés − prix initial|.
    Ces contrôles concernent le modèle effectivement calculé, élagué ou non.
    Ils ne disent pas que le pruning conserve les moments du modèle complet.
    """
    params.validate()
    gc.collect()
    start = perf_counter()
    tree = None
    try:
        tree = make_tree(params)
        four = _four_prices(tree)
        rows = tree.exercise_profile()
        price = float(tree.root.option_value)
        if not isfinite(price):
            raise ValueError("Le calcul produit un prix non fini.")
        # asdict ne porte que sur les dataclasses de résultats, jamais sur un
        # nœud. Ces colonnes gardent les noms d'origine du moteur et du CSV.
        profile = pd.DataFrame([asdict(row) for row in rows])
        comparison = pd.DataFrame([
            {"option_type": kind, "exercise_type": exercise,
             "price": four[f"{kind.lower()}_{exercise.lower()}"]}
            for kind in ("Call", "Put")
            for exercise in ("European", "American")
        ])
        black_price = None
        if has_active_cash(params):
            black_reason = "Pas de référence Black-Scholes exacte avec un dividende cash actif."
        elif params.exercise_type == "American":
            black_reason = "Black-Scholes valorise le contrat européen ; le contrat choisi est américain."
        else:
            black_price = float(_black_scholes(params).black_price(params.strike))
            black_reason = "Référence européenne exacte avec rendement continu q, sans cash actif."
        terminal = rows[-1]
        return {
            "price": price,
            "comparison": comparison,
            "node_count": tree.count_nodes(),
            # Temps de la synthèse entière : construction, quatre prix, profil.
            "elapsed_seconds": perf_counter() - start,
            "black_scholes": black_price,
            "black_scholes_reason": black_reason,
            "profile": profile,
            "mass_error": max(abs(row.survival_probability + row.cumulative_exercise_probability - 1)
                              for row in rows),
            "value_error": max(abs(row.total_value - price) for row in rows),
            "early_exercise_probability": sum(row.first_exercise_probability for row in rows[:-1]),
            "maturity_exercise_probability": terminal.first_exercise_probability,
            "expiration_probability": terminal.survival_probability,
        }
    finally:
        # Les liens parent/enfant/voisin forment des cycles. Effacer la référence
        # puis collecter permet de libérer la mémoire après chaque interaction.
        del tree
        gc.collect()


def _study_values(name: str, values: list, *, maximum: int = MAX_STUDY_POINTS) -> list:
    """Valider toute la série avant de commencer la première construction."""
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= maximum:
        raise ValueError(f"{name} doit contenir entre 1 et {maximum} valeurs.")
    return list(values)


def _step_values(steps: list[int]) -> list[int]:
    return [_validate_steps(value) for value in _study_values("La liste des nombres de pas", steps)]


def convergence_study(params: Parameters, steps: list[int]) -> pd.DataFrame:
    """Isoler l'erreur de discrétisation : contrat EUROPÉEN, pruning nul.

    Le taux, q, le strike et le type call/put restent ceux du formulaire.
    Une erreur de maillage est conservée dans la ligne correspondante au lieu
    de faire disparaître tous les autres résultats de l'étude.
    """
    params.validate()
    numbers = _step_values(steps)
    reference = float(_black_scholes(params).black_price(params.strike))
    european = replace(params, exercise_type="European", pruning_threshold=0.0)
    results = []
    for n in numbers:
        tree = None
        gc.collect()
        start = perf_counter()
        row = {"nb_steps": n, "price": float("nan"), "reference": reference,
               "error": float("nan"), "scaled_error": float("nan"),
               "seconds": 0.0, "node_count": 0, "status": "ok", "error_message": ""}
        try:
            tree = make_tree(european, nb_steps=n)
            value = _price(tree)
            row.update(price=value, error=value-reference, scaled_error=(value-reference)*n,
                       node_count=tree.count_nodes())
        except (ValueError, ArithmeticError) as exc:
            row.update(status="error", error_message=str(exc))
        finally:
            row["seconds"] = perf_counter() - start
            del tree
            gc.collect()
        results.append(row)
    return pd.DataFrame(results)


def strike_study(params: Parameters, strikes: list[float], steps: list[int]) -> pd.DataFrame:
    """Tracer l'erreur selon K ; ne construire qu'un seul arbre pour chaque N.

    Pour changer K, on revalorise les mêmes nœuds : le sous-jacent et leurs
    probabilités n'ont pas changé. Le payoff et les valeurs, eux, sont refaits.
    Comme la convergence en N, cette comparaison utilise le contrat européen
    sans pruning et refuse un cash actif.
    """
    params.validate()
    numbers = _step_values(steps)
    strike_values = [_finite_number("Le strike de l'étude", k)
                     for k in _study_values("La liste des strikes", strikes, maximum=MAX_STRIKES)]
    if any(k < 0 for k in strike_values):
        raise ValueError("Les strikes doivent être positifs ou nuls.")
    black = _black_scholes(params)
    european = replace(params, exercise_type="European", pruning_threshold=0.0)
    results = []
    for n in numbers:
        tree = None
        gc.collect()
        try:
            tree = make_tree(european, nb_steps=n)
            for strike in strike_values:
                tree.option.strike = strike
                value = _price(tree)
                reference = float(black.black_price(strike))
                results.append({"nb_steps": n, "strike": strike, "price": value,
                                "reference": reference, "error": value-reference})
        finally:
            del tree
            gc.collect()
    return pd.DataFrame(results)


def _pruned_count(tree: Tree) -> int:
    """Compter les transitions coupées par liens directs, sans liste de nœuds."""
    return sum(int(node.is_pruned)
               for trunk in tree.iter_columns() for node in tree.iter_column(trunk))


def pruning_study(params: Parameters, thresholds: list[float], nb_steps: int) -> pd.DataFrame:
    """Comparer les seuils au même arbre sans pruning, au N demandé.

    La référence seuil 0 est toujours calculée, même si elle n'est pas dans la
    liste saisie. Si cette construction échoue (possible avec un cash), on ne
    remplace pas secrètement la référence par un autre modèle : les écarts
    restent NaN et le message d'erreur de référence apparaît dans la table.
    """
    params.validate()
    n = _validate_steps(nb_steps)
    values = [_validate_threshold(value)
              for value in _study_values("La liste des seuils", thresholds)]
    # Dédoublonner les seuils économise des constructions identiques. La table
    # conserve l'ordre saisi et commence par la référence explicite seuil 0.
    ordered = list(dict.fromkeys([0.0, *values]))
    results = []
    reference = None
    for threshold in ordered:
        tree = None
        gc.collect()
        start = perf_counter()
        row = {"threshold": threshold, "price": float("nan"), "price_gap": float("nan"),
               "node_count": 0, "seconds": 0.0, "pruned_count": 0,
               "status": "ok", "error_message": ""}
        try:
            tree = make_tree(params, nb_steps=n, pruning_threshold=threshold)
            value = _price(tree)
            if threshold == 0:
                reference = value
            row.update(price=value, node_count=tree.count_nodes(), pruned_count=_pruned_count(tree))
            if reference is not None:
                row["price_gap"] = value-reference
        except (ValueError, ArithmeticError) as exc:
            row.update(status="error", error_message=str(exc))
        finally:
            row["seconds"] = perf_counter() - start
            del tree
            gc.collect()
        results.append(row)
    return pd.DataFrame(results)


def dividend_study(params: Parameters, amounts: list[float]) -> pd.DataFrame:
    """Changer seulement le cash ; garder le taux et q de l'écran.

    Pour l'expérience du cours à taux nul, l'appelant fournit explicitement
    ``replace(params, rate=0.0)``. Une date de détachement est nécessaire pour
    comparer des montants positifs, même si le cash de départ était désactivé.
    """
    params.validate()
    values = [_finite_number("Le dividende cash", value)
              for value in _study_values("La liste des dividendes", amounts)]
    if any(value < 0 for value in values):
        raise ValueError("Les dividendes cash doivent être positifs ou nuls.")
    if any(value > 0 for value in values) and params.ex_dividend_date is None:
        raise ValueError("Choisir une date de détachement pour étudier des dividendes cash positifs.")
    return _four_contract_study(params, "dividend_amount", values)


def rate_study(params: Parameters, rates: list[float]) -> pd.DataFrame:
    """Comparer taux positifs/négatifs et prime américain moins européen."""
    params.validate()
    values = [_finite_number("Le taux", value)
              for value in _study_values("La liste des taux", rates)]
    return _four_contract_study(params, "rate", values)


def _four_contract_study(params: Parameters, field: str, values: list[float]) -> pd.DataFrame:
    """Boucle commune : un arbre par marché, puis quatre rétropropagations.

    Les autres scénarios continuent si un maillage cash n'est pas admissible.
    Ni prix ni probabilités ne sont forcés pour masquer le problème du modèle.
    """
    results = []
    keys = ("call_european", "call_american", "put_european", "put_american")
    for value in values:
        tree = None
        gc.collect()
        row = {field: value, **{key: float("nan") for key in keys},
               "status": "ok", "error_message": ""}
        if field == "rate":
            row.update(call_premium=float("nan"), put_premium=float("nan"))
        try:
            scenario = replace(params, **{field: value})
            tree = make_tree(scenario)
            prices = _four_prices(tree)
            row.update(prices)
            if field == "rate":
                row.update(call_premium=prices["call_american"]-prices["call_european"],
                           put_premium=prices["put_american"]-prices["put_european"])
        except (ValueError, ArithmeticError) as exc:
            row.update(status="error", error_message=str(exc))
        finally:
            del tree
            gc.collect()
        results.append(row)
    return pd.DataFrame(results)


def debug_tree(params: Parameters, steps: int, color_by: str, show_neighbors: bool) -> dict:
    """Construire un petit arbre indépendant, vérifier puis rendre son SVG.

    Le petit N est un paramètre de visualisation : il ne remplace pas le N du
    prix principal. En particulier, le pas de détachement cash dépend de N.
    Le graphe dessiné et son prix peuvent donc différer du grand arbre de la
    synthèse. Cette différence est explicite dans l'interface.

    Le débogueur historique imprime ses contrôles et écrit un fichier. On
    capture ces messages pour l'écran et on lit le SVG dans un répertoire
    temporaire supprimé automatiquement ; aucun fichier de session partagé
    n'est écrasé si plusieurs utilisateurs ouvrent l'application.
    """
    params.validate()
    n = _validate_steps(steps, maximum=MAX_DEBUG_STEPS)
    DebugTree.validate_color(color_by)
    if not isinstance(show_neighbors, bool):
        raise ValueError("show_neighbors doit être un booléen.")
    tree = None
    debugger = None
    gc.collect()
    try:
        tree = make_tree(params, nb_steps=n)
        _price(tree)
        debugger = DebugTree(tree)
        log = StringIO()
        with redirect_stdout(log), TemporaryDirectory(prefix="arbre_streamlit_") as folder:
            errors = debugger.check()
            if errors:
                # Un graphe invalide ne doit pas être présenté comme un dessin
                # vérifié. Le rapport reste consultable même sans image.
                svg = ""
            else:
                path = debugger.export_svg(
                    Path(folder) / "arbre.svg", color_by, show_neighbors, run_checks=False,
                )
                svg = path.read_text(encoding="utf-8")
        return {
            "svg": svg,
            "report": log.getvalue(),
            "errors": int(errors),
            "node_count": int(debugger.node_count),
            "pruned_count": int(debugger.pruned_node_count),
            "weighted_mean_bias": float(debugger.pruning_weighted_mean_bias),
            "terminal_mean_bias": float(debugger.discounted_terminal_mean_bias),
        }
    finally:
        del debugger
        del tree
        gc.collect()


def check_tree(params: Parameters) -> dict:
    """Contrôler l'arbre au N principal sans produire un SVG gigantesque.

    Le diagnostic numérique peut porter sur 1 000 pas alors que le dessin
    lisible doit rester petit. Il utilise ici exactement N et le pruning de
    la valorisation principale, sans modifier les paramètres du formulaire.
    Les messages ne sont pas envoyés à stdout : seul l'écran qui a demandé
    ce contrôle les affiche.
    """
    params.validate()
    tree = None
    debugger = None
    gc.collect()
    try:
        tree = make_tree(params)
        _price(tree)
        debugger = DebugTree(tree)
        log = StringIO()
        with redirect_stdout(log):
            errors = debugger.check()
        return {
            "report": log.getvalue(),
            "errors": int(errors),
            "node_count": int(debugger.node_count),
            "pruned_count": int(debugger.pruned_node_count),
            "weighted_mean_bias": float(debugger.pruning_weighted_mean_bias),
            "terminal_mean_bias": float(debugger.discounted_terminal_mean_bias),
        }
    finally:
        del debugger
        del tree
        gc.collect()
