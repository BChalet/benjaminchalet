"""Reproduire les courbes et comparer un arbre complet à un arbre élagué.

Exécution : python exemple_complet.py --output-dir resultats
Les graphiques ne nécessitent pas de fenêtre ; ajouter --show pour les ouvrir.
Les paramètres ci-dessous sont des exemples, pas des données de calibration.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
from pathlib import Path
from time import perf_counter

from arbre_binomial import Market, Model, Option, Tree
from debug_arbre import DebugTree
from profils_option import OptionProfiles


PRICING_DATE = dt.date(2026, 9, 15)
MATURITY = dt.date(2027, 9, 2)


def market_example() -> Market:
    """Taux positif + put américain : l'exercice anticipé peut être utile."""
    return Market(100.0, 0.07, 0.30, dividend_amount=2.0,
                  ex_dividend_date=dt.date(2027, 3, 15))


def make_tree(steps: int, threshold: float, exercise: str = "American") -> Tree:
    return Tree(Model(PRICING_DATE, steps, threshold),
                Option(101.0, MATURITY, "Put", exercise), market_example()).build()


def compare_pruning(steps: int, destination: Path) -> None:
    """Mesurer séparément l'erreur d'élagage et le coût build+price.

    Le premier arbre (seuil nul) fournit la référence au MÊME nombre de pas.
    Cela isole l'effet du pruning de l'erreur de discrétisation temporelle.
    Les arbres précédents sont libérés entre deux mesures : pas besoin de
    conserver simultanément plusieurs graphes de N² objets en mémoire.
    Le temps est une observation de cette machine, pas une garantie.

    Cette comparaison est faite SANS CASH pour isoler le pruning binomial.
    Avec sigma=30 %, le cas cash des profils à 200 pas peut produire des
    probabilités trinômiales négatives à 1000 pas, dans les états extrêmes
    du graphe complet. Le cours avertit que ces probabilités ne sont pas
    toujours positives : on conserve l'erreur du moteur, sans les forcer.
    """
    rows = []  # Seulement des métriques scalaires, jamais des nœuds.
    reference = None
    for threshold in (0.0, 1e-7, 1e-4):
        gc.collect()
        start = perf_counter()
        tree = Tree(Model(PRICING_DATE, steps, threshold),
                    Option(101.0, MATURITY, "Put", "American"),
                    Market(100.0, 0.07, 0.30)).build()
        price = tree.price()
        seconds = perf_counter() - start
        if reference is None:
            reference = price
        pruned_count = sum(node.is_pruned for trunk in tree.iter_columns()
                           for node in tree.iter_column(trunk))
        rows.append((steps, threshold, price, price-reference,
                     tree.count_nodes(), pruned_count, seconds))
        del tree
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("N", "seuil", "prix", "ecart_au_complet", "noeuds",
                         "transitions_elaguees", "secondes_build_price", "dividende_cash"))
        writer.writerows((*row, 0.0) for row in rows)
    for n, threshold, price, gap, count, cuts, seconds in rows:
        print(f"N={n}, seuil={threshold:g} : prix={price:.10f}, écart={gap:+.3g}, "
              f"{count} nœuds, {cuts} coupes, {seconds:.3f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200, help="Pas pour les profils (défaut 200).")
    parser.add_argument("--benchmark-steps", type=int, default=1000,
                        help="Pas pour comparer les seuils (défaut 1000).")
    parser.add_argument("--output-dir", type=Path, default=Path("resultats"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    # Valider les nombres avant de commencer les calculs/exportations.
    Model(PRICING_DATE, args.steps)
    Model(PRICING_DATE, args.benchmark_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Même sous-jacent : seuls les choix d'exercice changent. Les décisions
    # sont intégralement recalculées par exercise_profile() après mutation.
    tree = make_tree(args.steps, 1e-7)
    for exercise in ("European", "American"):
        tree.option.exercise_type = exercise
        study = OptionProfiles(tree)
        study.plot(args.output_dir / exercise.lower(), show=args.show)
        rows = study.compute()
        print(f"{exercise} : prix={tree.root.option_value:.10f}, "
              f"P(exercice)={rows[-1].cumulative_exercise_probability:.4%}, "
              f"P(expiration sans paiement)={rows[-1].survival_probability:.4%}")
    del tree

    # Avec 16 pas et 1e-3, les coupes sont visibles. Le seuil volontairement
    # élevé sert à comprendre le dessin ; il n'est pas une recommandation
    # de précision. Le cas à 1000 pas ci-dessous mesure des seuils plus fins.
    for label, threshold in (("complet", 0.0), ("elague", 1e-3)):
        small_tree = make_tree(16, threshold)
        DebugTree(small_tree).export_svg(args.output_dir / f"arbre_{label}.svg",
                                         show_neighbors=True)
        del small_tree
    print("Comparaison du pruning binomial sans dividende cash :")
    compare_pruning(args.benchmark_steps, args.output_dir / "comparaison_pruning.csv")


if __name__ == "__main__":
    main()
