"""Courbes de valeur et d'exercice demandées aux pages 22–23 du cours.

Le mot « valeur » désigne ici une MOYENNE PAR DATE dans l'arbre, et non
une courbe du prix en fonction du spot. Toutes les valeurs sont actualisées
à la date initiale ; toutes les probabilités sont risque-neutres (mesure Q).

Trois probabilités répondent à trois questions différentes :

* ``exercise_region_probability`` : à une date donnée, quelle est la masse
  des nœuds où il faudrait exercer ? C'est la courbe décrite dans le PDF.
  On suit ici l'arbre entier, même après un exercice théorique antérieur.
* ``first_exercise_probability`` : quelle masse exerce POUR LA PREMIÈRE FOIS
  à cette date ? On retire les contrats dès leur premier exercice.
* ``cumulative_exercise_probability`` : quelle masse a déjà exercé depuis
  la date initiale ? C'est le cumul de la distribution précédente.

Avant T, ``survival_probability`` est la masse encore vivante après décision.
À T, par convention du moteur, ce champ conserve la masse qui expire sans
paiement. Elle est distinguée graphiquement des contrats vivants avant T.

Il ne faut donc PAS additionner les valeurs de la première courbe : un
même chemin peut passer plusieurs fois par une zone d'exercice. Les deux
autres courbes évitent ce double comptage. À maturité, seul un payoff
strictement positif compte comme exercice ; sinon le contrat expire.

Exemple Python, à partir d'un arbre déjà créé :

    from profils_option import OptionProfiles

    etude = OptionProfiles(tree)
    lignes = etude.compute()       # construit si nécessaire, puis reprice
    print(lignes[0].total_value)   # prix initial
    fichiers = etude.plot("resultats/profils")

Exemple en ligne de commande :

    python profils_option.py --steps 200 --output-dir resultats/profils

Le moteur ``arbre_binomial.py`` fait tous les calculs financiers. Ce module
ne garde que des lignes de nombres, jamais un tableau de nœuds de l'arbre.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from math import isfinite
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from arbre_binomial import Market, Model, Option, Tree


class ProfileRow(Protocol):
    """Champs lus dans une ligne scalaire renvoyée par le moteur.

    Un ``Protocol`` sert uniquement à décrire les attributs utilisés ici.
    Il n'impose pas une seconde classe de résultats au moteur : la même
    dataclass renvoyée par ``Tree.exercise_profile()`` est réutilisée.
    """

    step: int
    time_years: float
    date: dt.date | dt.datetime
    discounted_mean_value: float
    exercise_region_probability: float
    first_exercise_probability: float
    cumulative_exercise_probability: float
    survival_probability: float
    discounted_alive_value: float
    discounted_exercise_cashflow: float
    cumulative_discounted_cashflow: float
    total_value: float


# Cet ordre est aussi celui des colonnes du CSV. Les noms sont identiques
# aux attributs du moteur pour faciliter le passage du graphique au code.
CSV_FIELDS = (
    "step", "time_years", "date", "discounted_mean_value",
    "exercise_region_probability", "first_exercise_probability",
    "cumulative_exercise_probability", "survival_probability",
    "discounted_alive_value", "discounted_exercise_cashflow",
    "cumulative_discounted_cashflow", "total_value",
)


class OptionProfiles:
    """Calcule, trace et exporte les profils d'un ``Tree``.

    ``compute()`` renvoie une ligne par date, racine et maturité incluses.
    ``plot()`` produit deux figures, chacune en PNG et en SVG, et un CSV.
    Le SVG conserve des courbes et du texte vectoriels : on peut zoomer
    ou l'insérer dans un rapport sans pixellisation.

    ``results`` contient seulement les dernières lignes scalaires. Après
    modification du strike ou du type d'exercice, relancer ``compute()``
    ou ``plot()`` pour recalculer les valeurs. Pour changer le marché ou
    le maillage, créer un NOUVEL arbre, dont la géométrie sera différente.
    """

    def __init__(self, tree: Tree) -> None:
        if not isinstance(tree, Tree):
            raise TypeError("OptionProfiles attend un objet Tree.")
        self.tree = tree
        self.results: tuple[ProfileRow, ...] | None = None

    def compute(self) -> tuple[ProfileRow, ...]:
        """Construit si nécessaire et demande un profil repricé au moteur.

        L'actualisation, le calcul de la zone d'exercice et la propagation
        des probabilités vivantes appartiennent au moteur. Les séparer du
        dessin évite de dupliquer une formule financière dans deux fichiers.

        ``Tree.build()`` est idempotente : l'appeler sur un arbre construit
        ne crée pas de nouveaux nœuds. ``exercise_profile()`` refait le
        pricing, ce qui évite de lire des valeurs d'option périmées.
        """
        self.tree.build()
        self.results = tuple(self.tree.exercise_profile())
        if not self.results:
            raise RuntimeError("Le moteur a renvoyé un profil vide.")
        return self.results

    def _rows(self) -> tuple[ProfileRow, ...]:
        """Fournit les résultats ; lance le calcul au premier besoin."""
        if self.results is None:
            return self.compute()
        return self.results

    def _description(self) -> str:
        """Paramètres courts pour identifier une figure dans un rapport."""
        tree = self.tree
        exercise = (
            "américain" if tree.option.exercise_type.lower() == "american"
            else "européen"
        )
        return (
            f"{tree.option.type.capitalize()} {exercise} — "
            f"S₀={tree.market.spot:g}, K={tree.option.strike:g}, "
            f"r={tree.market.rate:.1%}, σ={tree.market.vol:.1%}, "
            f"N={tree.model.nb_steps}, pruning={tree.model.pruning_threshold:g}"
        )

    def _decorate_time_axis(self, ax: Axes) -> None:
        """Marque le temps exact du dividende, distinct de sa colonne."""
        tree = self.tree
        ax.set_xlabel(
            f"Temps depuis le {tree.model.pricing_date:%d/%m/%Y} "
            "(années, ACT/365)"
        )
        ax.grid(alpha=0.25)
        ex_date = tree.market.ex_dividend_date
        if (
            tree.market.dividend_amount > 0
            and ex_date is not None
            and tree.model.pricing_date < ex_date <= tree.option.maturity
        ):
            # Le maillage peut ne pas contenir exactement la date cash.
            # Cette ligne indique la vraie date ; le moteur affecte le
            # saut au pas qui contient cette date, conformément au cours.
            ex_time = (ex_date - tree.model.pricing_date).days / 365.0
            ax.axvline(ex_time, color="0.45", linestyle=":", linewidth=1.2,
                       label=f"Dividende {tree.market.dividend_amount:g} "
                             f"le {ex_date:%d/%m/%Y}")

    @staticmethod
    def _save_figure(fig: Figure, output_dir: Path, stem: str) -> dict[str, Path]:
        """Sauvegarde la même figure en bitmap et en format vectoriel."""
        files = {
            "png": output_dir / f"{stem}.png",
            "svg": output_dir / f"{stem}.svg",
        }
        fig.savefig(files["png"], dpi=180, bbox_inches="tight")
        # svg.fonttype='none' garde le texte sélectionnable dans le SVG.
        with plt.rc_context({"svg.fonttype": "none"}):
            fig.savefig(files["svg"], bbox_inches="tight")
        return files

    def _value_figure(self) -> Figure:
        """Distingue moyenne du PDF et valeur des contrats encore vivants."""
        rows = self._rows()
        times = [row.time_years for row in rows]
        initial_price = rows[0].total_value
        fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
        fig.suptitle(self._description(), fontsize=12)

        # MOYENNE DEMANDÉE DANS LE PDF : Q_j n'est pas absorbée après
        # exercice. On moyenne la valeur d'option sur TOUS les nœuds de
        # chaque colonne et on actualise à t=0 : exp(-r*t) Σ_j Q_j V_j.
        # L'option européenne donne une constante par la propriété de la
        # tour des espérances. L'américaine donne une supermartingale :
        # cette moyenne peut décroître lorsque l'exercice devient optimal.
        axes[0].plot(times, [row.discounted_mean_value for row in rows],
                     color="tab:blue", linewidth=2,
                     label="Moyenne brute actualisée (PDF, arbre entier)")
        axes[0].axhline(initial_price, color="black", linestyle="--",
                       linewidth=1, label=f"Prix initial : {initial_price:.6f}")
        axes[0].set_title("Moyenne de la valeur d'option par date")
        axes[0].set_ylabel("Valeur actualisée à la date initiale")
        self._decorate_time_axis(axes[0])
        axes[0].legend(fontsize=9)

        # CONTRÔLE DE CONSERVATION : à une date t, le prix initial est
        # la valeur avant décision des contrats encore vivants + les
        # cashflows d'exercice STRICTEMENT antérieurs à t.
        # Le flux de t est déjà dans la valeur vivante AVANT décision.
        # L'ajouter à nouveau compterait donc ce flux deux fois.
        previous_cashflows = [
            row.cumulative_discounted_cashflow - row.discounted_exercise_cashflow
            for row in rows
        ]
        axes[1].plot(times, [row.discounted_alive_value for row in rows],
                     color="tab:blue", label="Contrats vivants, avant décision")
        axes[1].plot(times, previous_cashflows, color="tab:orange",
                     label="Cashflows reçus strictement avant cette date")
        axes[1].plot(times, [row.total_value for row in rows],
                     color="tab:green", linewidth=2, linestyle="--",
                     label="Somme : prix initial conservé")
        axes[1].set_title("Contrôle : valeur vivante + cashflows passés")
        axes[1].set_ylabel("Valeur actualisée à la date initiale")
        self._decorate_time_axis(axes[1])
        axes[1].legend(fontsize=9)
        return fig

    def _exercise_figure(self) -> Figure:
        """Dessine la zone brute, la première distribution, puis son cumul."""
        rows = self._rows()
        times = [row.time_years for row in rows]
        fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
        fig.suptitle(self._description() + "\nProbabilités risque-neutres Q",
                     fontsize=12)

        # La dernière ligne est spéciale : à T il s'agit du payoff final,
        # pas d'un exercice ANTICIPÉ. On la dessine donc séparément.
        early_times = times[:-1]
        maturity = rows[-1]
        axes[0].plot(early_times,
                     [row.exercise_region_probability for row in rows[:-1]],
                     color="tab:blue", label="Zone d'exercice anticipé, arbre entier")
        axes[0].scatter([times[-1]], [maturity.exercise_region_probability],
                        marker="D", color="tab:purple", zorder=3,
                        label="À maturité : payoff > 0, arbre entier")
        axes[0].set_title("Courbe du PDF : masse des nœuds en zone d'exercice")
        axes[0].text(0.02, 0.96,
                     "Un même chemin peut contribuer à plusieurs dates : "
                     "ne pas sommer cette courbe.",
                     transform=axes[0].transAxes, va="top", fontsize=9,
                     bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})

        # Ici chaque contrat est absorbé à son premier exercice. Les
        # hauteurs sont des PROBABILITÉS PAR PAS, pas une densité par an.
        # En raffinant N, une hauteur peut baisser simplement parce que
        # la même masse est répartie sur davantage de dates.
        step_width = self.tree.deltaT * 0.75
        axes[1].bar(early_times,
                    [row.first_exercise_probability for row in rows[:-1]],
                    width=step_width, color="tab:orange",
                    label="Premier exercice anticipé à cette date")
        axes[1].vlines(times[-1], 0, maturity.first_exercise_probability,
                       color="tab:purple", linewidth=2)
        axes[1].scatter([times[-1]], [maturity.first_exercise_probability],
                        marker="D", color="tab:purple", zorder=3,
                        label="Premier exercice à maturité, payoff > 0")
        axes[1].set_title("Distribution du premier exercice : aucun double comptage")

        axes[2].step(times, [row.cumulative_exercise_probability for row in rows],
                     where="post", color="tab:green",
                     label="Probabilité cumulée d'avoir exercé")
        axes[2].step(early_times, [row.survival_probability for row in rows[:-1]],
                     where="post", color="tab:red", linestyle="--",
                     label="Contrat vivant après décision, avant T")
        # À T, le moteur réutilise le champ survival_probability pour la
        # masse qui expire sans paiement. Le contrat n'est alors plus
        # vivant : on distingue cette masse par une croix à la date T.
        # Le cumul final d'exercice n'a donc pas besoin de valoir 100 %.
        expired_probability = maturity.survival_probability
        axes[2].scatter([times[-1]], [expired_probability], color="tab:red",
                        marker="x", s=55, zorder=3,
                        label="À T : expiration sans paiement")
        axes[2].set_title(
            "Cumul et survie — "
            f"expiration sans exercice à T : {expired_probability:.2%}"
        )
        for index, ax in enumerate(axes):
            ax.set_ylabel("Probabilité risque-neutre")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            if index == 1:
                # Une probabilité PAR PAS est souvent de quelques %.
                # Une échelle systématique de 0 à 100 % écraserait toute
                # la distribution près de zéro. L'axe reste en % ; seule
                # sa plage s'adapte à la plus grande probabilité du profil.
                maximum = max(row.first_exercise_probability for row in rows)
                ax.set_ylim(0.0, min(1.08, max(0.01, 1.2 * maximum)))
            else:
                ax.set_ylim(-0.03, 1.08)
            self._decorate_time_axis(ax)
            ax.legend(fontsize=9, loc="upper left" if ax is axes[2] else "best")
        return fig

    def export_csv(self, output_dir: str | Path) -> Path:
        """Exporte les nombres en précision complète, avec une date ISO.

        Les probabilités sont entre 0 et 1 (0.25 signifie 25 %), sans
        actualisation. Les champs ``discounted_*`` sont déjà actualisés.
        À T, ``survival_probability`` est la masse qui expire sans paiement,
        et non la probabilité d'un contrat vivant après sa maturité.
        Le séparateur est la virgule ; utiliser l'import CSV dans Excel
        si sa configuration française attend un point-virgule.
        """
        rows = self._rows()
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        csv_path = destination / "profils_option.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                values = {field: getattr(row, field) for field in CSV_FIELDS}
                values["date"] = row.date.isoformat()
                writer.writerow(values)
        return csv_path

    def plot_value_curve(self, output_dir: str | Path, show: bool = False) -> dict[str, Path]:
        """Exporte uniquement la figure de valeur en PNG et en SVG.

        Réutilise le dernier ``compute()`` ; calcule si nécessaire. Pour
        repricer explicitement un arbre déjà étudié, utiliser ``plot()``.
        """
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        fig = self._value_figure()
        files = self._save_figure(fig, destination, "valeur_option_par_date")
        if show:
            plt.show()
        plt.close(fig)
        return files

    def plot_exercise_probability(self, output_dir: str | Path,
                                  show: bool = False) -> dict[str, Path]:
        """Exporte uniquement les trois profils de probabilité en PNG/SVG."""
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        fig = self._exercise_figure()
        files = self._save_figure(fig, destination, "probabilite_exercice_par_date")
        if show:
            plt.show()
        plt.close(fig)
        return files

    def plot(self, output_dir: str | Path, show: bool = False) -> dict[str, Path]:
        """Recalcule puis écrit les cinq fichiers et renvoie leurs chemins.

        ``show=False`` convient aux scripts et aux machines sans écran.
        ``show=True`` affiche les deux figures après leur sauvegarde.
        """
        self.compute()
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        # Les deux figures sont créées avant plt.show() pour les afficher
        # en même temps. finally ferme uniquement nos propres figures.
        value_figure = self._value_figure()
        exercise_figure = self._exercise_figure()
        try:
            value_files = self._save_figure(
                value_figure, destination, "valeur_option_par_date"
            )
            exercise_files = self._save_figure(
                exercise_figure, destination, "probabilite_exercice_par_date"
            )
            csv_path = self.export_csv(destination)
            if show:
                plt.show()
        finally:
            plt.close(value_figure)
            plt.close(exercise_figure)
        return {
            "value_png": value_files["png"],
            "value_svg": value_files["svg"],
            "exercise_png": exercise_files["png"],
            "exercise_svg": exercise_files["svg"],
            "csv": csv_path,
        }


def _positive_int(text: str) -> int:
    """Validation argparse, avec une erreur lisible plutôt qu'une traceback."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("entier strictement positif attendu") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("le nombre de pas doit être >= 1")
    return value


def _nonnegative_float(text: str) -> float:
    """Refuse aussi NaN et l'infini, qui ne sont pas des seuils utiles."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("nombre positif ou nul attendu") from exc
    if not isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("le seuil doit être fini et compris entre 0 et 1")
    return value


def main() -> None:
    """Démonstration reproductible : put américain avec dividende cash."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=_positive_int, default=200,
                        help="nombre de pas de l'arbre (défaut : 200)")
    parser.add_argument("--pruning-threshold", type=_nonnegative_float, default=1e-7,
                        help="seuil de pruning ; 0 le désactive (défaut : 1e-7)")
    parser.add_argument("--exercise-type", choices=("american", "european"),
                        default="american", help="style d'exercice (défaut : american)")
    parser.add_argument("--output-dir", type=Path, default=Path("resultats/profils"),
                        help="dossier de sortie (défaut : resultats/profils)")
    parser.add_argument("--show", action="store_true", help="afficher les figures")
    args = parser.parse_args()

    tree = Tree(
        model=Model(pricing_date=dt.date(2026, 9, 15), nb_steps=args.steps,
                    pruning_threshold=args.pruning_threshold),
        option=Option(strike=101.0, maturity=dt.date(2027, 9, 2),
                      type="Put", exercise_type=args.exercise_type),
        market=Market(spot=100.0, rate=0.07, vol=0.30, dividend_yield=0.0,
                      dividend_amount=2.0, ex_dividend_date=dt.date(2027, 3, 15)),
    )
    study = OptionProfiles(tree)
    try:
        files = study.plot(args.output_dir, show=args.show)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(status=2, message=f"Erreur lors du calcul ou de l'export : {exc}\n")

    rows = study._rows()
    print(f"Prix initial : {rows[0].total_value:.10f}")
    print("Probabilités sous la mesure risque-neutre Q :")
    print(f"  Premier exercice avant T : "
          f"{sum(row.first_exercise_probability for row in rows[:-1]):.4%}")
    print(f"  Premier exercice à T : {rows[-1].first_exercise_probability:.4%}")
    print(f"  Exercice cumulé : {rows[-1].cumulative_exercise_probability:.4%}")
    for name, path in files.items():
        print(f"  {name} : {path.resolve()}")


if __name__ == "__main__":
    main()
