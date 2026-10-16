"""Tests financiers des profils de valeur et de premier exercice.

Lancement depuis le dossier du projet : ``python -m unittest -v test_profils``.

Ces tests distinguent deux expériences :

* on laisse le sous-jacent parcourir tout l'arbre, même après une occasion
  d'exercice : on obtient la moyenne brute du cours et la masse de la région
  d'exercice ;
* on arrête chaque contrat à son PREMIER exercice : on obtient les flux versés
  au porteur et la probabilité que son contrat soit encore vivant.

Le contrôle le plus important énumère les chemins d'un petit arbre. Il ne se
sert ni de ``node.proba``, ni de ``node.alive_proba``, ni des décisions calculées
par le moteur. Deux chemins qui se recombinent dans un même nœud gardent donc
des historiques d'exercice distincts. Ce test détecte notamment le double
comptage qui résulterait d'une simple somme des régions d'exercice.

Aucune collection de nœuds n'est créée : les parcours suivent les liens. Les
petites listes de l'oracle contiennent uniquement des nombres, un par date.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, fields, is_dataclass
import gc
from math import exp
import unittest

from arbre_binomial import Market, Model, Node, Option, Tree


PRICING_DATE = dt.date(2026, 9, 15)
MATURITY = PRICING_DATE + dt.timedelta(days=365)


def make_tree(
    n: int = 60,
    *,
    strike: float = 105.0,
    kind: str = "Put",
    exercise: str = "American",
    rate: float = 0.07,
    vol: float = 0.25,
    q: float = 0.0,
    cash: float = 0.0,
    pruning: float = 0.0,
) -> Tree:
    """Construire un cas reproductible ; le cash éventuel est versé à J+180."""
    return Tree(
        Model(PRICING_DATE, n, pruning_threshold=pruning),
        Option(strike, MATURITY, kind, exercise),
        Market(
            spot=100.0,
            rate=rate,
            vol=vol,
            dividend_yield=q,
            dividend_amount=cash,
            ex_dividend_date=PRICING_DATE + dt.timedelta(days=180) if cash else None,
        ),
    ).build()


@dataclass(frozen=True)
class PathTotals:
    """Résultats numériques de l'énumération indépendante des chemins."""

    mean_value: tuple[float, ...]
    region: tuple[float, ...]
    first: tuple[float, ...]
    alive_value: tuple[float, ...]
    cashflow: tuple[float, ...]
    expired_without_payoff: float


def enumerate_paths(tree: Tree) -> PathTotals:
    """Oracle exponentiel, réservé aux arbres de quelques pas.

    Chaque appel récursif représente UN chemin et conserve un simple booléen
    indiquant si le contrat a déjà été exercé. Il continue à parcourir le
    sous-jacent après cet exercice pour mesurer aussi les statistiques brutes.
    Cette méthode coûte cher, mais n'utilise pas l'algorithme de propagation de
    masses entre nœuds recombinés que nous voulons contrôler.
    """
    n = tree.model.nb_steps
    mean_value = [0.0] * (n + 1)
    region = [0.0] * (n + 1)
    first = [0.0] * (n + 1)
    alive_value = [0.0] * (n + 1)
    cashflow = [0.0] * (n + 1)
    expired = 0.0
    delta = (tree.option.maturity - tree.model.pricing_date).days / (365.0 * n)
    discount = exp(-tree.market.rate * delta)

    def payoff(node: Node) -> float:
        # Écrire le payoff ici évite d'utiliser la même méthode que le moteur.
        difference = node.spot_price - tree.option.strike
        return max(difference if tree.option.type.lower() == "call" else -difference, 0.0)

    def value_and_decision(node: Node) -> tuple[float, bool]:
        """Recalculer Bellman sans lire option_value ou exercise_now."""
        intrinsic = payoff(node)
        if node.step == n:
            return intrinsic, intrinsic > 0.0
        continuation = 0.0
        for branch in ("up", "mid", "down"):
            child = getattr(node, "next_" + branch)
            if child is not None:
                continuation += getattr(node, "p_" + branch) * value_and_decision(child)[0]
        continuation *= discount
        if tree.option.exercise_type.lower() == "european":
            return continuation, False
        tolerance = 1e-12 * max(1.0, abs(intrinsic), abs(continuation))
        exercise = intrinsic > 0.0 and intrinsic > continuation + tolerance
        return (intrinsic if exercise else continuation), exercise

    def walk(node: Node, probability: float, already_exercised: bool) -> None:
        nonlocal expired
        value, exercise = value_and_decision(node)
        step = node.step
        date_discount = exp(-tree.market.rate * delta * step)
        mean_value[step] += probability * date_discount * value
        if exercise:
            region[step] += probability
        if not already_exercised:
            # La valeur du contrat vivant est mesurée AVANT la décision du jour.
            alive_value[step] += probability * date_discount * value
            if exercise:
                first[step] += probability
                cashflow[step] += probability * date_discount * payoff(node)
        stopped = already_exercised or exercise
        if step == n:
            if not stopped:
                expired += probability
            return
        for branch in ("up", "mid", "down"):
            child = getattr(node, "next_" + branch)
            if child is not None:
                walk(child, probability * getattr(node, "p_" + branch), stopped)

    walk(tree.root, 1.0, False)
    return PathTotals(
        tuple(mean_value), tuple(region), tuple(first), tuple(alive_value),
        tuple(cashflow), expired,
    )


class ProfileTests(unittest.TestCase):
    def tearDown(self) -> None:
        # Les liens des arbres forment des cycles ; ne pas les conserver entre
        # deux tests indépendants lors d'une exécution de toute la suite.
        gc.collect()

    def assert_value_close(self, actual: float, expected: float) -> None:
        self.assertAlmostEqual(actual, expected, delta=2e-10 * max(1.0, abs(expected)))

    def assert_mass_balance(self, tree: Tree, profile: tuple) -> None:
        """Vérifier la conservation après chaque décision et chaque transition."""
        cumulative = 0.0
        for row, column in zip(profile, tree.iter_columns()):
            cumulative += row.first_exercise_probability
            raw_mass = sum(node.proba for node in tree.iter_column(column))
            alive_before = sum(node.alive_proba for node in tree.iter_column(column))
            self.assertAlmostEqual(raw_mass, 1.0, delta=2e-12)
            self.assertAlmostEqual(row.cumulative_exercise_probability, cumulative, delta=2e-12)
            self.assertAlmostEqual(row.survival_probability + cumulative, 1.0, delta=2e-12)
            self.assertAlmostEqual(
                alive_before - row.first_exercise_probability,
                row.survival_probability,
                delta=2e-12,
            )
            self.assertGreaterEqual(row.first_exercise_probability, -1e-14)
            self.assertGreaterEqual(row.survival_probability, -1e-14)

    def test_europeenne_moyenne_actualisee_constante_et_format_scalaire(self) -> None:
        # L'identité vaut avec rendement continu, avec cash, et avec pruning :
        # elle repose sur l'actualisation et les probabilités réellement utilisées.
        for cash, pruning in ((0.0, 0.0), (2.0, 0.0), (0.0, 0.005)):
            with self.subTest(cash=cash, pruning=pruning):
                tree = make_tree(exercise="European", q=0.02, cash=cash, pruning=pruning)
                profile = tree.exercise_profile()
                self.assertIsInstance(profile, tuple)
                self.assertEqual(len(profile), tree.model.nb_steps + 1)
                # Une heure conserve la position exacte des dates de grille
                # lorsque le pas ne représente pas un nombre entier de jours.
                self.assertEqual(profile[0].date, dt.datetime.combine(PRICING_DATE, dt.time()))
                self.assertEqual(profile[-1].date, dt.datetime.combine(MATURITY, dt.time()))
                self.assertAlmostEqual(profile[0].time_years, 0.0)
                self.assertAlmostEqual(profile[-1].time_years, 1.0)
                for step, row in enumerate(profile):
                    self.assertTrue(is_dataclass(row))
                    # Les lignes sont exportables et ne cachent aucun nœud.
                    for field in fields(row):
                        scalar = getattr(row, field.name)
                        self.assertTrue(
                            scalar is None or isinstance(scalar, (int, float, str, bool, dt.date)),
                            msg=f"Le champ {field.name} n'est pas scalaire : {type(scalar)}",
                        )
                    self.assertEqual(row.step, step)
                    self.assertAlmostEqual(row.time_years, step / tree.model.nb_steps)
                    self.assert_value_close(row.discounted_mean_value, tree.root.option_value)
                    self.assert_value_close(row.total_value, tree.root.option_value)
                    if step < tree.model.nb_steps:
                        self.assertEqual(row.exercise_region_probability, 0.0)
                        self.assertEqual(row.first_exercise_probability, 0.0)
                self.assert_mass_balance(tree, profile)

    def test_americaine_moyenne_brute_decroissante_et_valeur_totale_constante(self) -> None:
        tree = make_tree(n=80)
        profile = tree.exercise_profile()
        price = tree.root.option_value
        self.assertGreater(sum(row.first_exercise_probability for row in profile[:-1]), 0.0)
        previous_mean = profile[0].discounted_mean_value
        cashflows_before = 0.0
        for row in profile:
            self.assertLessEqual(row.discounted_mean_value, previous_mean + 2e-10)
            self.assert_value_close(row.discounted_alive_value + cashflows_before, price)
            self.assert_value_close(row.total_value, price)
            cashflows_before += row.discounted_exercise_cashflow
            self.assert_value_close(row.cumulative_discounted_cashflow, cashflows_before)
            previous_mean = row.discounted_mean_value
        self.assert_value_close(cashflows_before, price)
        self.assert_mass_balance(tree, profile)

    def test_premier_exercice_compare_a_une_enumeration_des_chemins(self) -> None:
        # Le cas sans cash occupe plusieurs fois la région d'exercice : sommer
        # ces probabilités brutes produit même un résultat supérieur à 100 %.
        # L'autre cas traverse un vrai pas trinomial de détachement du cash.
        for cash in (0.0, 2.0):
            with self.subTest(cash=cash):
                tree = make_tree(n=6, strike=110, rate=0.10, vol=0.20, q=0.01 if cash else 0, cash=cash)
                reference = enumerate_paths(tree)
                profile = tree.exercise_profile()
                if cash == 0:
                    self.assertGreater(sum(reference.region), 1.0)
                self.assertLessEqual(sum(reference.first), 1.0 + 1e-12)
                for row in profile:
                    i = row.step
                    self.assert_value_close(row.discounted_mean_value, reference.mean_value[i])
                    self.assertAlmostEqual(row.exercise_region_probability, reference.region[i], delta=2e-12)
                    self.assertAlmostEqual(row.first_exercise_probability, reference.first[i], delta=2e-12)
                    self.assert_value_close(row.discounted_alive_value, reference.alive_value[i])
                    self.assert_value_close(row.discounted_exercise_cashflow, reference.cashflow[i])
                self.assertAlmostEqual(profile[-1].survival_probability, reference.expired_without_payoff, delta=2e-12)
                self.assert_value_close(sum(reference.cashflow), tree.root.option_value)
                self.assert_mass_balance(tree, profile)

    def test_pruning_conserve_masses_et_flux_du_contrat(self) -> None:
        tree = make_tree(n=100, pruning=0.005)
        profile = tree.exercise_profile()
        saw_pruning = False
        for column in tree.iter_columns():
            for node in tree.iter_column(column):
                if node.is_pruned:
                    saw_pruning = True
                    self.assertIsNone(node.next_up)
                    self.assertIsNone(node.next_down)
                    self.assertIsNotNone(node.next_mid)
                    self.assertEqual(node.p_mid, 1.0)
        self.assertTrue(saw_pruning, "Ce cas doit réellement exercer le pruning.")
        full_binomial_count = (tree.model.nb_steps + 1) * (tree.model.nb_steps + 2) // 2
        self.assertLess(tree.count_nodes(), full_binomial_count)
        self.assert_mass_balance(tree, profile)
        self.assert_value_close(sum(row.discounted_exercise_cashflow for row in profile), tree.root.option_value)
        for row in profile:
            self.assert_value_close(row.total_value, tree.root.option_value)

    def test_exercice_immediat_ne_compte_pas_deux_fois_le_flux_du_jour(self) -> None:
        # Sans volatilité, le put très ITM perd de sa valeur intrinsèque lorsque
        # S croît au taux positif : son exercice optimal est immédiat.
        tree = make_tree(n=8, strike=150, rate=0.10, vol=0.0)
        profile = tree.exercise_profile()
        self.assertEqual(profile[0].first_exercise_probability, 1.0)
        self.assertEqual(profile[0].discounted_exercise_cashflow, 50.0)
        self.assertEqual(profile[0].discounted_alive_value, 50.0)
        self.assertEqual(profile[0].total_value, 50.0)
        for row in profile:
            self.assertEqual(row.survival_probability, 0.0)
            self.assert_value_close(row.total_value, 50.0)
            if row.step > 0:
                self.assertEqual(row.first_exercise_probability, 0.0)
                self.assertEqual(row.discounted_alive_value, 0.0)
                self.assertEqual(row.discounted_exercise_cashflow, 0.0)
        self.assert_mass_balance(tree, profile)

    def test_egalite_continuer_et_payoff_nul_jamais_exerce(self) -> None:
        # À r = sigma = 0, attendre et exercer sont exactement équivalents.
        # Le contrat positif attend la maturité ; celui à payoff nul expire.
        # Le troisième strike contrôle que tout payoff terminal STRICTEMENT
        # positif est exercé, même s'il est inférieur à la tolérance préterminale.
        for strike in (90.0, 100.0, 100.0 - 1e-13):
            with self.subTest(strike=strike):
                tree = make_tree(n=5, kind="Call", strike=strike, rate=0.0, vol=0.0)
                profile = tree.exercise_profile()
                for row in profile[:-1]:
                    self.assertEqual(row.first_exercise_probability, 0.0)
                    self.assertEqual(row.survival_probability, 1.0)
                terminal_exercise = float(strike < 100.0)
                self.assertEqual(profile[-1].first_exercise_probability, terminal_exercise)
                self.assertEqual(profile[-1].survival_probability, 1.0 - terminal_exercise)
                for column in tree.iter_columns():
                    for node in tree.iter_column(column):
                        self.assertEqual(node.exercise_now, node.step == 5 and strike < 100.0)

    def test_call_americain_sans_dividende_pas_exerce_avant_maturite(self) -> None:
        tree = make_tree(n=40, kind="Call", strike=100, rate=0.05)
        profile = tree.exercise_profile()
        for row in profile[:-1]:
            self.assertEqual(row.first_exercise_probability, 0.0)
            self.assertEqual(row.exercise_region_probability, 0.0)
        self.assert_mass_balance(tree, profile)

    def test_modification_option_recalcule_decisions_et_masses(self) -> None:
        tree = make_tree(n=30, strike=110, rate=0.10, vol=0.20)
        initial = tree.exercise_profile()
        self.assertGreater(sum(row.first_exercise_probability for row in initial[:-1]), 0.0)

        # Le même arbre devient européen : aucune ancienne décision américaine
        # ne doit subsister. exercise_profile() doit refaire le backward pricing.
        tree.option.exercise_type = "European"
        european = tree.exercise_profile()
        for row in european[:-1]:
            self.assertEqual(row.first_exercise_probability, 0.0)
            self.assertEqual(row.exercise_region_probability, 0.0)

        # Changer ensuite le type ET le strike donne un contrat sans payoff.
        # Toute ancienne masse absorbée doit disparaître au second recalcul.
        tree.option.type = "Call"
        tree.option.strike = 10_000.0
        worthless = tree.exercise_profile()
        for row in worthless:
            self.assertEqual(row.discounted_mean_value, 0.0)
            self.assertEqual(row.first_exercise_probability, 0.0)
            self.assertEqual(row.cumulative_exercise_probability, 0.0)
            self.assertAlmostEqual(row.survival_probability, 1.0, delta=2e-12)
        self.assert_mass_balance(tree, worthless)
        for column in tree.iter_columns():
            for node in tree.iter_column(column):
                self.assertFalse(node.exercise_now)
                self.assertAlmostEqual(node.alive_proba, node.proba, delta=2e-12)

        # Vérifier enfin qu'une option redevenue payante se comporte exactement
        # comme une nouvelle construction, sans contamination par les appels.
        tree.option.strike = 80.0
        recalculated = tree.exercise_profile()
        fresh = make_tree(n=30, kind="Call", strike=80, exercise="European", rate=0.10, vol=0.20)
        independent = fresh.exercise_profile()
        for actual, expected in zip(recalculated, independent):
            self.assert_value_close(actual.discounted_mean_value, expected.discounted_mean_value)
            self.assert_value_close(actual.total_value, expected.total_value)
            self.assertAlmostEqual(actual.first_exercise_probability, expected.first_exercise_probability, delta=2e-12)
        # Les lignes déjà retournées gardent leurs valeurs, indépendamment de
        # l'option mutable qui a servi à les calculer.
        self.assertGreater(sum(row.first_exercise_probability for row in initial[:-1]), 0.0)
        self.assert_mass_balance(tree, recalculated)


if __name__ == "__main__":
    unittest.main(verbosity=2)
