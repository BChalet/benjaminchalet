"""Contrôles financiers du modèle, exécutables avec ``python -m unittest -v``.

Les références Excel viennent des deux classeurs fournis. Les autres contrôles
utilisent des identités financières et les moments conditionnels du cours,
calculés indépendamment à partir des dates et des liens entre nœuds.
"""

from __future__ import annotations

import datetime as dt
import gc
from math import exp, expm1, log, sqrt
from statistics import NormalDist
import unittest

from arbre_binomial import Market, Model, Option, Tree
from convergence import Convergence


PRICING_DATE = dt.date(2026, 9, 15)
MATURITY = dt.date(2027, 9, 2)
EX_DATE = dt.date(2027, 3, 15)


def make_tree(
    n: int = 100,
    *,
    strike: float = 101.0,
    kind: str = "Call",
    exercise: str = "European",
    rate: float = 0.03,
    vol: float = 0.20,
    q: float = 0.0,
    cash: float = 0.0,
    ex_date: dt.date | None = None,
    maturity: dt.date = MATURITY,
) -> Tree:
    return Tree(
        Model(PRICING_DATE, n),
        Option(strike, maturity, kind, exercise),
        Market(100.0, rate, vol, q, cash, ex_date),
    )


class DividendTests(unittest.TestCase):
    def tearDown(self) -> None:
        # Les liens parent/enfant/voisins forment des cycles ; éviter de garder
        # les arbres d'un contrôle en mémoire pendant le suivant.
        gc.collect()

    def test_reference_excel_deux_pas(self) -> None:
        tree = make_tree(2).build()
        self.assertAlmostEqual(tree.price(), 8.303144127326599, places=11)
        expected_spots = (136.04429920303036, 102.93540883558833, 77.88417783193447)
        count = 0
        for node, expected in zip(tree.iter_column(tree.last_trunk), expected_spots):
            self.assertAlmostEqual(node.spot_price, expected, places=11)
            count += 1
        self.assertEqual(count, 3)
        self.assertEqual(sum(1 for _ in tree.iter_column(tree.last_trunk)), 3)

    def test_black_scholes_rendement_continu_reference(self) -> None:
        tree = make_tree(
            strike=100, rate=0.05, q=0.02,
            maturity=PRICING_DATE + dt.timedelta(days=365),
        )
        study = Convergence(tree.market, tree.option, PRICING_DATE)
        self.assertAlmostEqual(study.black_price(100), 9.227005508154036, places=12)

    def test_convergence_sans_cash_mille_pas(self) -> None:
        # Deux arbres seulement : cas du cours et rendement continu q = 2 %.
        normal = NormalDist()
        cases = (
            dict(q=0.0, rate=0.03, strike=101.0, maturity=MATURITY),
            dict(q=0.02, rate=0.05, strike=100.0,
                 maturity=PRICING_DATE + dt.timedelta(days=365)),
        )
        for parameters in cases:
            with self.subTest(**parameters):
                maturity = (parameters["maturity"] - PRICING_DATE).days / 365.0
                strike, rate, q = parameters["strike"], parameters["rate"], parameters["q"]
                d1 = (log(100 / strike) + (rate - q + 0.20**2 / 2) * maturity) / (0.20 * sqrt(maturity))
                d2 = d1 - 0.20 * sqrt(maturity)
                reference = 100 * exp(-q * maturity) * normal.cdf(d1) - strike * exp(-rate * maturity) * normal.cdf(d2)
                tree = make_tree(1000, **parameters).build()
                self.assertLess(abs(tree.price() - reference), 0.006)
                del tree
                gc.collect()

    def test_strike_nul_avec_dividendes_continu_et_cash(self) -> None:
        q, cash, rate = 0.02, 3.0, 0.03
        tree = make_tree(100, strike=0, q=q, cash=cash, ex_date=EX_DATE).build()
        maturity = (MATURITY - PRICING_DATE).days / 365.0
        ex_time = (EX_DATE - PRICING_DATE).days / 365.0
        # Actualiser séparément les flux : le cash réduit aussi les futurs
        # dividendes proportionnels, d'où le facteur q après son détachement.
        reference = 100 * exp(-q * maturity) - cash * exp(-rate * ex_time - q * (maturity - ex_time))
        self.assertAlmostEqual(tree.price(), reference, places=10)

    def test_parite_call_put_avec_dividendes_continu_et_cash(self) -> None:
        q, cash, rate, strike = 0.02, 3.0, 0.03, 101.0
        tree = make_tree(100, q=q, cash=cash, ex_date=EX_DATE).build()
        call = tree.price()
        tree.option.type = "Put"
        put = tree.price()
        maturity = (MATURITY - PRICING_DATE).days / 365.0
        ex_time = (EX_DATE - PRICING_DATE).days / 365.0
        reference = (
            100 * exp(-q * maturity)
            - cash * exp(-rate * ex_time - q * (maturity - ex_time))
            - strike * exp(-rate * maturity)
        )
        self.assertAlmostEqual(call - put, reference, places=10)

    def test_volatilite_nulle_cash_et_q(self) -> None:
        rate, q, cash = 0.03, 0.02, 3.0
        maturity = (MATURITY - PRICING_DATE).days / 365.0
        ex_time = (EX_DATE - PRICING_DATE).days / 365.0
        final_spot = 100 * exp((rate - q) * maturity) - cash * exp((rate - q) * (maturity - ex_time))
        tree = make_tree(100, vol=0, q=q, cash=cash, ex_date=EX_DATE).build()
        for kind in ("Call", "Put"):
            with self.subTest(kind=kind):
                tree.option.type = kind
                payoff = max(final_spot - 101, 0) if kind == "Call" else max(101 - final_spot, 0)
                self.assertAlmostEqual(tree.price(), exp(-rate * maturity) * payoff, places=11)
        self.assertAlmostEqual(tree.last_trunk.spot_price, final_spot, places=10)
        self.assertEqual(sum(1 for _ in tree.iter_column(tree.last_trunk)), 1)

    def test_dates_cash_bornes_et_alignement(self) -> None:
        # La période du cours compte 352 jours, dont 181 avant le 15 mars.
        cases = (
            (352, PRICING_DATE, None),
            (352, PRICING_DATE - dt.timedelta(days=1), None),
            (352, MATURITY + dt.timedelta(days=1), None),
            (352, MATURITY, 352),
            (352, EX_DATE, 181),
            (353, EX_DATE, 182),
            (100, EX_DATE, 52),
        )
        for n, ex_date, expected_step in cases:
            with self.subTest(n=n, ex_date=ex_date):
                tree = make_tree(n, cash=3, ex_date=ex_date)
                self.assertEqual(tree.dividend_step, expected_step)
                if expected_step is not None:
                    self.assertEqual(tree.transition_type(expected_step - 1), "trinomial")
                    if expected_step < n:
                        self.assertEqual(tree.transition_type(expected_step), "binomial")

    def test_cash_deja_detache_ou_apres_maturite_sans_effet(self) -> None:
        baseline = make_tree(20).build().price()
        for ex_date in (PRICING_DATE, MATURITY + dt.timedelta(days=1)):
            with self.subTest(ex_date=ex_date):
                tree = make_tree(20, cash=3, ex_date=ex_date).build()
                self.assertAlmostEqual(tree.price(), baseline, places=12)

    def test_cash_a_maturite_deduit_du_payoff_final(self) -> None:
        tree = make_tree(20, strike=0, cash=3, ex_date=MATURITY).build()
        duration = (MATURITY - PRICING_DATE).days / 365.0
        self.assertAlmostEqual(tree.price(), 100 - 3 * exp(-0.03 * duration), places=10)

    def test_moments_conditionnels_sur_branches_reelles(self) -> None:
        # Vérifier les trois contraintes sur les enfants effectivement reliés,
        # pour un détachement aligné puis à l'intérieur d'un pas.
        for n in (352, 100):
            with self.subTest(n=n):
                tree = make_tree(n, q=0.02, cash=3, ex_date=EX_DATE).build()
                delta = (MATURITY - PRICING_DATE).days / (365.0 * n)
                drift = 0.03 - 0.02
                ex_time = (EX_DATE - PRICING_DATE).days / 365.0
                for column in tree.iter_columns():
                    if column.step == n:
                        break
                    start, end = column.step * delta, (column.step + 1) * delta
                    cash_here = start < ex_time <= end + 1e-14
                    for node in tree.iter_column(column):
                        probability_sum = 0.0
                        mean = 0.0
                        for branch in ("up", "mid", "down"):
                            probability = getattr(node, "p_" + branch)
                            child = getattr(node, "next_" + branch)
                            self.assertGreaterEqual(probability, 0)
                            self.assertLessEqual(probability, 1)
                            if child is not None:
                                probability_sum += probability
                                mean += probability * child.spot_price
                        self.assertAlmostEqual(probability_sum, 1.0, places=12)
                        expected_mean = node.spot_price * exp(drift * delta)
                        variance_spot = node.spot_price
                        if cash_here:
                            remainder = end - ex_time
                            expected_mean -= 3 * exp(drift * remainder)
                            variance_spot -= 3 * remainder / delta
                            self.assertIsNotNone(node.next_mid)
                        else:
                            self.assertIsNone(node.next_mid)
                        expected_variance = variance_spot**2 * exp(2 * drift * delta) * expm1(0.20**2 * delta)
                        variance = 0.0
                        for branch in ("up", "mid", "down"):
                            child = getattr(node, "next_" + branch)
                            if child is not None:
                                variance += getattr(node, "p_" + branch) * (child.spot_price - mean)**2
                        self.assertAlmostEqual(mean, expected_mean, delta=max(1e-10, abs(expected_mean) * 1e-12))
                        self.assertAlmostEqual(variance, expected_variance, delta=max(1e-10, expected_variance * 1e-10))

    def test_exercice_americain_superieur_ou_egal_europeen(self) -> None:
        for kind in ("Call", "Put"):
            with self.subTest(kind=kind):
                tree = make_tree(50, kind=kind, q=0.02, cash=3, ex_date=EX_DATE).build()
                european = tree.price()
                tree.option.exercise_type = "American"
                self.assertGreaterEqual(tree.price() + 1e-12, european)

    def test_effet_cash_a_taux_nul_et_put_americain(self) -> None:
        no_cash = make_tree(100, rate=0).build()
        with_cash = make_tree(100, rate=0, cash=3, ex_date=EX_DATE).build()
        self.assertLess(with_cash.price(), no_cash.price())
        no_cash.option.type = with_cash.option.type = "Put"
        european = with_cash.price()
        self.assertGreater(european, no_cash.price())
        # À r=q=0, payer un cash rend le sous-jacent supermartingale : attendre
        # est optimal pour le put européen/ américain sur ce maillage.
        with_cash.option.exercise_type = "American"
        self.assertAlmostEqual(with_cash.price(), european, places=11)

    def test_gardes_sur_entrees_et_cash_non_representable(self) -> None:
        for n in (0, -1, True, 2.5):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    Model(PRICING_DATE, n)
        for q, cash in ((-0.01, 0), (0, -3)):
            with self.subTest(q=q, cash=cash):
                with self.assertRaises(ValueError):
                    Market(100, 0.03, 0.20, q, cash, EX_DATE)
        with self.assertRaises(ValueError):
            Market(100, 0.03, 0.20, dividend_amount=3)
        with self.assertRaisesRegex(ValueError, "non positive"):
            make_tree(1, rate=0, cash=100, ex_date=MATURITY).build()
        # E[S_T] = 20 reste positif, mais la variance avant détachement est
        # trop grande pour les trois prix locaux : refuser la probabilité < 0.
        with self.assertRaisesRegex(ValueError, "probabilités trinômiales invalides"):
            make_tree(1, rate=0, cash=80, ex_date=MATURITY).build()

    def test_black_scholes_refuse_cash_actif(self) -> None:
        tree = make_tree(20, q=0.02, cash=3, ex_date=EX_DATE)
        with self.assertRaisesRegex(ValueError, "référence exacte"):
            Convergence(tree.market, tree.option, PRICING_DATE).black_price(101)

    def test_construction_repetee_et_pricing_avant_build(self) -> None:
        tree = make_tree(20, cash=3, ex_date=EX_DATE)
        with self.assertRaisesRegex(RuntimeError, "build"):
            tree.price()
        tree.build()
        first_child = tree.root.next_trunk
        terminal = tree.last_trunk
        initial_price = tree.price()
        self.assertIs(tree.build(), tree)
        self.assertIs(tree.root.next_trunk, first_child)
        self.assertIs(tree.last_trunk, terminal)
        self.assertEqual(sum(1 for _ in tree.iter_columns()), 21)
        self.assertEqual(tree.price(), initial_price)


if __name__ == "__main__":
    unittest.main(verbosity=2)
