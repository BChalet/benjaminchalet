"""Vérifier les résultats exposés à Streamlit, sans démarrer de serveur.

Le moteur possède déjà ses contrôles financiers et de graphe. Ici, on contrôle
le raccordement de l'interface : les bons paramètres doivent atteindre le
moteur, les tableaux doivent rester cohérents, et les études de convergence
doivent réellement utiliser une européenne complète, même si l'utilisateur
avait sélectionné une américaine avec pruning dans la barre latérale.

Lancer depuis le dossier du projet :
    python -m unittest -v test_interface_calculs

Les arbres sont volontairement petits. Une vérification du raccordement ne
nécessite ni benchmark à 1 000 pas ni navigateur ni serveur Streamlit.
"""

from __future__ import annotations

from dataclasses import replace
import datetime as dt
from math import exp, isfinite, log, sqrt
from statistics import NormalDist
import unittest
from xml.etree import ElementTree

from interface_calculs import (
    Parameters,
    calculate_summary,
    convergence_study,
    debug_tree,
    dividend_study,
    pruning_study,
    rate_study,
    strike_study,
)


def example_parameters(**changes) -> Parameters:
    """Cas de référence du classeur Excel, paramétrable pour chaque contrôle."""
    base = Parameters(
        pricing_date=dt.date(2026, 9, 15),
        maturity=dt.date(2027, 9, 2),
        spot=100.0,
        strike=101.0,
        rate=0.03,
        vol=0.20,
        dividend_yield=0.0,
        dividend_amount=0.0,
        ex_dividend_date=None,
        nb_steps=2,
        pruning_threshold=0.0,
        option_type="Call",
        exercise_type="European",
    )
    return replace(base, **changes)


def independent_european_price(parameters: Parameters) -> float:
    """Référence fermée sans cash ; calcul indépendant de l'adaptateur testé."""
    time = (parameters.maturity - parameters.pricing_date).days / 365.0
    normal = NormalDist()
    root_time = parameters.vol * sqrt(time)
    d1 = (
        log(parameters.spot / parameters.strike)
        + (parameters.rate - parameters.dividend_yield + parameters.vol**2 / 2) * time
    ) / root_time
    d2 = d1 - root_time
    discounted_spot = parameters.spot * exp(-parameters.dividend_yield * time)
    discounted_strike = parameters.strike * exp(-parameters.rate * time)
    if parameters.option_type == "Call":
        return discounted_spot * normal.cdf(d1) - discounted_strike * normal.cdf(d2)
    return discounted_strike * normal.cdf(-d2) - discounted_spot * normal.cdf(-d1)


class InterfaceCalculsTests(unittest.TestCase):
    def test_resume_reference_excel_et_comparaison_quatre_contrats(self) -> None:
        # La référence extérieure vérifie que strike, dates, marché et N ne
        # sont ni remplacés par une valeur d'exemple ni mal convertis par l'UI.
        parameters = example_parameters()
        result = calculate_summary(parameters)
        self.assertAlmostEqual(result["price"], 8.303144127326599, places=11)
        self.assertEqual(result["node_count"], 6)
        self.assertGreaterEqual(result["elapsed_seconds"], 0.0)
        self.assertAlmostEqual(
            result["black_scholes"], independent_european_price(parameters), places=11
        )

        comparison = result["comparison"]
        self.assertEqual(len(comparison), 4)
        self.assertEqual(
            set(zip(comparison["option_type"], comparison["exercise_type"])),
            {("Call", "European"), ("Call", "American"),
             ("Put", "European"), ("Put", "American")},
        )
        chosen = comparison[
            (comparison["option_type"] == "Call")
            & (comparison["exercise_type"] == "European")
        ]
        self.assertAlmostEqual(chosen.iloc[0]["price"], result["price"], places=12)
        # Le tableau ne doit pas laisser le prix principal sur le dernier
        # contrat repricé : son profil reste celui de la européenne choisie.
        self.assertEqual(len(result["profile"]), parameters.nb_steps + 1)
        self.assertAlmostEqual(
            result["profile"].iloc[0]["total_value"], result["price"], places=12
        )

    def test_resume_probabilites_et_valeur_sont_coherentes(self) -> None:
        parameters = example_parameters(
            option_type="Put", exercise_type="American", nb_steps=12,
            rate=0.07, vol=0.30, pruning_threshold=0.01,
        )
        result = calculate_summary(parameters)
        profile = result["profile"]
        # Vérifier les libellés de fin de vie affichés par l'interface à partir
        # des lignes du profil, plutôt que seulement leur somme affichée.
        first_exercises = profile["first_exercise_probability"]
        self.assertAlmostEqual(
            result["early_exercise_probability"], first_exercises.iloc[:-1].sum(), places=12
        )
        self.assertAlmostEqual(
            result["maturity_exercise_probability"], first_exercises.iloc[-1], places=12
        )
        self.assertAlmostEqual(
            result["expiration_probability"], profile.iloc[-1]["survival_probability"], places=12
        )
        self.assertAlmostEqual(
            result["early_exercise_probability"] + result["maturity_exercise_probability"]
            + result["expiration_probability"], 1.0, places=12
        )
        self.assertLess(result["mass_error"], 1e-11)
        self.assertLess(result["value_error"], 1e-10)
        self.assertLess(abs(profile["total_value"] - result["price"]).max(), 1e-10)
        self.assertIsNone(result["black_scholes"])
        self.assertTrue(result["black_scholes_reason"])

    def test_cash_inactif_ne_supprime_pas_la_reference_black_scholes(self) -> None:
        # Le moteur considère le spot déjà ex-dividende à la date initiale.
        # Un cash passé ou postérieur à la maturité ne doit donc pas masquer
        # une référence européenne parfaitement applicable.
        baseline = calculate_summary(example_parameters())
        for ex_date in (
            dt.date(2026, 9, 15),
            dt.date(2026, 8, 15),
            dt.date(2027, 10, 15),
        ):
            with self.subTest(ex_date=ex_date):
                result = calculate_summary(example_parameters(
                    dividend_amount=3.0, ex_dividend_date=ex_date,
                ))
                self.assertAlmostEqual(result["price"], baseline["price"], places=12)
                self.assertAlmostEqual(
                    result["black_scholes"], baseline["black_scholes"], places=12
                )

    def test_parametres_invalides_rejetes_avant_un_calcul(self) -> None:
        cases = (
            {"spot": 0.0},
            {"strike": -1.0},
            {"vol": -0.1},
            {"rate": float("nan")},
            {"maturity": dt.date(2026, 9, 15)},
            {"nb_steps": 0},
            {"nb_steps": True},
            {"nb_steps": 2001},
            {"pruning_threshold": -0.1},
            {"dividend_amount": 2.0, "ex_dividend_date": None},
            {"option_type": "Inconnu"},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    example_parameters(**changes).validate()

    def test_convergence_force_europeenne_complete(self) -> None:
        # On part exprès de paramètres qui n'ont aucune référence BS directe.
        # L'étude annonce une comparaison européenne : elle doit convertir
        # l'exercice et supprimer le pruning au lieu de comparer deux modèles.
        parameters = example_parameters(
            exercise_type="American", option_type="Put", pruning_threshold=0.20,
        )
        study = convergence_study(parameters, [2, 5, 8])
        self.assertEqual(study["nb_steps"].tolist(), [2, 5, 8])
        self.assertTrue((study["status"] == "ok").all())
        reference = independent_european_price(parameters)
        for row in study.itertuples(index=False):
            expected = calculate_summary(replace(
                parameters, nb_steps=int(row.nb_steps),
                exercise_type="European", pruning_threshold=0.0,
            ))
            self.assertAlmostEqual(row.price, expected["price"], places=12)
            self.assertAlmostEqual(row.reference, reference, places=12)
            self.assertAlmostEqual(row.error, row.price - reference, places=12)
            self.assertAlmostEqual(row.scaled_error, row.nb_steps * row.error, places=12)
            self.assertEqual(row.node_count, (row.nb_steps + 1) * (row.nb_steps + 2) // 2)

    def test_etude_strike_utilise_strikes_demandes(self) -> None:
        parameters = example_parameters(exercise_type="American", pruning_threshold=0.10)
        study = strike_study(parameters, [90.0, 110.0], steps=[6])
        self.assertEqual(study["strike"].tolist(), [90.0, 110.0])
        self.assertTrue((study["nb_steps"] == 6).all())
        for row in study.itertuples(index=False):
            contract = replace(parameters, strike=row.strike, exercise_type="European",
                               pruning_threshold=0.0, nb_steps=6)
            self.assertAlmostEqual(row.reference, independent_european_price(contract), places=12)
            self.assertAlmostEqual(row.price, calculate_summary(contract)["price"], places=12)
            self.assertAlmostEqual(row.error, row.price - row.reference, places=12)

    def test_tableau_pruning_compare_au_graphe_complet(self) -> None:
        parameters = example_parameters(option_type="Put", exercise_type="American")
        # Le formulaire peut contenir N=2 ; le N propre à l'étude doit gagner.
        # Même absente de la saisie, la référence seuil 0 doit rester présente.
        study = pruning_study(parameters, [0.05, 0.20], nb_steps=20)
        self.assertEqual(study["threshold"].tolist(), [0.0, 0.05, 0.20])
        self.assertTrue((study["status"] == "ok").all())
        baseline = calculate_summary(replace(parameters, nb_steps=20))
        self.assertAlmostEqual(study.iloc[0]["price"], baseline["price"], places=12)
        for row in study.itertuples(index=False):
            self.assertAlmostEqual(row.price_gap, row.price - baseline["price"], places=12)
            self.assertGreaterEqual(row.seconds, 0.0)
        self.assertLess(study.iloc[-1]["node_count"], study.iloc[0]["node_count"])
        self.assertGreater(study.iloc[-1]["pruned_count"], 0)

    def test_svg_et_diagnostic_proviennent_du_petit_arbre_demande(self) -> None:
        parameters = example_parameters(nb_steps=200, pruning_threshold=0.05)
        result = debug_tree(parameters, steps=6, color_by="probability", show_neighbors=True)
        root = ElementTree.fromstring(result["svg"])
        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
        self.assertEqual(result["errors"], 0)
        self.assertTrue(result["report"])
        self.assertLessEqual(result["node_count"], 28)
        self.assertGreater(result["pruned_count"], 0)
        # Les biais sont des mesures et non des erreurs de construction.
        self.assertTrue(isfinite(result["weighted_mean_bias"]))
        self.assertTrue(isfinite(result["terminal_mean_bias"]))

    def test_etude_taux_conserve_les_taux_negatifs_et_calcule_les_primes(self) -> None:
        parameters = example_parameters(nb_steps=4)
        study = rate_study(parameters, [-0.05, 0.0, 0.05])
        self.assertEqual(study["rate"].tolist(), [-0.05, 0.0, 0.05])
        self.assertTrue((study["status"] == "ok").all())
        for row in study.itertuples(index=False):
            self.assertAlmostEqual(
                row.call_premium, row.call_american - row.call_european, places=12
            )
            self.assertAlmostEqual(
                row.put_premium, row.put_american - row.put_european, places=12
            )
            self.assertGreaterEqual(row.call_premium, -1e-12)
            self.assertGreaterEqual(row.put_premium, -1e-12)
        negative_rate = calculate_summary(replace(parameters, rate=-0.05))
        self.assertAlmostEqual(study.iloc[0]["call_european"], negative_rate["price"], places=12)

    def test_etude_dividendes_continue_apres_un_cash_inadmissible(self) -> None:
        parameters = example_parameters(
            nb_steps=4, rate=0.0, vol=0.30,
            ex_dividend_date=dt.date(2027, 3, 15),
        )
        # Le scénario intermédiaire échoue dans la construction ; le dernier
        # scénario valide doit néanmoins produire ses quatre prix.
        study = dividend_study(parameters, [0.0, 1000.0, 2.0])
        self.assertEqual(study["status"].tolist(), ["ok", "error", "ok"])
        self.assertTrue(study.iloc[1]["error_message"])
        self.assertTrue(study.iloc[1:2]["call_european"].isna().all())
        last_contract = calculate_summary(replace(parameters, dividend_amount=2.0))
        self.assertAlmostEqual(study.iloc[2]["call_european"], last_contract["price"], places=12)
        self.assertGreaterEqual(study.iloc[2]["call_american"], study.iloc[2]["call_european"])
        self.assertGreaterEqual(study.iloc[2]["put_american"], study.iloc[2]["put_european"])

    def test_cash_actif_refuse_pour_convergence_et_explique_dans_resume(self) -> None:
        parameters = example_parameters(
            nb_steps=4, vol=0.30, dividend_amount=2.0,
            ex_dividend_date=dt.date(2027, 3, 15),
        )
        result = calculate_summary(parameters)
        self.assertIsNone(result["black_scholes"])
        self.assertTrue(result["black_scholes_reason"])
        with self.assertRaises(ValueError):
            convergence_study(parameters, [2, 4])
        with self.assertRaises(ValueError):
            strike_study(parameters, [90.0, 110.0], steps=[4])

    def test_tableau_pruning_erreurs_ne_produit_pas_un_faux_ecart(self) -> None:
        parameters = example_parameters(
            dividend_amount=1000.0, ex_dividend_date=dt.date(2027, 3, 15),
        )
        study = pruning_study(parameters, [0.01], nb_steps=2)
        self.assertTrue((study["status"] == "error").all())
        self.assertTrue(study["error_message"].str.len().gt(0).all())
        self.assertTrue(study["price"].isna().all())
        self.assertTrue(study["price_gap"].isna().all())


if __name__ == "__main__":
    unittest.main()
