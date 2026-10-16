"""Tests de diagnostic : distinguer un élagage valide d'un arbre corrompu.

Les corruptions sont intentionnelles et restent limitées à l'arbre du test.
On vérifie le contrat public ``check() -> nombre d'erreurs`` et le contenu
XML du SVG, sans figer les coordonnées ou les couleurs du dessin.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import datetime as dt
import io
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from arbre_binomial import Market, Model, Node, Option, Tree
from debug_arbre import DebugTree


class DebugTreeTests(unittest.TestCase):
    @staticmethod
    def make_tree(n: int = 8, *, pruning: float = 0, cash: float = 0, vol: float = 0.2) -> Tree:
        start = dt.date(2026, 9, 15)
        tree = Tree(
            Model(start, n, pruning_threshold=pruning),
            Option(101, start + dt.timedelta(days=365), "Put", "American"),
            Market(100, 0.03, vol, dividend_yield=0.02,
                   dividend_amount=cash,
                   ex_dividend_date=start + dt.timedelta(days=180) if cash else None),
        ).build()
        tree.price()
        return tree

    def checked(self, debugger: DebugTree) -> tuple[int, str]:
        # Le diagnostic reste affiché lors d'un usage normal. Le capturer ici
        # rend les tests lisibles et permet de vérifier erreurs/informations.
        output = io.StringIO()
        with redirect_stdout(output):
            errors = debugger.check()
        return errors, output.getvalue()

    def test_regular_tree_checks_zero_strike_and_restores_original_option(self) -> None:
        tree = self.make_tree()
        original = tree.option
        original_value = tree.root.option_value
        debugger = DebugTree(tree)
        errors, report = self.checked(debugger)
        self.assertEqual(errors, 0, report)
        self.assertIn("Call européen K=0", report)
        self.assertIs(tree.option, original)
        self.assertEqual(tree.root.option_value, original_value)
        first_count = debugger.node_count
        self.assertEqual(self.checked(debugger)[0], 0)
        self.assertEqual(debugger.node_count, first_count)

    def test_valid_pruning_and_cash_are_information_not_errors(self) -> None:
        tree = self.make_tree(20, pruning=1e-3, cash=3)
        debugger = DebugTree(tree)
        errors, report = self.checked(debugger)
        self.assertEqual(errors, 0, report)
        self.assertGreater(debugger.pruned_node_count, 0)
        self.assertGreater(debugger.pruning_weighted_abs_mean_bias, 0)
        self.assertNotEqual(debugger.discounted_terminal_mean_bias, 0)
        self.assertIn("test exact K=0 non appliqué", report)
        self.assertLess(tree.count_nodes(), self.make_tree(20, cash=3).count_nodes())

    def test_zero_volatility_with_cash(self) -> None:
        errors, report = self.checked(DebugTree(self.make_tree(20, cash=3, vol=0)))
        self.assertEqual(errors, 0, report)

    def test_numerical_exercise_tie_is_kept(self) -> None:
        start = dt.date(2026, 9, 15)
        tree = Tree(Model(start, 8), Option(101, start + dt.timedelta(days=365), "Put", "American"),
                    Market(100, 1e-15, 0)).build()
        tree.price()
        self.assertFalse(tree.root.exercise_now)
        errors, report = self.checked(DebugTree(tree))
        self.assertEqual(errors, 0, report)

    def test_wrong_incoming_probability_is_detected(self) -> None:
        tree = self.make_tree()
        tree.root.next_up.proba += 0.01
        errors, report = self.checked(DebugTree(tree))
        self.assertGreater(errors, 0)
        self.assertIn("somme entrante", report)
        self.assertIn("somme des probabilités", report)

    def test_broken_vertical_reciprocal_link_is_detected(self) -> None:
        tree = self.make_tree()
        top = tree.root.next_up
        self.assertIsNotNone(top.below)
        top.below.above = None
        errors, report = self.checked(DebugTree(tree))
        self.assertGreater(errors, 0)
        self.assertIn("lien below/above incorrect", report)

    def test_child_with_correct_spot_but_outside_column_is_detected(self) -> None:
        tree = self.make_tree()
        old_child = tree.root.next_up
        # Copier un prix et une date ne suffit pas : il faut l'identité du
        # nœud présent dans la chaîne de voisins de la colonne suivante.
        tree.root.next_up = Node(tree, old_child.spot_price, old_child.step)
        errors, report = self.checked(DebugTree(tree))
        self.assertGreater(errors, 0)
        self.assertIn("hors de la colonne suivante", report)

    def test_wrong_branch_probability_is_detected(self) -> None:
        tree = self.make_tree()
        tree.root.p_up += 0.1
        errors, report = self.checked(DebugTree(tree))
        self.assertGreater(errors, 0)
        self.assertIn("somme des probabilités locales", report)

    def test_terminal_payoff_corruption_is_detected(self) -> None:
        tree = self.make_tree()
        tree.last_trunk.option_value += 1
        errors, report = self.checked(DebugTree(tree))
        self.assertGreater(errors, 0)
        self.assertIn("valeur terminale différente du payoff", report)

    def test_svg_is_xml_with_nodes_probabilities_exercise_and_pruning(self) -> None:
        tree = self.make_tree(20, pruning=1e-3, cash=3)
        debugger = DebugTree(tree)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            path = debugger.export_svg(Path(folder) / "arbre.svg", show_neighbors=True)
            root = ET.parse(path).getroot()
            namespace = {"svg": "http://www.w3.org/2000/svg"}
            count = sum(1 for group in root.findall(".//svg:g", namespace)
                        if group.attrib.get("id", "").startswith("node-"))
            self.assertEqual(count, tree.count_nodes())
            labels = " ".join(root.itertext())
            self.assertIn("π =", labels)
            self.assertIn("p=1", labels)
            self.assertIn("PRUNING", labels)
            self.assertIn("EXERCER", labels)
            self.assertIn("Δm=", labels)
            self.assertIn("payoff=", labels)
            self.assertIn("continuation=", labels)
            self.assertIn("viewBox", root.attrib)
            # Les annotations restent du texte vectoriel, et non une image
            # raster incorporée à un fichier portant l'extension .svg.
            self.assertEqual(root.findall(".//svg:image", namespace), [])

    def test_drawing_limit_is_node_count_and_explicit(self) -> None:
        tree = self.make_tree(20)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "limite explicite 10"):
                DebugTree(tree).export_svg(Path(folder) / "trop_grand.svg", max_nodes=10)
            self.assertFalse((Path(folder) / "trop_grand.svg").exists())


if __name__ == "__main__":
    unittest.main()
