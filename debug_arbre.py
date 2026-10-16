"""Vérifier un arbre chaîné et l'exporter en SVG, sans Graphviz.

Le débogueur a deux usages complémentaires :

* ``DebugTree(tree).check()`` parcourt tous les liens et renvoie un nombre
  d'erreurs. Il convient aussi aux grands arbres, sans fenêtre graphique.
* ``DebugTree(tree).export_svg("arbre.svg")`` produit un dessin vectoriel :
  on peut ouvrir le fichier dans un navigateur et zoomer sans pixellisation.

Comme le demande le cours (p. 18), aucun tableau, dictionnaire ou liste de
nœuds ne sert à accéder à l'arbre : on suit uniquement les liens du tronc et
les voisins ``above``/``below``. Les petits tuples de noms de branches ne
contiennent que des chaînes de caractères, jamais des nœuds.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Iterator, TextIO
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex

from arbre_binomial import Market, Model, Node, Option, Tree, TrunkNode


# Limiter le nombre de nœuds plutôt que le nombre de dates permet de montrer
# un pruning sur 20 pas, tout en évitant de produire un SVG de plusieurs Mo
# par inadvertance. L'appelant peut augmenter cette limite explicitement.
DEFAULT_MAX_DRAW_NODES = 1_200
BRANCHES = ("up", "mid", "down")


class DebugTree:
    """Contrôler structure, probabilités, moments et valeurs de l'option.

    Trois curseurs de parents suivent respectivement les branches up, mid et
    down, dans l'ordre décroissant des spots. Quand on descend dans la colonne
    des enfants, ces curseurs ne remontent jamais : les probabilités entrantes
    sont vérifiées en temps linéaire, sans rechercher tous les parents pour
    chaque enfant. Le même principe valide l'appartenance des enfants à la
    colonne suivante.

    Un nœud élagué envoie toute sa masse vers un enfant intérieur. Cela
    conserve la probabilité totale, mais change les moments locaux. Ce biais
    est mesuré et annoncé ; il ne constitue pas une erreur de construction.
    """

    def __init__(self, tree: Tree) -> None:
        self.tree = tree
        self.output = Path(__file__).resolve().parent
        self.errors = self.node_count = self.branch_count = 0
        self.pruned_node_count = 0
        self.pruning_weighted_mean_bias = 0.0
        self.pruning_weighted_abs_mean_bias = 0.0
        self.pruning_weighted_variance_bias = 0.0
        self.pruning_max_abs_mean_bias = 0.0
        self.terminal_spot_expectation = 0.0
        self.discounted_terminal_mean_bias = 0.0
        self._previous_has_pruning = False
        self._child_up: Node | None = None
        self._child_mid: Node | None = None
        self._child_down: Node | None = None
        self._parent_up: Node | None = None
        self._parent_mid: Node | None = None
        self._parent_down: Node | None = None

    @staticmethod
    def close(actual: float, expected: float) -> bool:
        """Tolérance pour les sommes et moments, sans exiger l'égalité binaire."""
        return bool(np.isclose(actual, expected, rtol=1e-9, atol=1e-10))

    def error(self, message: str, node: Node | None = None) -> None:
        self.errors += 1
        location = "" if node is None else f"Pas {node.step}, S={node.spot_price:.6f} : "
        print("ERREUR :", location + message)

    def columns(self) -> Iterator[TrunkNode]:
        """Parcours borné : un cycle du tronc ne peut pas boucler indéfiniment."""
        trunk = self.tree.root
        previous = None
        for step in range(self.tree.model.nb_steps + 1):
            if not isinstance(trunk, TrunkNode) or trunk.step != step:
                raise ValueError(f"Tronc absent ou mauvaise date au pas {step}")
            if trunk.previous_trunk is not previous:
                raise ValueError(f"Lien précédent du tronc incorrect au pas {step}")
            yield trunk
            previous, trunk = trunk, trunk.next_trunk
        if trunk is not None or previous is not self.tree.last_trunk:
            raise ValueError("Fin du tronc incorrecte")

    def top(self, trunk: TrunkNode) -> Node:
        node: Node = trunk
        moves = 0
        while node.above is not None:
            node = node.above
            moves += 1
            if moves >= trunk.column_size:
                raise ValueError(f"Cycle ou trop de voisins au-dessus au pas {trunk.step}")
        return node

    def column_nodes(self, trunk: TrunkNode) -> Iterator[Node]:
        """Valider la taille annoncée et la présence du tronc dans la chaîne."""
        if trunk.column_size < 1:
            raise ValueError(f"Taille de colonne invalide au pas {trunk.step}")
        node: Node | None = self.top(trunk)
        count = 0
        found_trunk = False
        while node is not None:
            count += 1
            if count > trunk.column_size or node.step != trunk.step:
                raise ValueError(f"Colonne incohérente au pas {trunk.step}")
            found_trunk = found_trunk or node is trunk
            yield node
            node = node.below
        if count != trunk.column_size or not found_trunk:
            raise ValueError(
                f"Colonne {trunk.step} : {count} nœuds ; taille annoncée {trunk.column_size}"
            )

    def is_pruned(self, node: Node) -> bool:
        # Le moteur expose ce drapeau. Le repli permet également d'inspecter
        # les anciens arbres, qui représentaient le pruning par leur forme.
        marker = getattr(node, "is_pruned", None)
        if marker is not None:
            return bool(marker)
        return (
            self.tree.transition_type(node.step) == "binomial"
            and node.next_mid is not None
            and node.next_up is None and node.next_down is None
        )

    def in_column(self, child: Node, trunk: TrunkNode, branch: str) -> bool:
        """Appeler dans l'ordre des parents, de haut en bas, pour chaque branche."""
        attribute = "_child_" + branch
        cursor = getattr(self, attribute)
        moves = 0
        while cursor is not None and cursor.spot_price > child.spot_price:
            cursor = cursor.below
            moves += 1
            if moves > trunk.column_size:
                raise ValueError("Cycle pendant la recherche d'un enfant")
        setattr(self, attribute, cursor)
        return cursor is child

    def check_child(self, node: Node, child: Node | None, trunk: TrunkNode, branch: str) -> None:
        probability = getattr(node, "p_" + branch)
        if child is None:
            if abs(probability) > 1e-12:
                self.error(f"enfant {branch} manquant avec probabilité non nulle", node)
            return
        self.branch_count += 1
        if child.step != node.step + 1 or not self.in_column(child, trunk, branch):
            self.error(f"enfant {branch} hors de la colonne suivante", node)
        if not np.isfinite(child.spot_price) or child.spot_price <= 0:
            self.error(f"prix de l'enfant {branch} non positif ou non fini", node)

    def check_neighbors(self, node: Node) -> None:
        if node.above is not None:
            if node.above.step != node.step or node.above.below is not node:
                self.error("lien above/below incorrect", node)
        if node.below is not None:
            if node.below.step != node.step or node.below.above is not node:
                self.error("lien below/above incorrect", node)
            if node.spot_price <= node.below.spot_price:
                self.error("prix mal ordonnés dans la colonne", node)
            if self.tree.transition_type(node.step) == "binomial":
                # Un bord élagué conserve le même enfant partagé, mais ce lien
                # s'appelle next_mid et sa probabilité devient 1.
                upper_down = node.next_mid if self.is_pruned(node) else node.next_down
                lower_up = node.below.next_mid if self.is_pruned(node.below) else node.below.next_up
                if upper_down is None or upper_down is not lower_up:
                    self.error("recombinaison binomiale incorrecte", node)

    def incoming(self, node: Node, previous: TrunkNode, branch: str) -> tuple[int, float]:
        """Nombre de branches entrantes et somme π(parent) × p(parent,enfant)."""
        attribute = "_parent_" + branch
        cursor = getattr(self, attribute)
        count = 0
        mass = 0.0
        moves = 0
        while cursor is not None:
            child = getattr(cursor, "next_" + branch)
            # Le parent courant doit être conservé pour l'enfant suivant :
            # son enfant est déjà situé plus bas que celui que l'on contrôle.
            if child is not None and child.spot_price < node.spot_price:
                break
            if child is node:
                count += 1
                mass += cursor.proba * getattr(cursor, "p_" + branch)
            cursor = cursor.below
            moves += 1
            if moves > previous.column_size:
                raise ValueError("Cycle pendant la recherche des parents")
        setattr(self, attribute, cursor)
        return count, mass

    def check_parents(self, node: Node, previous: TrunkNode | None) -> None:
        if not np.isfinite(node.proba) or not -1e-12 <= node.proba <= 1 + 1e-12:
            self.error("probabilité d'atteinte en dehors de [0,1]", node)
        if previous is None:
            if node is not self.tree.root:
                self.error("nœud supplémentaire à la date initiale", node)
            if not self.close(node.proba, 1):
                self.error("la probabilité de la racine doit valoir 1", node)
            return
        count = 0
        mass = 0.0
        for branch in BRANCHES:
            branch_count, branch_mass = self.incoming(node, previous, branch)
            count += branch_count
            mass += branch_mass
        if count == 0:
            self.error("nœud sans branche entrante", node)
        if not self.close(node.proba, mass):
            self.error(f"probabilité stockée {node.proba:.12g} ≠ somme entrante {mass:.12g}", node)
        kind = self.tree.transition_type(previous.step)
        if kind == "binomial" and not self._previous_has_pruning:
            expected = 1 if node.above is None or node.below is None else 2
            if count != expected:
                self.error(f"{count} branches entrantes au lieu de {expected}", node)
        elif kind == "deterministic" and count != 1:
            self.error(f"{count} branches entrantes dans une transition déterministe", node)
        # Avec pruning, le nouveau bord peut recevoir deux branches. La règle
        # naïve « un bord a exactement un parent » ne s'applique donc plus.

    def check_parameters(self) -> None:
        expected_dt = (
            (self.tree.option.maturity - self.tree.model.pricing_date).days
            / 365 / self.tree.model.nb_steps
        )
        if not self.close(self.tree.deltaT, expected_dt):
            self.error("pas de temps incorrect")
        if not self.close(self.tree.df, np.exp(-self.tree.market.rate * expected_dt)):
            self.error("facteur d'actualisation incorrect")
        if not self.close(self.tree.root.spot_price, self.tree.market.spot):
            self.error("prix de la racine différent du spot")
        if self.tree.market.vol > 0:
            up, down = self.tree.p_up, self.tree.p_down
            if not (0 <= up <= 1 and 0 <= down <= 1) or not self.close(up + down, 1):
                self.error("probabilités binomiales communes incorrectes")

    @staticmethod
    def local_moments(node: Node) -> tuple[float, float]:
        """Calculer Var(S') par les écarts à la moyenne, plus stable que E[S'²]−E[S']²."""
        mean = 0.0
        for branch in BRANCHES:
            child = getattr(node, "next_" + branch)
            if child is not None:
                mean += getattr(node, "p_" + branch) * child.spot_price
        variance = 0.0
        for branch in BRANCHES:
            child = getattr(node, "next_" + branch)
            if child is not None:
                variance += getattr(node, "p_" + branch) * (child.spot_price - mean) ** 2
        return mean, variance

    def check_moments(self, node: Node) -> None:
        probability_sum = 0.0
        for branch in BRANCHES:
            probability = getattr(node, "p_" + branch)
            probability_sum += probability
            if not np.isfinite(probability) or not -1e-12 <= probability <= 1 + 1e-12:
                self.error(f"probabilité {branch} en dehors de [0,1]", node)
        if not self.close(probability_sum, 1):
            self.error("la somme des probabilités locales ne vaut pas 1", node)
        mean, variance = self.local_moments(node)
        kind = self.tree.transition_type(node.step)
        if self.is_pruned(node):
            self.pruned_node_count += 1
            if kind != "binomial" or (node.above is not None and node.below is not None):
                self.error("le pruning du moteur doit concerner un bord binomial", node)
            if node.proba >= self.tree.model.pruning_threshold:
                self.error("nœud élagué alors que sa probabilité n'est pas sous le seuil", node)
            if node.next_up is not None or node.next_down is not None or node.next_mid is None:
                self.error("pruning : un seul enfant central doit subsister", node)
            if not self.close(node.p_mid, 1) or not self.close(node.p_up + node.p_down, 0):
                self.error("pruning : la branche conservée doit avoir probabilité 1", node)
            mean_bias = mean - node.forward()
            variance_bias = variance - node.variance()
            self.pruning_weighted_mean_bias += node.proba * mean_bias
            self.pruning_weighted_abs_mean_bias += node.proba * abs(mean_bias)
            self.pruning_weighted_variance_bias += node.proba * variance_bias
            self.pruning_max_abs_mean_bias = max(self.pruning_max_abs_mean_bias, abs(mean_bias))
            return  # Ces deux moments sont volontairement altérés par l'élagage.
        if not self.close(mean, node.forward()):
            self.error("espérance différente du forward après dividendes", node)
        if not self.close(variance, node.variance()):
            self.error("variance différente de la variance prévue par le modèle", node)
        if kind == "binomial":
            if node.next_mid is not None:
                self.error("branche centrale non marquée comme pruning en transition binomiale", node)
            if node.next_up is None or node.next_down is None:
                self.error("deux enfants requis dans une transition binomiale", node)
            else:
                if not self.close(node.next_up.spot_price, node.forward() * self.tree.sqrtAlpha):
                    self.error("prix de l'enfant haut binomial incorrect", node)
                if not self.close(node.next_down.spot_price, node.forward() / self.tree.sqrtAlpha):
                    self.error("prix de l'enfant bas binomial incorrect", node)
            if not self.close(node.p_up, self.tree.p_up) or not self.close(node.p_down, self.tree.p_down):
                self.error("probabilités binomiales différentes des paramètres communs", node)
        elif kind == "trinomial":
            if node.next_up is None or node.next_mid is None or node.next_down is None:
                self.error("trois enfants requis au détachement discret", node)
            else:
                if node.next_up.below is not node.next_mid or node.next_mid.below is not node.next_down:
                    self.error("les trois enfants ne sont pas voisins sur la grille", node)
                if not self.close(node.next_up.spot_price / node.next_mid.spot_price, self.tree.alpha):
                    self.error("rapport haut/milieu incorrect sur la grille", node)
                if not self.close(node.next_mid.spot_price / node.next_down.spot_price, self.tree.alpha):
                    self.error("rapport milieu/bas incorrect sur la grille", node)
        elif kind == "deterministic":
            if node.next_up is not None or node.next_down is not None or node.next_mid is None:
                self.error("transition déterministe : seul l'enfant central doit exister", node)
            if not self.close(node.p_mid, 1):
                self.error("transition déterministe : probabilité centrale différente de 1", node)

    def check_valuation(self, node: Node) -> None:
        """Contrôler une valorisation déjà calculée ; ne pas imposer price() à check()."""
        if node.option_value is None:
            return
        if not np.isfinite(node.option_value):
            self.error("valeur de l'option non finie", node)
            return
        intrinsic = self.tree.option.intrinsic_value(node.spot_price)
        if node.step == self.tree.model.nb_steps:
            if not self.close(node.option_value, intrinsic):
                self.error("valeur terminale différente du payoff", node)
            if hasattr(node, "exercise_now") and bool(node.exercise_now) != (intrinsic > 0):
                self.error("décision terminale incompatible avec un payoff positif", node)
            return
        continuation = 0.0
        for branch in BRANCHES:
            child = getattr(node, "next_" + branch)
            if child is not None:
                if child.option_value is None:
                    self.error("enfant non valorisé alors que le parent l'est", node)
                    return
                continuation += getattr(node, "p_" + branch) * child.option_value
        continuation *= self.tree.df
        american = self.tree.option.exercise_type.lower() == "american"
        tolerance = self.tree.exercise_tolerance(intrinsic, continuation)
        should_exercise = american and intrinsic > 0 and intrinsic > continuation + tolerance
        expected = intrinsic if should_exercise else continuation
        if not self.close(node.option_value, expected):
            self.error("valeur incompatible avec continuation/exercice", node)
        stored = getattr(node, "continuation_value", None)
        if stored is not None and not self.close(stored, continuation):
            self.error("valeur de continuation stockée incorrecte", node)
        decision = getattr(node, "exercise_now", None)
        if decision is not None and bool(decision) != should_exercise:
            self.error("décision d'exercice incorrecte (à égalité numérique, conserver)", node)

    def check_column(self, trunk: TrunkNode) -> None:
        if trunk.next_trunk is not None:
            first = self.top(trunk.next_trunk)
            self._child_up = self._child_mid = self._child_down = first
        if trunk.previous_trunk is not None:
            first = self.top(trunk.previous_trunk)
            self._parent_up = self._parent_mid = self._parent_down = first
        mass = 0.0
        current_has_pruning = False
        for node in self.column_nodes(trunk):
            self.node_count += 1
            mass += node.proba
            current_has_pruning = current_has_pruning or self.is_pruned(node)
            if node.tree is not self.tree:
                self.error("référence vers un autre arbre", node)
            if not np.isfinite(node.spot_price) or node.spot_price <= 0:
                self.error("prix du nœud non positif ou non fini", node)
            self.check_neighbors(node)
            self.check_parents(node, trunk.previous_trunk)
            if node.step < self.tree.model.nb_steps:
                if not isinstance(trunk.next_trunk, TrunkNode):
                    raise ValueError("Colonne suivante absente")
                for branch in BRANCHES:
                    self.check_child(node, getattr(node, "next_" + branch), trunk.next_trunk, branch)
                self.check_moments(node)
            else:
                self.terminal_spot_expectation += node.proba * node.spot_price
                if any(getattr(node, "next_" + branch) is not None for branch in BRANCHES):
                    self.error("branche après maturité", node)
                if self.is_pruned(node):
                    self.error("un nœud terminal n'a pas de transition à élaguer", node)
            self.check_valuation(node)
        if not self.close(mass, 1):
            self.error(f"colonne {trunk.step} : somme des probabilités = {mass:.12g}, au lieu de 1")
        self._previous_has_pruning = current_has_pruning

    def check_zero_strike(self) -> None:
        """Identité call européen K=0 = forward actualisé, seulement sans pruning.

        Le changement d'option est temporaire. Un finally remet l'option de
        l'utilisateur et recalcule ses valeurs/décisions dans tous les nœuds.
        """
        original = self.tree.option
        try:
            self.tree.option = Option(0, original.maturity, "Call", "European")
            price = self.tree.price()
            expected = self.tree.expected_zero_strike()
            if not self.close(price, expected):
                self.error("le call européen de strike nul diffère du forward actualisé")
            print(f"Call européen K=0 : {price:.10f} ; forward actualisé : {expected:.10f}")
        finally:
            self.tree.option = original
            self.tree.price()

    def check(self) -> int:
        """Afficher le diagnostic et renvoyer le nombre d'erreurs, réinitialisé à chaque appel."""
        self.errors = self.node_count = self.branch_count = self.pruned_node_count = 0
        self.pruning_weighted_mean_bias = self.pruning_weighted_abs_mean_bias = 0.0
        self.pruning_weighted_variance_bias = self.pruning_max_abs_mean_bias = 0.0
        self.terminal_spot_expectation = self.discounted_terminal_mean_bias = 0.0
        self._previous_has_pruning = False
        try:
            self.check_parameters()
            for trunk in self.columns():
                self.check_column(trunk)
            if self.errors == 0:
                if self.pruned_node_count == 0:
                    self.check_zero_strike()
                else:
                    discounted_mean = self.tree.df**self.tree.model.nb_steps * self.terminal_spot_expectation
                    self.discounted_terminal_mean_bias = discounted_mean - self.tree.expected_zero_strike()
                    print(
                        f"PRUNING : {self.pruned_node_count} transitions monomiales ; "
                        f"Σ π·|Δm| = {self.pruning_weighted_abs_mean_bias:.6g} ; "
                        f"max |Δm| = {self.pruning_max_abs_mean_bias:.6g}."
                    )
                    print(
                        f"Biais de E[S(T)] actualisé : {self.discounted_terminal_mean_bias:+.10g} ; "
                        f"Σ π·Δvariance = {self.pruning_weighted_variance_bias:+.6g}."
                    )
                    print("Information : les moments des nœuds coupés sont altérés ; test exact K=0 non appliqué.")
        except (ValueError, AttributeError, TypeError, ZeroDivisionError, OverflowError) as exc:
            self.error(str(exc))
        print(f"DEBUG : {self.node_count} nœuds parcourus, {self.branch_count} branches")
        print(f"Bilan : {self.errors} erreur(s) détectée(s)")
        return self.errors

    def value(self, node: Node, color_by: str) -> float:
        if color_by == "spot":
            return node.spot_price
        if color_by == "probability":
            return node.proba
        if node.option_value is None:
            raise ValueError("Appeler tree.price() avant de colorer les valeurs de l'option.")
        return node.option_value

    @staticmethod
    def validate_color(color_by: str) -> None:
        if color_by not in ("spot", "option", "probability"):
            raise ValueError("color_by doit être 'spot', 'option' ou 'probability'.")

    def color_scale(self, color_by: str) -> Normalize:
        minimum, maximum = math.inf, -math.inf
        for trunk in self.columns():
            for node in self.column_nodes(trunk):
                value = self.value(node, color_by)
                minimum, maximum = min(minimum, value), max(maximum, value)
        return Normalize(minimum, maximum if maximum > minimum else minimum + 1)

    def decision(self, node: Node) -> str:
        if node.step == self.tree.model.nb_steps:
            return "MATURITÉ : EXERCER" if self.tree.option.intrinsic_value(node.spot_price) > 0 else "EXPIRATION SANS VALEUR"
        action = "EXERCER" if getattr(node, "exercise_now", False) else "CONSERVER"
        return action + (" · PRUNING" if self.is_pruned(node) else "")

    def _prepare_drawing(self, color_by: str, max_nodes: int, run_checks: bool) -> int:
        self.validate_color(color_by)
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
            raise ValueError("max_nodes doit être un entier strictement positif.")
        # Compter par les liens validés, sans stocker les nœuds dans une liste.
        count = sum(1 for trunk in self.columns() for _ in self.column_nodes(trunk))
        if count > max_nodes:
            raise ValueError(
                f"Dessin refusé : {count} nœuds, limite explicite {max_nodes}. "
                "Réduire l'arbre ou augmenter max_nodes ; check() n'a pas cette limite."
            )
        if self.tree.root.option_value is None:
            self.tree.price()
        if run_checks and self.check() != 0:
            raise ValueError("Dessin interrompu : corriger les erreurs signalées puis relancer.")
        return count

    def export_svg(
        self, path: str | Path, color_by: str = "option", show_neighbors: bool = False,
        *, max_nodes: int = DEFAULT_MAX_DRAW_NODES, run_checks: bool = True,
    ) -> Path:
        """Écrire un SVG autonome, vectoriel et zoomable, et renvoyer son chemin.

        L'axe vertical est logarithmique : deux voisins de rapport alpha ont
        le même espacement. Les spots S restent affichés en euros. Chaque boîte
        contient S, V et π (probabilité risque neutre d'atteindre ce nœud,
        sans arrêter les trajectoires déjà exercées). Les probabilités p des
        flèches sont conditionnelles au nœud de départ. π et p diffèrent donc.

        Le pruning apparaît en orange, l'exercice anticipé en rouge, et le
        tronc avec un double contour. Au survol, un titre donne les moments
        locaux, y compris le biais causé par une coupure.
        """
        count = self._prepare_drawing(color_by, max_nodes, run_checks)
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        norm = self.color_scale(color_by)
        min_log, max_log, min_gap = math.inf, -math.inf, math.inf
        for trunk in self.columns():
            for node in self.column_nodes(trunk):
                log_spot = math.log(node.spot_price)
                min_log, max_log = min(min_log, log_spot), max(max_log, log_spot)
                if node.below is not None:
                    min_gap = min(min_gap, log_spot - math.log(node.below.spot_price))
        if min_gap <= 0:
            raise ValueError("Le dessin nécessite des spots strictement ordonnés et positifs.")
        # Au moins 92 pixels entre les centres de deux voisins de colonne.
        # En volatilité nulle il n'y a qu'un nœud par date : échelle de secours.
        scale_y = 92 / min_gap if math.isfinite(min_gap) else 300 / max(max_log - min_log, 1e-6)
        box_width, box_height = 174.0, 76.0
        margin_x, top_y, step_width = 130.0, 230.0, 270.0
        width = 2 * margin_x + self.tree.model.nb_steps * step_width
        height = max(580.0, top_y + (max_log - min_log) * scale_y + 180)
        if max(width, height) > 100_000:
            raise ValueError("Le canevas dépasserait 100 000 pixels : réduire l'arbre pour le dessin.")

        def x(node: Node) -> float:
            return margin_x + node.step * step_width

        def y(node: Node) -> float:
            return top_y + (max_log - math.log(node.spot_price)) * scale_y

        def write_text(svg: TextIO, px: float, py: float, label: str, css: str = "", anchor: str = "middle") -> None:
            svg.write(f'<text x="{px:.2f}" y="{py:.2f}" text-anchor="{anchor}" class="{css}">{escape(label)}</text>\n')

        with destination.open("w", encoding="utf-8") as svg:
            svg.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            svg.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.2f} {height:.2f}" role="img" aria-labelledby="title description">\n')
            svg.write('<title id="title">Arbre de valorisation : construction, exercice et pruning</title>\n')
            svg.write('<desc id="description">Dessin vectoriel. Axe vertical logarithmique. S : prix du sous-jacent ; V : valeur de l’option ; π : probabilité du nœud ; p : probabilité conditionnelle de branche.</desc>\n')
            svg.write('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>\n')
            svg.write('<style>text{font-family:DejaVu Sans,Arial,sans-serif;font-size:12px;fill:#172033}.heading{font-size:23px;font-weight:700}.subheading{font-size:14px}.small{font-size:11px}.decision{font-size:10px;font-weight:700}.edge-label{font-size:10px;paint-order:stroke;stroke:#fff;stroke-width:4px;stroke-linejoin:round}.forward{fill:none;stroke:#0284c7;stroke-width:2;stroke-dasharray:7 5}</style>\n')
            svg.write(f'<rect width="{width:.2f}" height="{height:.2f}" fill="#f8fafc"/>\n')
            title = f"{self.tree.option.type} {self.tree.option.exercise_type} · K={self.tree.option.strike:g} · {self.tree.model.nb_steps} pas · {count} nœuds"
            write_text(svg, 36, 38, title, "heading", "start")
            write_text(svg, 36, 65, "S : sous-jacent (€) · V : option (€) · π : probabilité du nœud · p : probabilité conditionnelle de branche", "subheading", "start")
            write_text(svg, 36, 89, "Contours : rouge = exercice anticipé ; orange = pruning ; double = tronc. Flèche orange p=1 : masse redirigée vers l’intérieur.", "small", "start")
            write_text(svg, 36, 111, "π ne tient pas compte des exercices antérieurs. Axe vertical : log(S). Pointillés bleus : forward théorique sans biais de pruning.", "small", "start")
            write_text(svg, 36, 133, f"r={self.tree.market.rate:.2%} · σ={self.tree.market.vol:.2%} · q={self.tree.market.dividend_yield:.2%} · dividende cash={self.tree.market.dividend_amount:g} € · seuil pruning={self.tree.model.pruning_threshold:g}", "small", "start")
            # Colonnes et dates. Aucune collection de nœuds n'est nécessaire.
            for trunk in self.columns():
                px = x(trunk)
                svg.write(f'<line x1="{px:.2f}" y1="192" x2="{px:.2f}" y2="{height - 90:.2f}" stroke="#e2e8f0"/>\n')
                write_text(svg, px, 164, f"t{trunk.step} = {trunk.step * self.tree.deltaT:.3f} an")
                column_mass = sum(node.proba for node in self.column_nodes(trunk))
                write_text(svg, px, 183, f"{trunk.column_size} nœuds · Σπ={column_mass:.6f}", "small")
                if trunk.step == self.tree.dividend_step:
                    svg.write(f'<line x1="{px:.2f}" y1="192" x2="{px:.2f}" y2="{height - 90:.2f}" stroke="#dc2626" stroke-dasharray="5 5" opacity="0.45"/>\n')
                    write_text(svg, px, 205, "FIN DU PAS DIVIDENDE CASH", "decision")
            # La courbe théorique sert de repère. Son écart à la moyenne réelle
            # du graphe n'est pas nul lorsque le pruning altère les moments.
            previous_x = previous_y = None
            for trunk in self.columns():
                theoretical = self.tree.mean_at_time(trunk.step * self.tree.deltaT)
                if theoretical > 0:
                    px = x(trunk)
                    py = top_y + (max_log - math.log(theoretical)) * scale_y
                    if previous_x is not None:
                        svg.write(f'<path class="forward" d="M {previous_x:.2f} {previous_y:.2f} L {px:.2f} {py:.2f}"/>\n')
                    previous_x, previous_y = px, py
            # Toutes les flèches sont placées AVANT les boîtes, afin que celles-ci
            # masquent proprement les traits derrière leurs annotations.
            for trunk in self.columns():
                for node in self.column_nodes(trunk):
                    for branch in BRANCHES:
                        child = getattr(node, "next_" + branch)
                        if child is None:
                            continue
                        probability = getattr(node, "p_" + branch)
                        pruned = self.is_pruned(node)
                        stroke = "#ea580c" if pruned else "#94a3b8"
                        thickness = 2.8 if pruned else 1.1
                        dash = ' stroke-dasharray="5 3"' if pruned or probability == 0 else ""
                        x1, x2 = x(node) + box_width / 2, x(child) - box_width / 2
                        y1, y2 = y(node), y(child)
                        svg.write(f'<path d="M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}" fill="none" stroke="{stroke}" stroke-width="{thickness}" marker-end="url(#arrow)"{dash}/>\n')
                        label = "p=1" if self.close(probability, 1) else f"p={probability:.3g}"
                        write_text(svg, (x1 + x2) / 2, (y1 + y2) / 2 - 5, label, "edge-label")
                    if show_neighbors and node.below is not None:
                        svg.write(f'<path d="M {x(node):.2f} {y(node):.2f} L {x(node):.2f} {y(node.below):.2f}" stroke="#64748b" stroke-dasharray="2 4" opacity="0.45"/>\n')
            for trunk in self.columns():
                for node in self.column_nodes(trunk):
                    px, py = x(node), y(node)
                    value = self.value(node, color_by)
                    fill = to_hex(plt.get_cmap("viridis")(norm(value)))
                    text_color = "#ffffff" if norm(value) < 0.6 else "#111827"
                    exercising = node.step < self.tree.model.nb_steps and getattr(node, "exercise_now", False)
                    border = "#dc2626" if exercising else "#ea580c" if self.is_pruned(node) else "#475569"
                    svg.write(f'<g id="node-{node.step}-{id(node)}">\n')
                    detail = f"Pas {node.step} ; S={node.spot_price:.12g} ; V={node.option_value:.12g} ; π={node.proba:.12g} ; {self.decision(node)}"
                    detail += f" ; payoff={self.tree.option.intrinsic_value(node.spot_price):.12g}"
                    continuation = getattr(node, "continuation_value", None)
                    detail += f" ; continuation={continuation:.12g}" if continuation is not None else " ; continuation=sans objet à maturité"
                    if node.step < self.tree.model.nb_steps:
                        mean, variance = self.local_moments(node)
                        detail += f" ; moyenne={mean:.12g} ; Δm={mean-node.forward():+.6g} ; variance={variance:.12g} ; Δvar={variance-node.variance():+.6g}"
                    svg.write(f'<title>{escape(detail)}</title>\n')
                    svg.write(f'<rect x="{px-box_width/2:.2f}" y="{py-box_height/2:.2f}" width="{box_width}" height="{box_height}" rx="9" fill="{fill}" stroke="{border}" stroke-width="{3 if exercising or self.is_pruned(node) else 1.2}"/>\n')
                    if isinstance(node, TrunkNode):
                        svg.write(f'<rect x="{px-box_width/2+4:.2f}" y="{py-box_height/2+4:.2f}" width="{box_width-8}" height="{box_height-8}" rx="6" fill="none" stroke="{text_color}" stroke-opacity="0.65"/>\n')
                    svg.write(f'<g style="fill:{text_color}">\n')
                    # Un style explicite sur chaque texte prévaut sur la règle
                    # générale CSS « text{fill:...} » du document.
                    for offset, label, size in (
                        (-18, f"S = {node.spot_price:.4f}", 12),
                        (-2, f"V = {node.option_value:.5f}", 12),
                        (14, f"π = {node.proba:.5g}", 11),
                        (29, self.decision(node), 9),
                    ):
                        svg.write(f'<text x="{px:.2f}" y="{py+offset:.2f}" text-anchor="middle" style="fill:{text_color};font-size:{size}px">{escape(label)}</text>\n')
                    svg.write('</g></g>\n')
            color_name = {"spot": "S", "option": "V", "probability": "π"}[color_by]
            write_text(svg, 36, height - 50, f"Couleur : {color_name}, minimum {norm.vmin:.6g} → maximum {norm.vmax:.6g}. Zoomer dans le navigateur pour lire les détails ; survoler un nœud pour ses moments.", "small", "start")
            svg.write('</svg>\n')
        print("SVG enregistré :", destination)
        return destination

    def draw_edges(self, ax: Axes, node: Node, show_neighbors: bool) -> None:
        for branch in BRANCHES:
            child = getattr(node, "next_" + branch)
            if child is not None:
                pruned = self.is_pruned(node)
                ax.plot((node.step, child.step), (node.spot_price, child.spot_price),
                        color="#ea580c" if pruned else "#94a3b8",
                        linewidth=2 if pruned else 0.7, zorder=1)
                if pruned:
                    ax.text(node.step + 0.5, math.sqrt(node.spot_price * child.spot_price),
                            "p=1", fontsize=6, color="#c2410c", zorder=4)
        if show_neighbors and node.below is not None:
            ax.plot((node.step, node.below.step), (node.spot_price, node.below.spot_price),
                    ":", color="gray", linewidth=0.6, zorder=0)

    def draw_node(self, ax: Axes, node: Node, norm: Normalize, color_by: str) -> None:
        value = self.value(node, color_by)
        color = plt.get_cmap("viridis")(norm(value))
        exercising = node.step < self.tree.model.nb_steps and getattr(node, "exercise_now", False)
        edge = "#dc2626" if exercising else "#ea580c" if self.is_pruned(node) else "#f8fafc"
        size = 48 if self.tree.model.nb_steps <= 6 else 31
        ax.plot(node.step, node.spot_price, "o", markersize=size,
                markerfacecolor=color, markeredgecolor=edge, markeredgewidth=2, zorder=3)
        label = f"S {node.spot_price:.2f}\nV {node.option_value:.2f}\nπ {node.proba:.2g}"
        ax.text(node.step, node.spot_price, label, ha="center", va="center",
                fontsize=7 if size == 48 else 5,
                color="white" if norm(value) < 0.6 else "black", zorder=4)

    def draw_forward(self, ax: Axes) -> None:
        previous = self.tree.market.spot
        for step in range(1, self.tree.model.nb_steps + 1):
            forward = self.tree.mean_at_time(step * self.tree.deltaT)
            label = "Forward théorique (sans biais de pruning)" if step == 1 else "_nolegend_"
            ax.plot((step - 1, step), (previous, forward), "--",
                    color="#0284c7", label=label, zorder=2)
            if step == self.tree.dividend_step:
                ax.axvline(step, color="#dc2626", linestyle=":", alpha=0.5,
                           label="Fin du pas contenant le détachement")
            previous = forward
        ax.legend(loc="upper left", fontsize=8)

    def decorate(self, ax: Axes, norm: Normalize, color_by: str) -> None:
        self.draw_forward(ax)
        ax.set(title="Arbre : S sous-jacent ; V option ; π probabilité du nœud\nRouge : exercice anticipé ; orange : pruning (p=1)",
               xlabel="Pas de temps", ylabel="Prix du sous-jacent (échelle logarithmique)",
               xticks=range(self.tree.model.nb_steps + 1), yscale="log")
        ax.margins(x=0.05, y=0.12)
        ax.grid(axis="x", alpha=0.15)
        label = {"option": "Valeur de l'option", "spot": "Prix du sous-jacent", "probability": "Probabilité du nœud"}[color_by]
        ax.figure.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label=label)

    def show(
        self, color_by: str = "option", show_neighbors: bool = False,
        *, max_nodes: int = DEFAULT_MAX_DRAW_NODES,
    ) -> None:
        """Compatibilité avec l'ancien appel : SVG + aperçu PNG et fenêtre Matplotlib."""
        self.export_svg(self.output / "debug_arbre.svg", color_by, show_neighbors, max_nodes=max_nodes)
        norm = self.color_scale(color_by)
        fig, ax = plt.subplots(figsize=(max(15, min(36, self.tree.model.nb_steps * 1.4)), 11), layout="constrained")
        for trunk in self.columns():
            for node in self.column_nodes(trunk):
                self.draw_edges(ax, node, show_neighbors)
                self.draw_node(ax, node, norm, color_by)
        self.decorate(ax, norm, color_by)
        destination = self.output / "debug_arbre.png"
        fig.savefig(destination, dpi=160)
        print("Aperçu PNG enregistré :", destination)
        plt.show()
        plt.close(fig)


def main() -> None:
    """Exemple reproductible ; --no-show fonctionne aussi sur un serveur sans écran."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20, help="Nombre de pas (défaut : 20).")
    parser.add_argument("--pruning-threshold", type=float, default=1e-3,
                        help="Seuil volontairement visible pour l'exemple ; 0 désactive le pruning.")
    parser.add_argument("--output", type=Path, default=Path("debug_arbre.svg"), help="Fichier SVG de destination.")
    parser.add_argument("--color-by", choices=("spot", "option", "probability"), default="option")
    parser.add_argument("--show-neighbors", action="store_true", help="Dessiner aussi les liens verticaux.")
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_DRAW_NODES)
    parser.add_argument("--no-show", action="store_true", help="Exporter seulement le SVG, sans fenêtre graphique.")
    args = parser.parse_args()
    model = Model(pricing_date=dt.date(2026, 9, 15), nb_steps=args.steps,
                  pruning_threshold=args.pruning_threshold)
    market = Market(spot=100, rate=0.03, vol=0.20,
                    dividend_yield=0.02, dividend_amount=3.0,
                    ex_dividend_date=dt.date(2027, 3, 15))
    option = Option(strike=101, maturity=dt.date(2027, 9, 2), type="Put", exercise_type="American")
    tree = Tree(model=model, option=option, market=market).build()
    tree.price()
    debugger = DebugTree(tree)
    debugger.export_svg(args.output, args.color_by, args.show_neighbors, max_nodes=args.max_nodes)
    if not args.no_show:
        # L'API show() reste disponible pour un usage interactif dans un IDE.
        # Ici on dessine directement pour ne pas faire deux fois les contrôles.
        norm = debugger.color_scale(args.color_by)
        fig, ax = plt.subplots(figsize=(max(15, min(36, args.steps * 1.4)), 11), layout="constrained")
        for trunk in debugger.columns():
            for node in debugger.column_nodes(trunk):
                debugger.draw_edges(ax, node, args.show_neighbors)
                debugger.draw_node(ax, node, norm, args.color_by)
        debugger.decorate(ax, norm, args.color_by)
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
