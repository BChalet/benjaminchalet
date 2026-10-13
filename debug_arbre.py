from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from arbre_binomial import Market, Model, Node, Option, Tree, TrunkNode


class DebugTree:
    """Contrôles des liens et des moments, sans collection de nœuds.

    Les trois curseurs par colonne suivent les enfants et les parents dans
    l'ordre décroissant des spots. Chaque lien est parcouru un nombre borné
    de fois : le contrôle reste linéaire en nombre de nœuds/branches.
    La variance contrôlée à la date cash est celle de l'approximation du cours,
    et non celle d'un modèle lognormal exact après détachement.
    """

    def __init__(self, tree: Tree) -> None:
        self.tree = tree
        self.errors = 0
        self.node_count = 0
        self.branch_count = 0
        self.output = Path(__file__).resolve().parent
        self._child_up: Node | None = None
        self._child_mid: Node | None = None
        self._child_down: Node | None = None
        self._parent_up: Node | None = None
        self._parent_mid: Node | None = None
        self._parent_down: Node | None = None

    def close(self, actual: float, expected: float) -> bool:
        return bool(np.isclose(actual, expected, rtol=1e-9, atol=1e-10))

    def error(self, message: str, node: Node | None = None) -> None:
        self.errors += 1
        location = "" if node is None else f"Pas {node.step}, S={node.spot_price:.6f} : "
        print("ERREUR :", location + message)

    def columns(self) -> Iterator[TrunkNode]:
        trunk = self.tree.root
        previous = None
        for step in range(self.tree.model.nb_steps + 1):
            if not isinstance(trunk, TrunkNode) or trunk.step != step:
                raise ValueError(f"Tronc absent ou mauvaise date au pas {step}")
            if trunk.previous_trunk is not previous:
                raise ValueError(f"Lien précédent du tronc incorrect au pas {step}")
            yield trunk
            previous = trunk
            trunk = trunk.next_trunk
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

    def in_column(self, child: Node, trunk: TrunkNode, branch: str = "up") -> bool:
        """Recherche monotone : appeler dans l'ordre des parents de haut en bas."""
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
                if node.next_down is not node.below.next_up:
                    self.error("recombinaison binomiale incorrecte", node)

    def count_incoming(self, node: Node, previous: TrunkNode, branch: str) -> int:
        attribute = "_parent_" + branch
        cursor = getattr(self, attribute)
        count = 0
        moves = 0
        while cursor is not None:
            child = getattr(cursor, "next_" + branch)
            if child is not None and child.spot_price < node.spot_price:
                break
            if child is node:
                count += 1
            cursor = cursor.below
            moves += 1
            if moves > previous.column_size:
                raise ValueError("Cycle pendant la recherche des parents")
        setattr(self, attribute, cursor)
        return count

    def check_parents(self, node: Node, previous: TrunkNode | None) -> None:
        if previous is None:
            if node is not self.tree.root:
                self.error("nœud supplémentaire à la date initiale", node)
            return
        count = self.count_incoming(node, previous, "up")
        count += self.count_incoming(node, previous, "mid")
        count += self.count_incoming(node, previous, "down")
        kind = self.tree.transition_type(previous.step)
        if count == 0:
            self.error("nœud sans branche entrante", node)
        elif kind == "binomial":
            expected = 1 if node.above is None or node.below is None else 2
            if count != expected:
                self.error(f"{count} branches entrantes au lieu de {expected}", node)
        elif kind == "deterministic" and count != 1:
            self.error(f"{count} branches entrantes dans une transition déterministe", node)

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
            up = self.tree.p_up
            down = self.tree.p_down
            if not (0 <= up <= 1 and 0 <= down <= 1) or not self.close(up + down, 1):
                self.error("probabilités binomiales communes incorrectes")

    def check_moments(self, node: Node) -> None:
        mean = 0.0
        probability_sum = 0.0
        for branch in ("up", "mid", "down"):
            probability = getattr(node, "p_" + branch)
            child = getattr(node, "next_" + branch)
            probability_sum += probability
            if not np.isfinite(probability) or not -1e-12 <= probability <= 1 + 1e-12:
                self.error(f"probabilité {branch} en dehors de [0,1]", node)
            if child is not None:
                mean += probability * child.spot_price
        if not self.close(probability_sum, 1):
            self.error("la somme des probabilités locales ne vaut pas 1", node)
        variance = 0.0
        for branch in ("up", "mid", "down"):
            child = getattr(node, "next_" + branch)
            if child is not None:
                variance += getattr(node, "p_" + branch) * (child.spot_price - mean) ** 2
        if not self.close(mean, node.forward()):
            self.error("espérance différente du forward après dividendes", node)
        if not self.close(variance, node.variance()):
            self.error("variance différente de la variance prévue par le modèle", node)

        kind = self.tree.transition_type(node.step)
        if kind == "binomial":
            if node.next_mid is not None:
                self.error("branche centrale dans une transition binomiale", node)
            if node.next_up is None or node.next_down is None:
                self.error("deux enfants requis dans une transition binomiale", node)
            else:
                if not self.close(node.next_up.spot_price, node.forward() * self.tree.sqrtAlpha):
                    self.error("prix de l'enfant haut binomial incorrect", node)
                if not self.close(node.next_down.spot_price, node.forward() / self.tree.sqrtAlpha):
                    self.error("prix de l'enfant bas binomial incorrect", node)
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

    def check_column(self, trunk: TrunkNode) -> None:
        if trunk.next_trunk is not None:
            first = self.top(trunk.next_trunk)
            self._child_up = self._child_mid = self._child_down = first
        if trunk.previous_trunk is not None:
            first = self.top(trunk.previous_trunk)
            self._parent_up = self._parent_mid = self._parent_down = first
        for node in self.column_nodes(trunk):
            self.node_count += 1
            if node.tree is not self.tree:
                self.error("référence vers un autre arbre", node)
            self.check_neighbors(node)
            self.check_parents(node, trunk.previous_trunk)
            if node.step < self.tree.model.nb_steps:
                if not isinstance(trunk.next_trunk, TrunkNode):
                    raise ValueError("Colonne suivante absente")
                self.check_child(node, node.next_up, trunk.next_trunk, "up")
                self.check_child(node, node.next_mid, trunk.next_trunk, "mid")
                self.check_child(node, node.next_down, trunk.next_trunk, "down")
                self.check_moments(node)
            elif node.next_up is not None or node.next_mid is not None or node.next_down is not None:
                self.error("branche après maturité", node)

    def check(self) -> int:
        self.errors = self.node_count = self.branch_count = 0
        try:
            self.check_parameters()
            for trunk in self.columns():
                self.check_column(trunk)
        except (ValueError, AttributeError, ZeroDivisionError) as exc:
            self.error(str(exc))
        print(f"DEBUG : {self.node_count} nœuds parcourus, {self.branch_count} branches")
        if self.errors == 0:
            self.check_zero_strike()
        print(f"Bilan : {self.errors} erreur(s) détectée(s)")
        return self.errors

    def check_zero_strike(self) -> None:
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

    def value(self, node: Node, color_by: str) -> float:
        return node.spot_price if color_by == "spot" else node.option_value

    def color_scale(self, color_by: str) -> Normalize:
        minimum = float("inf")
        maximum = -float("inf")
        for trunk in self.columns():
            for node in self.column_nodes(trunk):
                value = self.value(node, color_by)
                minimum = min(minimum, value)
                maximum = max(maximum, value)
        return Normalize(minimum, maximum if maximum > minimum else minimum + 1)

    def draw_edges(self, ax: Axes, node: Node, show_neighbors: bool) -> None:
        for branch in ("up", "mid", "down"):
            child = getattr(node, "next_" + branch)
            if child is not None:
                ax.plot((node.step, child.step), (node.spot_price, child.spot_price),
                        color="#94a3b8", linewidth=1, zorder=1)
        if show_neighbors and node.below is not None:
            ax.plot((node.step, node.below.step), (node.spot_price, node.below.spot_price),
                    ":", color="gray", zorder=0)

    def draw_node(self, ax: Axes, node: Node, norm: Normalize, color_by: str) -> None:
        value = self.value(node, color_by)
        color = plt.get_cmap("viridis")(norm(value))
        size = 46 if self.tree.model.nb_steps <= 6 else 30
        ax.plot(node.step, node.spot_price, "o", markersize=size,
                markerfacecolor=color, markeredgecolor="white", zorder=3)
        label = f"S {node.spot_price:.2f}\nV {node.option_value:.2f}"
        ax.text(node.step, node.spot_price, label, ha="center", va="center",
                fontsize=8 if size == 46 else 6,
                color="white" if norm(value) < 0.55 else "black", zorder=4)

    def draw_forward(self, ax: Axes) -> None:
        previous = self.tree.market.spot
        for step in range(1, self.tree.model.nb_steps + 1):
            forward = self.tree.mean_at_time(step * self.tree.deltaT)
            label = "Espérance risque neutre E[S(t)]" if step == 1 else "_nolegend_"
            ax.plot((step - 1, step), (previous, forward), "--",
                    color="#ea580c", label=label, zorder=2)
            if self.tree.transition_type(step - 1) == "trinomial":
                ax.axvline(step, color="#dc2626", linestyle=":", alpha=0.5,
                           label="Fin du pas contenant le détachement")
            previous = forward
        ax.legend(loc="upper left")

    def decorate(self, ax: Axes, norm: Normalize, color_by: str) -> None:
        self.draw_forward(ax)
        ax.set(title="Arbre binomial avec dividendes\nS : sous-jacent ; V : option",
               xlabel="Pas de temps", ylabel="Prix du sous-jacent",
               xticks=range(self.tree.model.nb_steps + 1))
        ax.margins(x=0.08, y=0.12)
        ax.grid(axis="x", alpha=0.15)
        label = "Valeur de l'option" if color_by == "option" else "Prix du sous-jacent"
        ax.figure.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label=label)

    def show(self, color_by: str = "option", show_neighbors: bool = False) -> None:
        if not 1 <= self.tree.model.nb_steps <= 10:
            raise ValueError("Pour le dessin : 1 à 10 pas. Sinon utiliser check().")
        if color_by not in ("spot", "option"):
            raise ValueError("color_by doit être 'spot' ou 'option'.")
        if self.check() != 0:
            print("Dessin interrompu : corriger les erreurs signalées puis relancer.")
            return
        norm = self.color_scale(color_by)
        fig, ax = plt.subplots(figsize=(15, 9), layout="constrained")
        for trunk in self.columns():
            for node in self.column_nodes(trunk):
                self.draw_edges(ax, node, show_neighbors)
                self.draw_node(ax, node, norm, color_by)
        self.decorate(ax, norm, color_by)
        destination = self.output / "debug_arbre.png"
        fig.savefig(destination, dpi=160)
        print("Figure enregistrée :", destination)
        plt.show()


def main() -> None:
    pricing_date = dt.date(2026, 9, 15)
    maturity = dt.date(2027, 9, 2)
    model = Model(pricing_date=pricing_date, nb_steps=6)
    market = Market(spot=100, rate=0.03, vol=0.20,
                    dividend_yield=0.02, dividend_amount=3.0,
                    ex_dividend_date=dt.date(2027, 3, 15))
    option = Option(strike=101, maturity=maturity, type="Call", exercise_type="European")
    tree = Tree(model=model, option=option, market=market)
    tree.build()
    tree.price()
    debugger = DebugTree(tree)
    debugger.show(color_by="option", show_neighbors=True)
    # Pour un grand arbre : debugger.check(). Aucun pruning n'est supposé ici.


if __name__ == "__main__":
    main()
