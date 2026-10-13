"""Arbre binomial avec rendement q et pas trinomial au détachement cash.

Adaptation du code fourni, avec les moments du cours Dauphine (pages 5/10).
Les nœuds sont accessibles uniquement par leurs liens ; aucun conteneur de
nœuds n'est utilisé. Un seul montant cash et une seule ex-date sont modélisés.
"""

from __future__ import annotations

import numpy as np
import datetime as dt
from typing import Callable, Iterator

# MODEL

class Model:

    def __init__(self, pricing_date: dt.date, nb_steps: int) -> None:
        if not isinstance(pricing_date, dt.date) or isinstance(pricing_date, dt.datetime):
            raise ValueError("pricing_date doit être une date (datetime.date).")
        if isinstance(nb_steps, bool) or not isinstance(nb_steps, int) or nb_steps < 1:
            raise ValueError("nb_steps doit être un entier strictement positif.")
        self.pricing_date = pricing_date
        self.nb_steps = nb_steps

# OPTION

class Option:

    def __init__(
        self,
        strike: float,
        maturity: dt.date,
        type: str,
        exercise_type: str
    ) -> None:
        if not np.isfinite(strike) or strike < 0:
            raise ValueError("Le strike doit être fini et positif ou nul.")
        if not isinstance(maturity, dt.date) or isinstance(maturity, dt.datetime):
            raise ValueError("maturity doit être une date (datetime.date).")
        if type.lower() not in ("call", "put"):
            raise ValueError("Option type must be 'Call' or 'Put'")
        if exercise_type.lower() not in ("european", "american"):
            raise ValueError("Exercise type must be 'European' or 'American'")
        self.strike = strike
        self.maturity = maturity
        self.type = type
        self.exercise_type = exercise_type

    # Valeur d'exercice du call ou du put pour un prix du sous-jacent.
    def intrinsic_value(self,spot_price: float) -> float:
        K = self.strike
        if self.type.lower() == "call":
            return max(spot_price - K, 0.0)

        elif self.type.lower() == "put":
            return max(K - spot_price,0.0)

        else:
            raise ValueError("Option type must be 'Call' or 'Put'")

# MARKET

class Market:

    def __init__(
        self,
        spot: float,
        rate: float,
        vol: float,
        dividend_yield: float = 0.0,
        dividend_amount: float = 0.0,
        ex_dividend_date: dt.date | None = None
    ) -> None:
        if not all(np.isfinite(x) for x in (spot, rate, vol, dividend_yield, dividend_amount)):
            raise ValueError("Les paramètres de marché doivent être finis.")
        if spot <= 0 or vol < 0 or dividend_yield < 0 or dividend_amount < 0:
            raise ValueError("Il faut spot > 0, vol >= 0, q >= 0 et dividende cash >= 0.")
        if dividend_amount > 0 and ex_dividend_date is None:
            raise ValueError("Un dividende cash positif nécessite sa date de détachement.")
        if ex_dividend_date is not None and (
            not isinstance(ex_dividend_date, dt.date) or isinstance(ex_dividend_date, dt.datetime)
        ):
            raise ValueError("ex_dividend_date doit être une date (datetime.date).")
        self.spot = spot
        self.rate = rate
        self.vol = vol
        # q est un rendement annuel (0.02 = 2 %), D un montant en euros.
        self.dividend_yield = dividend_yield
        self.dividend_amount = dividend_amount
        self.ex_dividend_date = ex_dividend_date

class Node:
    # Un nœud conserve son prix et ses liens vers les autres nœuds.

    def __init__(self,tree: Tree,spot_price: float,step: int) -> None:
        self.tree = tree
        self.spot_price = spot_price
        self.step = step
        self.next_up = None
        self.next_down = None
        self.next_mid = None
        self.p_up = tree.p_up
        self.p_down = tree.p_down
        self.p_mid = 0.0
        self.above = None
        self.below = None
        self.option_value = None

    # Espérance conditionnelle sur le prochain pas, dividendes inclus.
    def forward(self) -> float:
        return self.tree.expected_spot(self.spot_price, self.step)

    def variance(self) -> float:
        return self.tree.conditional_variance(self.spot_price, self.step)

    def compute_option_value(self) -> None:
        # European continuation value
        expectation = 0.0
        if self.next_up is not None:
            expectation += self.p_up * self.next_up.option_value
        if self.next_mid is not None:
            expectation += self.p_mid * self.next_mid.option_value
        if self.next_down is not None:
            expectation += self.p_down * self.next_down.option_value
        hold_value = self.tree.df * expectation

        # European
        if self.tree.option.exercise_type.lower() == "european":
            self.option_value = hold_value

        # American
        elif self.tree.option.exercise_type.lower() == "american":
            exercise_value = self.tree.option.intrinsic_value(self.spot_price)
            self.option_value = max(hold_value,exercise_value)

        else:
            raise ValueError("Exercise type must be 'European' or 'American'")

# TRUNK NODE

class TrunkNode(Node):
    def __init__(self,tree: Tree,spot_price: float,step: int) -> None:
        # Run Node.__init__()
        super().__init__(tree,spot_price,step)

        # Additional links specific to trunk
        self.previous_trunk = None
        self.next_trunk = None
        self.column_size = 1

# TREE

class Tree:

    def __init__(self,model: Model,option: Option,market: Market) -> None:
        self.model = model
        self.option = option
        self.market = market

        self.duration_days = (option.maturity - model.pricing_date).days
        if self.duration_days <= 0:
            raise ValueError("La maturité doit être postérieure à la date de calcul.")
        self.maturity_years = self.duration_days / 365.0
        self.drift = market.rate - market.dividend_yield

        # Un seul dividende cash : affectation entière à (t_i, t_{i+1}].
        # Ce calcul évite les erreurs d'arrondi de dates signalées dans le cours.
        self.dividend_step = None
        self.dividend_time = None
        if market.dividend_amount > 0 and model.pricing_date < market.ex_dividend_date <= option.maturity:
            ex_days = (market.ex_dividend_date - model.pricing_date).days
            self.dividend_time = ex_days / 365.0
            self.dividend_step = (ex_days * model.nb_steps + self.duration_days - 1) // self.duration_days
            self.remaining_after_dividend = (
                self.dividend_step * self.duration_days - ex_days * model.nb_steps
            ) / (365.0 * model.nb_steps)

        # TIME STEP
        maturity_years = (option.maturity- model.pricing_date).days / 365.0
        self.deltaT = (maturity_years/ model.nb_steps)
        #discount factor
        self.df = np.exp(-market.rate * self.deltaT)
        self.growth = np.exp(self.drift * self.deltaT)
        self.variance_factor = np.expm1(market.vol**2 * self.deltaT)

        # TREE PARAMETERS
        self.sqrtAlpha = self.compute_sqrt_alpha()
        # Ratio entre deux voisins d'une colonne, y compris le pas trinomial.
        self.alpha = self.sqrtAlpha**2
        if market.vol > 0 and (not np.isfinite(self.alpha) or self.alpha <= 1):
            raise ValueError("La volatilité et le pas de temps produisent un maillage non représentable.")
        self.p_up = (1/(1 + self.sqrtAlpha))
        self.p_down = (self.sqrtAlpha/(1 + self.sqrtAlpha))

        # ROOT
        self.root = TrunkNode(tree=self,spot_price=market.spot,step=0)
        self.last_trunk = self.root
        self._built = False

    # Compute sqrt(alpha)
    def compute_sqrt_alpha(self) -> float:
        sigma = self.market.vol
        a = 1 + self.variance_factor / 2
        # Cette formule donne directement sqrt(alpha).
        return a + np.sqrt(self.variance_factor * (1 + self.variance_factor / 4))

    def transition_type(self, step: int) -> str:
        if step >= self.model.nb_steps:
            return "terminal"
        if self.market.vol == 0:
            return "deterministic"
        return "trinomial" if step + 1 == self.dividend_step else "binomial"

    def expected_spot(self, spot: float, step: int) -> float:
        mean = spot * self.growth
        if step + 1 == self.dividend_step:
            mean -= self.market.dividend_amount * np.exp(self.drift * self.remaining_after_dividend)
        return float(mean)

    def conditional_variance(self, spot: float, step: int) -> float:
        base = spot
        if step + 1 == self.dividend_step:
            # Approximation intra-pas du cours, pages 5 et 10 : w = (t_fin-t_ex)/dt.
            # Elle ne constitue pas la variance exacte d'un GBM avec saut cash intérieur.
            base -= self.market.dividend_amount * self.remaining_after_dividend / self.deltaT
        return float((base * self.growth)**2 * self.variance_factor)

    def mean_at_time(self, time: float) -> float:
        mean = self.market.spot * np.exp(self.drift * time)
        if self.dividend_time is not None and time >= self.dividend_time - 1e-14:
            mean -= self.market.dividend_amount * np.exp(self.drift * (time - self.dividend_time))
        return float(mean)

    def mean_at_maturity(self) -> float:
        return self.mean_at_time(self.maturity_years)

    def expected_zero_strike(self) -> float:
        return float(np.exp(-self.market.rate * self.maturity_years) * self.mean_at_maturity())

    def iter_columns(self) -> Iterator[TrunkNode]:
        trunk = self.root
        while trunk is not None:
            yield trunk
            trunk = trunk.next_trunk

    def iter_column(self, trunk: TrunkNode) -> Iterator[Node]:
        node = trunk
        while node.above is not None:
            node = node.above
        while node is not None:
            yield node
            node = node.below

    def positive_mean(self, node: Node) -> float:
        mean = node.forward()
        if mean <= 0 or not np.isfinite(mean):
            raise ValueError(
                f"Pas {node.step} -> {node.step+1}, S={node.spot_price:.10g} : "
                "espérance après dividende non positive. Le maillage cash positif ne peut "
                "représenter ce nœud. Réduire N pour un essai ou étudier un élagage explicite ; "
                "le dividende n'est pas tronqué silencieusement."
            )
        return mean

    # Étendre uniquement la chaîne de voisins, jamais un tableau de nœuds.
    def upper_neighbor(self, node: Node, trunk: TrunkNode) -> Node:
        if node.above is None:
            upper = self.create_node(node.spot_price * self.alpha, node.step)
            self.connect_vertical(upper, node)
            trunk.column_size += 1
        return node.above

    def lower_neighbor(self, node: Node, trunk: TrunkNode) -> Node:
        if node.below is None:
            lower = self.create_node(node.spot_price / self.alpha, node.step)
            self.connect_vertical(node, lower)
            trunk.column_size += 1
        return node.below

    def closest_middle(self, candidate: Node, mean: float, trunk: TrunkNode) -> Node:
        # Distance en prix, et non un milieu fixé par l'indice du parent.
        while mean > candidate.spot_price:
            upper = self.upper_neighbor(candidate, trunk)
            if abs(upper.spot_price - mean) >= abs(candidate.spot_price - mean):
                break
            candidate = upper
        while mean < candidate.spot_price:
            lower = self.lower_neighbor(candidate, trunk)
            if abs(lower.spot_price - mean) >= abs(candidate.spot_price - mean):
                break
            candidate = lower
        return candidate

    def connect_trinomial(self, node: Node, middle: Node, trunk: TrunkNode) -> None:
        node.next_mid = middle
        node.next_up = self.upper_neighbor(middle, trunk)
        node.next_down = self.lower_neighbor(middle, trunk)
        mean = self.positive_mean(node)
        variance = node.variance()
        # Résolution des 3 moments centrés : plus stable que E²+V puis soustraction.
        u = node.next_up.spot_price - middle.spot_price
        d = node.next_down.spot_price - middle.spot_price
        e = mean - middle.spot_price
        second = variance + e*e
        p_up = (second - e*d) / (u * (u-d))
        p_down = (second - e*u) / (d * (d-u))
        p_mid = 1.0 - p_up - p_down
        tolerance = 1e-12
        if not all(np.isfinite(p) and -tolerance <= p <= 1+tolerance for p in (p_up,p_mid,p_down)):
            raise ValueError(
                f"Pas {node.step} -> {node.step+1}, S={node.spot_price:.10g} : "
                f"probabilités trinômiales invalides (u={p_up:.6g}, m={p_mid:.6g}, d={p_down:.6g}). "
                "Les moments du cours ne sont pas représentables sur ces trois nœuds ; "
                "modifier le maillage ou étudier un élagage, sans forcer les probabilités."
            )
        # Ne corriger que d'éventuels débordements dus à l'arrondi machine.
        node.p_up = float(min(1.0, max(0.0, p_up)))
        node.p_down = float(min(1.0, max(0.0, p_down)))
        node.p_mid = float(min(1.0, max(0.0, p_mid)))
        total = node.p_up + node.p_mid + node.p_down
        node.p_up /= total
        node.p_mid /= total
        node.p_down /= total

    def build_dividend_column(self, trunk: TrunkNode, step: int) -> TrunkNode:
        next_trunk = TrunkNode(self, self.positive_mean(trunk), step)
        self.connect_trinomial(trunk, next_trunk, next_trunk)
        current = trunk.above
        candidate = next_trunk
        while current is not None:
            candidate = self.closest_middle(candidate, self.positive_mean(current), next_trunk)
            self.connect_trinomial(current, candidate, next_trunk)
            current = current.above
        current = trunk.below
        candidate = next_trunk
        while current is not None:
            candidate = self.closest_middle(candidate, self.positive_mean(current), next_trunk)
            self.connect_trinomial(current, candidate, next_trunk)
            current = current.below
        return next_trunk

    # Connecter deux voisins de la même colonne.
    def connect_vertical(self,upper_node: Node,lower_node: Node) -> None:
        upper_node.below = lower_node
        lower_node.above = upper_node

    # Create a standard node
    def create_node(self,spot_price: float,step: int) -> Node:
        return Node(tree=self,spot_price=spot_price,step=step)

    # Créer les deux enfants du tronc lorsque le tronc monte.
    def build_trunk_up(self,trunk: TrunkNode,step: int) -> TrunkNode:
        fwd = trunk.forward()
        next_trunk = TrunkNode(self,fwd * self.sqrtAlpha,step)
        lower_node = self.create_node(fwd / self.sqrtAlpha,step)
        trunk.next_up = next_trunk
        trunk.next_down = lower_node
        self.connect_vertical(next_trunk,lower_node)
        return next_trunk

    # Créer les deux enfants du tronc lorsque le tronc descend.
    def build_trunk_down(self,trunk: TrunkNode,step: int) -> TrunkNode:
        fwd = trunk.forward()
        next_trunk = TrunkNode(self,fwd / self.sqrtAlpha,step)
        upper_node = self.create_node(fwd * self.sqrtAlpha,step)
        trunk.next_down = next_trunk
        trunk.next_up = upper_node
        self.connect_vertical(upper_node,next_trunk)
        return next_trunk

    # Construire au-dessus du tronc en réutilisant les enfants déjà créés.
    def build_above(self,trunk: TrunkNode,step: int) -> None:
        current = trunk.above
        while current is not None:
            # L'enfant bas est l'enfant haut du voisin du dessous.
            current.next_down = current.below.next_up
            new_up = self.create_node(current.forward() * self.sqrtAlpha,step)
            current.next_up = new_up
            self.connect_vertical(new_up,current.next_down)
            current = current.above

    # Construire en dessous du tronc avec la même recombinaison.
    def build_below(self,trunk: TrunkNode,step: int) -> None:
        current = trunk.below
        while current is not None:
            # L'enfant haut est l'enfant bas du voisin du dessus.
            current.next_up = current.above.next_down
            new_down = self.create_node(current.forward() / self.sqrtAlpha,step)
            current.next_down = new_down
            self.connect_vertical(current.next_up,new_down)
            current = current.below

    # Construire une colonne : tronc, partie haute, puis partie basse.
    def build_next_column(self,trunk: TrunkNode,step: int) -> TrunkNode:
        kind = self.transition_type(step - 1)
        if kind == "deterministic":
            next_trunk = TrunkNode(self, self.positive_mean(trunk), step)
            trunk.next_mid = next_trunk
            trunk.p_up, trunk.p_mid, trunk.p_down = 0.0, 1.0, 0.0
        elif kind == "trinomial":
            next_trunk = self.build_dividend_column(trunk, step)
        elif step % 2 == 1:
            next_trunk = self.build_trunk_up(trunk,step)
        else:
            next_trunk = self.build_trunk_down(trunk,step)
        # Relier les dates pour pouvoir ensuite remonter l'arbre.
        trunk.next_trunk = next_trunk
        next_trunk.previous_trunk = trunk
        if kind == "binomial":
            self.build_above(trunk,step)
            self.build_below(trunk,step)
            next_trunk.column_size = trunk.column_size + 1
        return next_trunk

    # BUILD COMPLETE TREE
    def print_tree(self) -> None:
        trunk = self.root
        while trunk is not None:

            print(f"\nSTEP {trunk.step}")

            # On remonte tout en haut de la colonne
            node = trunk

            while node.above is not None:
                node = node.above

            # Puis on redescend et on affiche tous les noeuds
            while node is not None:

                trunk_marker = " <-- TRUNK" if node is trunk else ""

                print(
                    f"spot = {node.spot_price:.4f}"
                    f"{trunk_marker}"
                )

                if node.option_value is not None:
                    print(f"option = {node.option_value:.6f}")

                node = node.below

            trunk = trunk.next_trunk

    # Avancer de la racine à la dernière colonne.
    def build(self) -> Tree:
        if self._built:
            return self
        if self.root.next_trunk is not None:
            raise RuntimeError("Construction précédente incomplète : créer un nouvel arbre.")
        trunk = self.root

        for step in range(1,self.model.nb_steps + 1):
            trunk = self.build_next_column(trunk,step)
        self.last_trunk = trunk
        self._built = True
        return self

    # Appliquer une fonction à chaque nœud de la colonne.

    def apply_to_column(self,trunk: TrunkNode,function: Callable[[Node],None]) -> None:
        # trunk itself
        function(trunk)

        # Nodes above trunk
        node = trunk.above

        while node is not None:
            function(node)
            node = node.above

        # Nodes below trunk
        node = trunk.below

        while node is not None:

            function(node)

            node = node.below

    # BACKWARD OPTION PRICING

    def price(self) -> float:
        if not self._built:
            raise RuntimeError("Appeler tree.build() avant tree.price().")
        # 1. PAYOFF AT MATURITY
        self.apply_to_column(

            self.last_trunk,

            lambda node:
                setattr(
                    node,
                    "option_value",
                    self.option.intrinsic_value(node.spot_price)
                )
        )

        # 2. MOVE BACKWARD ALONG TRUNK
        trunk = (self.last_trunk.previous_trunk)

        while trunk is not None:

            # Calculate every node of current date
            self.apply_to_column(trunk,
                lambda node:
                    node.compute_option_value()
            )

            # Move one date backward
            trunk = trunk.previous_trunk
        # ROOT OPTION PRICE
        return self.root.option_value

# MAIN

def main() -> None:
    # Paramètres de l'exemple 
    pricing_date = dt.date(2026, 9, 15)
    maturity = dt.date(2027, 9, 2)
    model = Model(pricing_date=pricing_date,nb_steps=1000)
    option = Option(strike=101,maturity=maturity,
                    type="Call",exercise_type="European")
    market = Market(spot=100,rate=0.03,vol=0.20)
    tree = Tree(model=model,option=option,market=market)

    # Toujours construire l'arbre avant de calculer le prix.
    tree.build()
    price = tree.price()

    # Afficher les paramètres et le prix final.
    print("ARBRE BINOMIAL / PAS TRINOMIAL SI DIVIDENDE CASH")
    print(f"Date de calcul : {pricing_date:%d/%m/%Y}")
    print(f"Maturité : {maturity:%d/%m/%Y}")
    print(f"Option : {option.type} {option.exercise_type}")
    print(f"Spot : {market.spot} | Strike : {option.strike}")
    print(f"Taux : {market.rate:.1%} | Volatilité : {market.vol:.1%}")
    print(f"Rendement continu q : {market.dividend_yield:.1%}")
    print(f"Dividende cash : {market.dividend_amount:g} | Ex-date : {market.ex_dividend_date}")
    print(f"Nombre de pas : {model.nb_steps}")
    print(f"\nPrix de l'option : {price:.10f}")
    # tree.print_tree() # Permet d'afficher chaque step.

if __name__ == "__main__":
    main()
