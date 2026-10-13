from __future__ import annotations

import numpy as np
import datetime as dt
from typing import Callable

# MODEL

class Model:

    def __init__(self, pricing_date: dt.date, nb_steps: int) -> None:
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
        vol: float
    ) -> None:
        self.spot = spot
        self.rate = rate
        self.vol = vol

class Node:
    # Un nœud conserve son prix et ses liens vers les autres nœuds.

    def __init__(self,tree: Tree,spot_price: float,step: int) -> None:
        self.tree = tree
        self.spot_price = spot_price
        self.step = step
        self.next_up = None
        self.next_down = None
        self.above = None
        self.below = None
        self.option_value = None

    # Forward sur un pas de temps, sans dividende.
    def forward(self) -> float:
        return ( self.spot_price
            * np.exp(self.tree.market.rate * self.tree.deltaT))

    def compute_option_value(self) -> None:
        # European continuation value
        hold_value = (self.tree.df
            * (self.tree.p_up* self.next_up.option_value
                +
                self.tree.p_down* self.next_down.option_value))

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

# TREE

class Tree:

    def __init__(self,model: Model,option: Option,market: Market) -> None:
        self.model = model
        self.option = option
        self.market = market

        # TIME STEP
        maturity_years = (option.maturity- model.pricing_date).days / 365.0
        self.deltaT = (maturity_years/ model.nb_steps)
        #discount factor
        self.df = np.exp(-market.rate * self.deltaT)

        # TREE PARAMETERS
        self.sqrtAlpha = self.compute_sqrt_alpha()
        self.p_up = (1/(1 + self.sqrtAlpha))
        self.p_down = (self.sqrtAlpha/(1 + self.sqrtAlpha))

        # ROOT
        self.root = TrunkNode(tree=self,spot_price=market.spot,step=0)
        self.last_trunk = self.root

    # Compute sqrt(alpha)
    def compute_sqrt_alpha(self) -> float:
        sigma = self.market.vol
        a = (1 + np.exp(sigma**2 * self.deltaT)) / 2
        # Cette formule donne directement sqrt(alpha).
        return a + np.sqrt(a**2 - 1)

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
        if step % 2 == 1:
            next_trunk = self.build_trunk_up(trunk,step)
        else:
            next_trunk = self.build_trunk_down(trunk,step)
        # Relier les dates pour pouvoir ensuite remonter l'arbre.
        trunk.next_trunk = next_trunk
        next_trunk.previous_trunk = trunk
        self.build_above(trunk,step)
        self.build_below(trunk,step)
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
        trunk = self.root

        for step in range(1,self.model.nb_steps + 1):
            trunk = self.build_next_column(trunk,step)
        self.last_trunk = trunk
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
    print("ARBRE BINOMIAL")
    print(f"Date de calcul : {pricing_date:%d/%m/%Y}")
    print(f"Maturité : {maturity:%d/%m/%Y}")
    print(f"Option : {option.type} {option.exercise_type}")
    print(f"Spot : {market.spot} | Strike : {option.strike}")
    print(f"Taux : {market.rate:.1%} | Volatilité : {market.vol:.1%}")
    print(f"Nombre de pas : {model.nb_steps}")
    print(f"\nPrix de l'option : {price:.10f}")
    # tree.print_tree() # Permet d'afficher chaque step.

if __name__ == "__main__":
    main()
