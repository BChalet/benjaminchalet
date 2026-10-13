from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from arbre_binomial import Market, Model, Option, Node, TrunkNode, Tree

# DEBUGGING 
class DebugTree:

    def __init__(self,tree: Tree) -> None:
        self.tree = tree
        self.errors = 0
        self.node_count = 0
        self.branch_count = 0
        self.output = Path(__file__).resolve().parent

    # Comparer les nombres avec une tolérance adaptée aux arrondis
    def close(self,actual: float,expected: float) -> bool:
        return bool(np.isclose(actual,expected,rtol=1e-9,atol=1e-10))

    # Afficher chaque erreur 
    def error(self,message: str,node: Node | None = None) -> None:
        self.errors += 1
        location = "" if node is None else f"Pas {node.step}, S={node.spot_price:.6f} : "
        print("ERREUR :",location + message)

    # Suivre le tronc avec une borne pour éviter une boucle infinie
    def columns(self) -> Iterator[TrunkNode]:
        trunk = self.tree.root
        previous = None
        for step in range(self.tree.model.nb_steps + 1):
            if not isinstance(trunk,TrunkNode) or trunk.step != step:
                raise ValueError(f"Tronc absent ou mauvaise date au pas {step}")
            if trunk.previous_trunk is not previous:
                raise ValueError(f"Lien précédent du tronc incorrect au pas {step}")
            yield trunk
            previous = trunk
            trunk = trunk.next_trunk
        if trunk is not None or previous is not self.tree.last_trunk:
            raise ValueError("Fin du tronc incorrecte")

    # Parcourir une colonne de haut en bas ; arrêter si les liens forment un cycle.
    def column_nodes(self,trunk: TrunkNode) -> Iterator[Node]:
        node = trunk
        moves = 0
        while node.above is not None:
            node = node.above
            moves += 1
            if moves > trunk.step:
                raise ValueError(f"Cycle ou trop de voisins au-dessus au pas {trunk.step}")
        count = 0
        while node is not None:
            count += 1
            if count > trunk.step + 1 or node.step != trunk.step:
                raise ValueError(f"Colonne incohérente au pas {trunk.step}")
            yield node
            node = node.below
        if count != trunk.step + 1:
            raise ValueError(f"Colonne {trunk.step} : {count} nœuds au lieu de {trunk.step+1}")

    # Vérifier que l'enfant est bien l'un des objets de la colonne suivante.
    def in_column(self,child: Node,trunk: TrunkNode) -> bool:
        for node in self.column_nodes(trunk):
            if node is child:
                return True
        return False

    # Contrôler une branche à partir de l'objet enfant réellement référencé.
    def check_child(self,node: Node,child: Node | None,trunk: TrunkNode,factor: float) -> None:
        if child is None:
            self.error("enfant manquant",node)
            return
        self.branch_count += 1
        if child.step != node.step + 1 or not self.in_column(child,trunk):
            self.error("enfant hors de la colonne suivante",node)
        if not self.close(child.spot_price,node.forward()*factor):
            self.error("prix de l'enfant incorrect",node)

    # Les liens verticaux sont réciproques ; deux parents partagent le même enfant.
    def check_neighbors(self,node: Node) -> None:
        if node.above is not None:
            if node.above.step != node.step or node.above.below is not node:
                self.error("lien above/below incorrect",node)
        if node.below is not None:
            if node.below.step != node.step or node.below.above is not node:
                self.error("lien below/above incorrect",node)
            if node.spot_price < node.below.spot_price:
                self.error("prix mal ordonnés dans la colonne",node)
            if node.step < self.tree.model.nb_steps:
                if node.next_down is not node.below.next_up:
                    self.error("recombinaison incorrecte",node)

    # Un nœud intérieur a deux branches entrantes, un nœud extrême en a une.
    def check_parents(self,node: Node,previous: TrunkNode | None) -> None:
        if previous is None:
            if node is not self.tree.root:
                self.error("nœud supplémentaire à la date initiale",node)
            return
        count = 0
        for parent in self.column_nodes(previous):
            count += int(parent.next_up is node) + int(parent.next_down is node)
        expected = 1 if node.above is None or node.below is None else 2
        if count != expected:
            self.error(f"{count} branches entrantes au lieu de {expected}",node)

    # Probabilités et facteurs communs : contrôles financiers indépendants des liens.
    def check_parameters(self) -> None:
        up = self.tree.p_up
        down = self.tree.p_down
        if not (0 <= up <= 1 and 0 <= down <= 1):
            self.error("probabilités en dehors de [0,1]")
        if not self.close(up+down,1):
            self.error("la somme des probabilités ne vaut pas 1")
        expected_dt = (self.tree.option.maturity-self.tree.model.pricing_date).days/365/self.tree.model.nb_steps
        if not self.close(self.tree.deltaT,expected_dt):
            self.error("pas de temps incorrect")
        if not self.close(self.tree.df,np.exp(-self.tree.market.rate*expected_dt)):
            self.error("facteur d'actualisation incorrect")
        if not self.close(self.tree.root.spot_price,self.tree.market.spot):
            self.error("prix de la racine différent du spot")

    # Comparer l'espérance et la variance des enfants aux formules du cours.
    def check_moments(self,node: Node) -> None:
        if node.next_up is None or node.next_down is None:
            return
        up = node.next_up.spot_price
        down = node.next_down.spot_price
        mean = self.tree.p_up*up + self.tree.p_down*down
        forward = node.spot_price*np.exp(self.tree.market.rate*self.tree.deltaT)
        variance = self.tree.p_up*(up-mean)**2 + self.tree.p_down*(down-mean)**2
        # expm1(x) calcule exp(x)-1 avec une meilleure précision pour x petit.
        expected = forward**2*np.expm1(self.tree.market.vol**2*self.tree.deltaT)
        if not self.close(mean,forward):
            self.error("espérance différente du forward",node)
        if not self.close(variance,expected):
            self.error("variance différente de la variance théorique",node)

    # Contrôler les nœuds accessibles par les colonnes, sans les enregistrer.
    def check_column(self,trunk: TrunkNode) -> None:
        for node in self.column_nodes(trunk):
            self.node_count += 1
            if node.tree is not self.tree:
                self.error("référence vers un autre arbre",node)
            self.check_neighbors(node)
            self.check_parents(node,trunk.previous_trunk)
            if node.step < self.tree.model.nb_steps:
                if not isinstance(trunk.next_trunk,TrunkNode):
                    raise ValueError("Colonne suivante absente")
                self.check_child(node,node.next_up,trunk.next_trunk,self.tree.sqrtAlpha)
                self.check_child(node,node.next_down,trunk.next_trunk,1/self.tree.sqrtAlpha)
                self.check_moments(node)
            elif node.next_up is not None or node.next_down is not None:
                self.error("branche après maturité",node)

    # Le contrôle complet retourne le nombre d'erreurs, pas une liste.
    def check(self) -> int:
        self.errors = 0
        self.node_count = 0
        self.branch_count = 0
        self.check_parameters()
        try:
            for trunk in self.columns():
                self.check_column(trunk)
        except ValueError as exc:
            self.error(str(exc))
        print(f"DEBUG : {self.node_count} nœuds parcourus, {self.branch_count} branches")
        if self.errors == 0:
            self.check_zero_strike()
        print(f"Bilan : {self.errors} erreur(s) détectée(s)")
        return self.errors

    # Call européen K=0, sans dividende : sa valeur actuelle doit être S0.
    def check_zero_strike(self) -> None:
        original = self.tree.option
        try:
            self.tree.option = Option(0,original.maturity,"Call","European")
            price = self.tree.price()
            if not self.close(price,self.tree.market.spot):
                self.error("le call européen de strike nul ne vaut pas le spot")
            print(f"Call européen K=0 : {price:.10f} ; spot : {self.tree.market.spot:.10f}")
        finally:
            # Rétablir l'option et ses valeurs après le contrôle temporaire.
            self.tree.option = original
            self.tree.price()

    # Accéder à une valeur numérique sans conserver le nœud ailleurs.
    def value(self,node: Node,color_by: str) -> float:
        return node.spot_price if color_by == "spot" else node.option_value

    # Trouver les bornes de couleur par un parcours, sans tableau de valeurs.
    def color_scale(self,color_by: str) -> Normalize:
        minimum = float("inf")
        maximum = -float("inf")
        for trunk in self.columns():
            for node in self.column_nodes(trunk):
                value = self.value(node,color_by)
                minimum = min(minimum,value)
                maximum = max(maximum,value)
        return Normalize(minimum,maximum if maximum>minimum else minimum+1)

    # Les tuples ci-dessous contiennent les coordonnées des segments.
    def draw_edges(self,ax: Axes,node: Node,show_neighbors: bool) -> None:
        if node.next_up is not None:
            ax.plot((node.step,node.next_up.step),(node.spot_price,node.next_up.spot_price),
                    color="#94a3b8",linewidth=1,zorder=1)
        if node.next_down is not None:
            ax.plot((node.step,node.next_down.step),(node.spot_price,node.next_down.spot_price),
                    color="#94a3b8",linewidth=1,zorder=1)
        if show_neighbors and node.below is not None:
            ax.plot((node.step,node.below.step),(node.spot_price,node.below.spot_price),
                    ":",color="gray",zorder=0)

    # Un appel de dessin par nœud 
    def draw_node(self,ax: Axes,node: Node,norm: Normalize,color_by: str) -> None:
        value = self.value(node,color_by)
        color = plt.get_cmap("viridis")(norm(value))
        size = 46 if self.tree.model.nb_steps <= 6 else 30
        ax.plot(node.step,node.spot_price,"o",markersize=size,
                markerfacecolor=color,markeredgecolor="white",zorder=3)
        label = f"S {node.spot_price:.2f}\nV {node.option_value:.2f}"
        ax.text(node.step,node.spot_price,label,ha="center",va="center",
                fontsize=8 if size==46 else 6,
                color="white" if norm(value)<0.55 else "black",zorder=4)

    # Forward initial : un repère calculé aux dates de l'arbre, pas une branche.
    def draw_forward(self,ax: Axes) -> None:
        previous = self.tree.market.spot
        for step in range(1,self.tree.model.nb_steps+1):
            forward = self.tree.market.spot*np.exp(self.tree.market.rate*step*self.tree.deltaT)
            label = "Forward initial F(0,t)" if step==1 else "_nolegend_"
            ax.plot((step-1,step),(previous,forward),"--",
                    color="#ea580c",label=label,zorder=2)
            previous = forward
        ax.legend(loc="upper left")

    # Mise en forme commune ; les contrôles numériques restent lisibles au terminal.
    def decorate(self,ax: Axes,norm: Normalize,color_by: str) -> None:
        self.draw_forward(ax)
        ax.set(title="Arbre binomial\nS : sous-jacent ; V : option",
               xlabel="Pas de temps",ylabel="Prix du sous-jacent",
               xticks=range(self.tree.model.nb_steps+1))
        ax.margins(x=0.08,y=0.12)
        ax.grid(axis="x",alpha=0.15)
        label = "Valeur de l'option" if color_by=="option" else "Prix du sous-jacent"
        ax.figure.colorbar(ScalarMappable(norm=norm,cmap="viridis"),ax=ax,label=label)

    # Arrêter le dessin en cas d'erreur, sans essayer de réparer les connexions.
    def show(self,color_by: str = "option",show_neighbors: bool = False) -> None:
        if not 1 <= self.tree.model.nb_steps <= 10 or self.tree.market.vol <= 0:
            raise ValueError("Pour le dessin : 1 à 10 pas et vol > 0. Sinon utiliser check().")
        if color_by != "spot" and color_by != "option":
            raise ValueError("color_by doit être 'spot' ou 'option'.")
        if self.check() != 0:
            print("Dessin interrompu : corriger les erreurs signalées puis relancer.")
            return
        norm = self.color_scale(color_by)
        fig,ax = plt.subplots(figsize=(15,9),layout="constrained")
        for trunk in self.columns():
            for node in self.column_nodes(trunk):
                self.draw_edges(ax,node,show_neighbors)
                self.draw_node(ax,node,norm,color_by)
        self.decorate(ax,norm,color_by)
        fig.savefig(self.output / "debug_arbre.png",dpi=160)
        print("Figure enregistrée :",self.output / "debug_arbre.png")
        plt.show()

# MAIN : construire un petit arbre, puis vérifier et afficher ses liens.

def main() -> None:
    pricing_date = dt.date(2026,9,15)
    maturity = dt.date(2027,9,2)
    model = Model(pricing_date=pricing_date,nb_steps=1000)
    market = Market(spot=100,rate=0.03,vol=0.20)
    option = Option(strike=101,maturity=maturity,type="Call",exercise_type="European")
    tree = Tree(model=model,option=option,market=market)
    tree.build()
    debugger = DebugTree(tree)
    # Choisir 'spot' ou 'option' pour le dégradé ; True pour voir above/below.
    debugger.show(color_by="option",show_neighbors=True)
    #mettre debugger.check() pour ne pas dessiner mais vérifier les liens et les valeurs numériques si n>10
    #debugger.check()

# L'import de ce fichier ne lance pas son exemple.
if __name__ == "__main__":
    main()
