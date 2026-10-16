"""Arbre recombiné à liens directs : construction puis rétropropagation.

Base : fichiers ``arbre_binomial.py`` et ``convergence.py`` joints le 16/10/2026.
Compléments : décision d'exercice, profils par date et états de diagnostic.
Le maillage conserve ses liens ``above/below`` et ``next_up/mid/down`` : aucun
conteneur de nœuds n'est nécessaire pour construire ou parcourir l'arbre.
Les collections de résultats ne contiennent que des nombres et des dates.
"""

from __future__ import annotations

import numpy as np
import datetime as dt
from time import perf_counter
from typing import Callable, Iterator
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# RÉSULTATS PAR DATE : données de sortie, jamais un moyen d'accès aux nœuds.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ExerciseProfileRow:
    """Une ligne du profil risque-neutre calculé par ``exercise_profile``.

    Les valeurs monétaires sont toutes actualisées à la date de calcul.
    ``date`` peut contenir une heure : les N intervalles ont la même durée
    et ne correspondent pas nécessairement à un nombre entier de jours.
    ``survival_probability`` est mesurée APRÈS la décision de la date ; au
    dernier pas, elle représente les options expirant sans paiement.
    """

    step: int
    time_years: float
    date: dt.datetime
    discounted_mean_value: float
    exercise_region_probability: float
    first_exercise_probability: float
    cumulative_exercise_probability: float
    survival_probability: float
    discounted_alive_value: float
    discounted_exercise_cashflow: float
    cumulative_discounted_cashflow: float
    total_value: float


# MODEL

class Model:
    """Paramètres numériques : date de calcul, nombre de pas et élagage."""

    def __init__(self, pricing_date: dt.date, nb_steps: int, pruning_threshold: float = 0.0) -> None:
        if not isinstance(pricing_date, dt.date) or isinstance(pricing_date, dt.datetime):
            raise ValueError("pricing_date doit être une date (datetime.date).")
        if isinstance(nb_steps, bool) or not isinstance(nb_steps, int) or nb_steps < 1:
            raise ValueError("nb_steps doit être un entier strictement positif.")
        if not np.isfinite(pruning_threshold) or not 0 <= pruning_threshold <= 1:
            raise ValueError("Le seuil de pruning doit être fini et compris entre 0 et 1.")
        self.pricing_date = pricing_date
        self.nb_steps = nb_steps
        # Pruning : un nœud du bord dont la probabilité d'être atteint est sous
        # ce seuil n'a plus qu'un seul enfant (0 = pas de pruning).
        self.pruning_threshold = pruning_threshold

# OPTION

class Option:
    """Contrat : payoff call/put et possibilité d’exercice anticipé."""

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
    """Marché constant : spot, taux r, volatilité, rendement q et un cash D."""

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
        # Continuation = valeur actualisée des enfants, AVANT comparaison
        # avec le payoff. À maturité, il n'y a plus de continuation (None).
        self.continuation_value: float | None = None
        self.exercise_now = False
        # Ce drapeau décrit la construction, indépendamment de l'option.
        # Une transition déterministe (volatilité nulle) n'est pas un pruning.
        self.is_pruned = False
        # Masse qui atteint le nœud sans exercice antérieur ; remise à zéro
        # à chaque calcul du profil. Elle ne remplace jamais ``proba``.
        self.alive_proba = 0.0
        # Probabilité d'atteindre ce nœud depuis la racine (sert au pruning).
        self.proba = 0.0

    # Espérance conditionnelle sur le prochain pas, dividendes inclus.
    def forward(self) -> float:
        return self.tree.expected_spot(self.spot_price, self.step)

    def variance(self) -> float:
        return self.tree.conditional_variance(self.spot_price, self.step)

    def compute_option_value(self) -> None:
        """Rétropropager puis enregistrer la décision de gestion optimale.

        Les valeurs des enfants sont déjà calculées puisque ``Tree.price``
        parcourt les dates à rebours. Le facteur d'actualisation utilise r,
        même lorsque le rendement du sous-jacent est r-q.
        """
        expectation = 0.0
        for branch in ("up", "mid", "down"):
            child = getattr(self, "next_" + branch)
            if child is not None:
                expectation += getattr(self, "p_" + branch) * child.option_value
        self.continuation_value = float(self.tree.df * expectation)
        self.exercise_now = False
        exercise_type = self.tree.option.exercise_type.lower()
        if exercise_type == "european":
            self.option_value = self.continuation_value
        elif exercise_type == "american":
            payoff = self.tree.option.intrinsic_value(self.spot_price)
            # Une égalité à l'arrondi près ne déclenche pas d'exercice.
            # Cette convention évite des décisions artificielles sur un
            # call sans dividende ou sur des nœuds de payoff nul.
            tolerance = self.tree.exercise_tolerance(payoff, self.continuation_value)
            self.exercise_now = payoff > 0 and payoff > self.continuation_value + tolerance
            # Cohérence entre valeur et stratégie : dans la bande d'égalité
            # numérique on conserve la continuation. L'écart local à max()
            # est donc au plus égal à la tolérance annoncée.
            self.option_value = payoff if self.exercise_now else self.continuation_value
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
    """Graphe recombiné du sous-jacent, indépendant du strike et du payoff.

    1. ``build`` avance dans le temps et additionne les probabilités entrantes.
    2. ``price`` remonte le temps pour valoriser le contrat.
    3. ``exercise_profile`` avance à nouveau pour les statistiques d’arrêt.
    """

    def __init__(self,model: Model,option: Option,market: Market) -> None:
        self.model = model
        self.option = option
        self.market = market

        self.duration_days = (option.maturity - model.pricing_date).days
        if self.duration_days <= 0:
            raise ValueError("La maturité doit être postérieure à la date de calcul.")
        self.drift = market.rate - market.dividend_yield

        # Un seul dividende cash : affectation entière à (t_i, t_{i+1}).
        # Ce calcul évite les erreurs d'arrondi de dates signalées dans le cours.
        self.dividend_step = None
        if market.dividend_amount > 0 and model.pricing_date < market.ex_dividend_date <= option.maturity:
            ex_days = (market.ex_dividend_date - model.pricing_date).days
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
        self.root.proba = 1.0
        self.last_trunk = self.root
        self._built = False

    # Compute sqrt(alpha)
    def compute_sqrt_alpha(self) -> float:
        """Facteur binomial >1 obtenu en égalant moyenne et variance (p7–8).

        Le nom historique ``sqrtAlpha`` est conservé. Il correspond au
        facteur alpha BINOMIAL des diapositives ; ``self.alpha`` est son
        carré, c’est-à-dire le rapport entre voisins d’une même colonne.
        ``expm1(x)`` évite la perte de précision de ``exp(x)-1`` pour x petit.
        """
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
            # Approximation intra-pas du cours: w = (t_fin-t_ex)/dt.
            # Elle ne constitue pas la variance exacte d'un GBM avec saut cash intérieur.
            base -= self.market.dividend_amount * self.remaining_after_dividend / self.deltaT
        return float((base * self.growth)**2 * self.variance_factor)

    #avance le long du tronc avce next_trunk, date par date
    def iter_columns(self) -> Iterator[TrunkNode]:
        trunk = self.root
        while trunk is not None:
            yield trunk
            trunk = trunk.next_trunk
    #monte tout en haut d'une colonne avec above et redescend tout en bas avec below
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
        """Pas cash : tronc au forward puis trois enfants voisins par parent.

        Au-dessus et au-dessous, le curseur ``candidate`` reste sur la
        nouvelle colonne. Il se déplace par les liens verticaux pour trouver
        le milieu le plus proche du forward de chaque parent. Les voisins
        déjà créés sont réutilisés : l’arbre reste recombiné. Ce pas n’est
        volontairement pas élagué dans la version fournie.
        """
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

    # Pruning : seul un nœud du bord peut être coupé, s'il est trop improbable.
    def is_pruned(self, node: Node, is_edge: bool) -> bool:
        return is_edge and node.proba < self.model.pruning_threshold

    # Un nœud coupé va vers un seul enfant (déjà existant), avec probabilité 1.
    def connect_monomial(self, node: Node, child: Node) -> None:
        node.is_pruned = True
        node.next_up = None
        node.next_down = None
        node.next_mid = child
        node.p_up, node.p_mid, node.p_down = 0.0, 1.0, 0.0

    # Construire au-dessus du tronc en réutilisant les enfants déjà créés.
    # Renvoie 1 si le nœud du haut a été coupé, 0 sinon.
    def build_above(self,trunk: TrunkNode,step: int) -> int:
        current = trunk.above
        while current is not None:
            # L'enfant bas est l'enfant haut du voisin du dessous.
            down_child = current.below.next_up
            if self.is_pruned(current, current.above is None):
                # Pas de nouvel enfant en haut : la colonne ne grandit pas.
                self.connect_monomial(current, down_child)
                return 1
            current.next_down = down_child
            new_up = self.create_node(current.forward() * self.sqrtAlpha,step)
            current.next_up = new_up
            self.connect_vertical(new_up,current.next_down)
            current = current.above
        return 0

    # Construire en dessous du tronc avec la même recombinaison.
    # Renvoie 1 si le nœud du bas a été coupé, 0 sinon.
    def build_below(self,trunk: TrunkNode,step: int) -> int:
        current = trunk.below
        while current is not None:
            # L'enfant haut est l'enfant bas du voisin du dessus.
            up_child = current.above.next_down
            if self.is_pruned(current, current.below is None):
                self.connect_monomial(current, up_child)
                return 1
            current.next_up = up_child
            new_down = self.create_node(current.forward() / self.sqrtAlpha,step)
            current.next_down = new_down
            self.connect_vertical(current.next_up,new_down)
            current = current.below
        return 0

    # Transmettre la probabilité de chaque nœud de la colonne à ses enfants.
    def propagate_probabilities(self, trunk: TrunkNode) -> None:
        """Additionner toutes les contributions parentales dans les enfants.

        Il faut ``+=`` : un enfant recombiné reçoit plusieurs chemins. La
        colonne suivante est neuve et ses probabilités commencent à zéro.
        Ces masses décrivent le sous-jacent, sans arrêt américain ; elles
        servent aussi à décider l’élagage à la prochaine construction.
        """
        def push(node: Node) -> None:
            if node.next_up is not None:
                node.next_up.proba += node.p_up * node.proba
            if node.next_mid is not None:
                node.next_mid.proba += node.p_mid * node.proba
            if node.next_down is not None:
                node.next_down.proba += node.p_down * node.proba
        self.apply_to_column(trunk, push)

    # Nombre total de nœuds de l'arbre (pour mesurer l'effet du pruning).
    def count_nodes(self) -> int:
        return sum(trunk.column_size for trunk in self.iter_columns())

    # Construire une colonne : tronc, partie haute, puis partie basse.
    def build_next_column(self,trunk: TrunkNode,step: int) -> TrunkNode:
        """Construire une date puis la relier aux deux dates voisines.

        Hors cash, le tronc monte aux pas impairs et descend aux pas pairs.
        Il reste près du centre du triangle, permettant deux parcours
        indépendants : vers le haut puis vers le bas. La formule de taille
        ``ancienne + 1 - bords_coupés`` compte les objets réellement créés,
        pas les nœuds du triangle théorique complet.
        """
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
            pruned = self.build_above(trunk,step) + self.build_below(trunk,step)
            # Une colonne binomiale a un nœud de plus, moins les bords coupés.
            next_trunk.column_size = trunk.column_size + 1 - pruned
        self.propagate_probabilities(trunk)
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

    # -------------------------------------------------------------------------
    # VALEUR ET EXERCICE : mouvement arrière, puis mouvement avant des masses.
    # -------------------------------------------------------------------------
    @staticmethod
    def exercise_tolerance(payoff: float, continuation: float) -> float:
        """Tolérance absolue + relative, exprimée dans l'unité du payoff."""
        return 1e-12 * max(1.0, abs(payoff), abs(continuation))

    def price(self) -> float:
        """Recalculer toutes les valeurs et décisions pour l'option actuelle.

        On peut changer le strike, call/put ou européen/américain puis appeler
        cette méthode sans reconstruire le sous-jacent. Modifier le marché,
        les dates ou N nécessite en revanche un nouvel objet ``Tree``.
        """
        if not self._built:
            raise RuntimeError("Appeler tree.build() avant tree.price().")
        for node in self.iter_column(self.last_trunk):
            payoff = self.option.intrinsic_value(node.spot_price)
            node.option_value = payoff
            node.continuation_value = None
            # À maturité, « exercice » signifie un paiement strictement
            # positif. Une option à la monnaie ou hors monnaie expire.
            node.exercise_now = payoff > 0.0
        trunk = self.last_trunk.previous_trunk
        while trunk is not None:
            for node in self.iter_column(trunk):
                node.compute_option_value()
            trunk = trunk.previous_trunk
        return float(self.root.option_value)

    def mean_at_time(self, time_years: float) -> float:
        """Espérance théorique de S(t), dividende cash déjà détaché inclus.

        Cette référence concerne le modèle avant élagage. Le pruning fourni
        redirige les branches des bords et peut donc la modifier légèrement.
        """
        maturity_years = self.duration_days / 365.0
        if not np.isfinite(time_years) or not 0 <= time_years <= maturity_years + 1e-12:
            raise ValueError("Le temps doit être compris entre 0 et la maturité.")
        mean = self.market.spot * np.exp(self.drift * time_years)
        if self.dividend_step is not None:
            ex_time = (self.market.ex_dividend_date - self.model.pricing_date).days / 365.0
            if time_years >= ex_time - 1e-12:
                mean -= self.market.dividend_amount * np.exp(self.drift * (time_years - ex_time))
        return float(mean)

    def expected_zero_strike(self) -> float:
        """Call européen K=0 théorique = forward final actualisé (cours p21)."""
        maturity_years = self.duration_days / 365.0
        return float(np.exp(-self.market.rate * maturity_years) * self.mean_at_time(maturity_years))

    def exercise_profile(self) -> tuple[ExerciseProfileRow, ...]:
        """Calculer les profils de valeur et d'exercice du cours, pages 22–23.

        Premier passage (``price``) : résolution arrière du problème d'arrêt.
        Second passage : propagation avant de deux distributions distinctes.

        * ``proba`` existe depuis la construction : on ignore les exercices.
          Sa moyenne actualisée de V est la courbe demandée dans le PDF.
        * ``alive_proba`` est absorbée au PREMIER exercice. Elle permet de
          calculer une vraie distribution des dates de paiement et son cumul.

        Il n'y a ni renormalisation ni actualisation des probabilités. Seules
        les valeurs monétaires et les payoffs sont actualisés. Le pruning
        actuel redirige toute la masse avec p_mid=1 ; les profils concernent
        donc le modèle élagué lorsque le seuil est non nul.
        """
        self.price()  # Évite de réutiliser les décisions d'un ancien strike.
        for trunk in self.iter_columns():
            for node in self.iter_column(trunk):
                node.alive_proba = 0.0
        self.root.alive_proba = 1.0
        rows: list[ExerciseProfileRow] = []
        cumulative_exercise = 0.0
        past_cashflows = 0.0
        start_date = dt.datetime.combine(self.model.pricing_date, dt.time())
        for trunk in self.iter_columns():
            time_years = trunk.step * self.deltaT
            discount = float(np.exp(-self.market.rate * time_years))
            mean_value = region = first_exercise = alive_value = cashflow = survival = 0.0
            for node in self.iter_column(trunk):
                # La moyenne brute peut contenir des trajectoires qui auraient
                # déjà exercé. C'est volontaire pour reproduire le PDF.
                mean_value += node.proba * node.option_value
                alive_value += node.alive_proba * node.option_value
                if node.exercise_now:
                    region += node.proba
                    first_exercise += node.alive_proba
                    cashflow += node.alive_proba * self.option.intrinsic_value(node.spot_price)
                    # Absorption : aucune masse de ce nœud ne va aux enfants.
                else:
                    survival += node.alive_proba
                    for branch in ("up", "mid", "down"):
                        child = getattr(node, "next_" + branch)
                        if child is not None:
                            child.alive_proba += node.alive_proba * getattr(node, "p_" + branch)
            cumulative_exercise += first_exercise
            discounted_cashflow = discount * cashflow
            rows.append(ExerciseProfileRow(
                step=trunk.step,
                time_years=time_years,
                date=start_date + dt.timedelta(days=trunk.step * self.duration_days / self.model.nb_steps),
                discounted_mean_value=discount * mean_value,
                exercise_region_probability=region,
                first_exercise_probability=first_exercise,
                cumulative_exercise_probability=cumulative_exercise,
                survival_probability=survival,
                discounted_alive_value=discount * alive_value,
                discounted_exercise_cashflow=discounted_cashflow,
                cumulative_discounted_cashflow=past_cashflows + discounted_cashflow,
                # La valeur vivante est mesurée AVANT décision à cette date :
                # ajouter seulement les flux des dates STRICTEMENT antérieures,
                # sinon le paiement du jour serait compté deux fois.
                total_value=discount * alive_value + past_cashflows,
            ))
            past_cashflows += discounted_cashflow
        return tuple(rows)

# MAIN

def main() -> None:
    # Paramètres de l'exemple
    pricing_date = dt.date(2026, 9, 15)
    maturity = dt.date(2027, 9, 2)
    # pruning_threshold = 0 : pas de pruning ; 1e-7 : on coupe les nœuds de proba < 1e-7
    model = Model(pricing_date=pricing_date,nb_steps=1000,pruning_threshold=1e-7)
    option = Option(strike=101,maturity=maturity,
                    type="Put",exercise_type="European")
    market = Market(spot=100,rate=0.03,vol=0.20,dividend_yield=0,dividend_amount=3,ex_dividend_date=dt.date(2027, 3, 15))

    # Toujours construire l'arbre avant de calculer le prix (on chronomètre les deux).
    start = perf_counter()
    tree = Tree(model=model,option=option,market=market)
    tree.build()
    price = tree.price()
    elapsed = perf_counter() - start

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
    print(f"Seuil de pruning : {model.pruning_threshold:g}")
    print(f"\nPrix de l'option : {price:.10f}")
    print(f"Nombre de nœuds : {tree.count_nodes()}")
    print(f"Temps de calcul : {elapsed:.2f} s")
    # tree.print_tree() # Permet d'afficher chaque step.

if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# PRUNING SUR LA PARTIE BINOMIALE
# -----------------------------------------------------------------------------
# Le cours présente le pruning sur un arbre trinomial : un nœud trop improbable
# ne branche plus que vers son nœud du milieu, avec probabilité 1. Dans notre
# arbre binomial, il n'y a pas de nœud du milieu, donc on l'a adapté ainsi :
#
# 1. Chaque nœud garde sa probabilité d'être atteint depuis la racine (proba).
#    La racine vaut 1, puis chaque nœud transmet proba × p_up et proba × p_down
#    à ses enfants (propagate_probabilities).
#
# 2. Seuls les nœuds du bord (tout en haut ou tout en bas d'une colonne)
#    peuvent être coupés : cela évite de créer des trous dans une colonne.
#    Le seuil porte sur leur masse totale, y compris les contributions déjà
#    redirigées par des coupes antérieures.
#
# 3. Si un nœud du bord a une proba < pruning_threshold, on ne crée pas de
#    nouvel enfant vers l'extérieur. Il va, avec probabilité 1, vers l'enfant
#    déjà créé par son voisin (connect_monomial dans build_above / build_below).
#    La colonne suivante ne grandit donc pas de ce côté à ce pas. La largeur
#    dépend ensuite des probabilités cumulées et des coupes suivantes.
#
# Limite : le nœud redirigé ne va pas exactement vers son forward, donc
# l'espérance et la variance locales sont modifiées. Un seuil sur la seule
# probabilité ne donne PAS de borne générale d'erreur de prix : le payoff et
# le nombre de coupes comptent aussi. Comparer prix/nœuds/temps pour plusieurs
# seuils avec un arbre complet avant de choisir une précision de calcul.
# Le pas trinomial du dividende cash n'est pas élagué : il est construit en
# entier, comme sans pruning.
