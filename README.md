# Ajout des dividendes à ton arbre

Cette version prolonge tes trois scripts : les classes `Market`, `Model`,
`Option`, `Node`, `TrunkNode`, `Tree`, `DebugTree` et `Convergence` sont conservées.
Les scripts originaux sont fournis à part dans `originaux/`. Le modèle garde
les liens directs `above`, `below`, `next_up`, `next_down` et ajoute `next_mid`.
Aucun tableau, dictionnaire ou liste n'est utilisé pour accéder aux nœuds,
conformément à la page 18 du cours **Dauphine-M2_2026-27_tree_1d_v2.pdf**.

## Utilisation

Python 3.10 ou plus récent, NumPy et Matplotlib :

```bash
python -m pip install -r requirements.txt
python arbre_binomial.py
python exemple_dividendes.py
python debug_arbre.py
python -m unittest -v test_dividendes.py
```

Lancer ces commandes depuis le dossier contenant les scripts. `convergence.py`
produit l'étude de convergence et les graphiques pour un rendement continu.
Sur une machine sans écran, utiliser `MPLBACKEND=Agg` pour enregistrer les figures.
Dans cet environnement cloud, les caches de graphiques doivent être placés dans
un répertoire accessible en écriture, par exemple :

```bash
mkdir -p /tmp/arbres-mpl /tmp/arbres-cache
MPLCONFIGDIR=/tmp/arbres-mpl XDG_CACHE_HOME=/tmp/arbres-cache MPLBACKEND=Agg python exemple_dividendes.py
```

Pour changer les paramètres dans ton propre script :

```python
import datetime as dt
from arbre_binomial import Market, Model, Option, Tree

model = Model(pricing_date=dt.date(2026, 9, 15), nb_steps=1000)
option = Option(strike=101, maturity=dt.date(2027, 9, 2),
                type="Call", exercise_type="European")

# Choisir l'un des deux marchés ci-dessous.
market_continu = Market(spot=100, rate=0.03, vol=0.20,
                        dividend_yield=0.02)       # 2 % par an
market_cash = Market(spot=100, rate=0.03, vol=0.20,
                     dividend_amount=3.0,         # 3 euros
                     ex_dividend_date=dt.date(2027, 3, 15))

tree = Tree(model, option, market_cash).build()
print(tree.price())
```

Les nouveaux arguments sont optionnels : `Market(100, 0.03, 0.20)` conserve
le comportement sans dividendes. Le montant cash et la date ci-dessus sont des
paramètres d'illustration choisis pour l'exemple, pas des données imposées par
la consigne. Pour un put/américain, changer `type`/`exercise_type` puis rappeler
`tree.price()` ; le même arbre du sous-jacent peut être réutilisé. En revanche,
après modification du marché, des dates ou du nombre de pas, créer un nouvel arbre.

## 1. Dividende continu : conserver deux branches

Le paramètre `dividend_yield=q` représente un rendement continu annuel,
et non un montant en euros. Sous la probabilité risque-neutre :

\[
E=S e^{(r-q)h},\qquad
V=E^2\big(e^{\sigma^2h}-1\big),\qquad h=\Delta t.
\]

Dans ton paramétrage, les enfants sont `E * sqrtAlpha` et `E / sqrtAlpha`.
Le facteur de volatilité et les probabilités binomiales ne changent pas.
**L'actualisation reste au taux r**, donc `df = exp(-r*h)`.
Il ne faut pas remplacer r par r-q dans le facteur d'actualisation.

Pour la convergence, Black-Scholes devient :

\[
C=S_0e^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2),\quad
d_1=\frac{\ln(S_0/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt T},\quad
d_2=d_1-\sigma\sqrt T.
\]

`convergence.py` prend désormais q en compte pour les calls et les puts.

## 2. Dividende cash : trois branches sur le pas concerné

Oui, c'est la méthode présentée **page 10 du cours**. Elle conserve un
maillage recombiné et impose de recalculer les probabilités à chaque nœud.
Cela ne signifie pas que tous les modèles possibles de dividendes cash exigent
un trinomial : il existe aussi des méthodes binomiales avec interpolation ou
des arbres non recombinants.

Soustraire D dans ton arbre actuel puis réutiliser automatiquement
`current.below.next_up` casserait les prix des enfants : en notation habituelle,
`(S*u-D)*d` et `(S*d-D)*u` ne sont pas égaux. Une fois le pas de détachement
traité avec trois branches, cette version revient au binomial : le ratio entre
voisins de chaque colonne reste `alpha = sqrtAlpha**2`.

Le détachement est affecté à l'intervalle **(t_i, t_{i+1}]** qui contient sa
date. Si la date tombe entre deux colonnes, le dividende est capitalisé jusqu'à
la fin du pas, sans déplacer arbitrairement sa date. Pour q = 0, les moments
du cours, pages 5 et 10, sont :

\[
E=S e^{rh}-D e^{r(t_{i+1}-t_{ex})},
\qquad
V=(S-wD)^2 e^{2rh}\big(e^{\sigma^2h}-1\big),
\qquad w=\frac{t_{i+1}-t_{ex}}h.
\]

La formule de variance est **l'approximation intra-pas du cours**. Elle ne doit
pas être présentée comme la variance exacte d'un GBM avec saut cash à une date
intérieure quelconque. Si q et D sont tous deux renseignés, le code utilise
la même approximation avec la dérive `r-q` à la place de r ; l'actualisation
reste à r.

Le tronc suivant est placé à son forward après dividende. Pour chaque autre
parent, on choisit le nœud central M le plus proche de son propre E en distance
de prix ; ses voisins sont U = alpha*M et L = M/alpha. Le système à résoudre est :

\[
p_u+p_m+p_d=1,\quad
p_u U+p_m M+p_d L=E,\quad
p_u U^2+p_m M^2+p_d L^2=V+E^2.
\]

Le code emploie des formules centrées, plus stables numériquement :

\[
p_u=\frac{V+(E-M)(E-L)}{(U-M)(U-L)},\quad
p_d=\frac{V+(E-U)(E-M)}{(L-U)(L-M)},\quad p_m=1-p_u-p_d.
\]

Les probabilités sont stockées dans `node.p_up`, `node.p_mid`, `node.p_down`.
La rétropropagation utilise donc :

```python
hold_value = df * (p_up * value_up + p_mid * value_mid + p_down * value_down)
```

Pour une option américaine, on conserve `max(hold_value, intrinsic_value)`.
Les trois branches représentent une approximation de la distribution du
sous-jacent ; le montant du dividende lui-même est connu, pas aléatoire.

## Contrôles et résultats

Les deux Excel fournis sont des références binomiales sans dividendes.
Leur exemple à 2 pas est reproduit : **call européen = 8.3031441273**
(S=100, K=101, r=3 %, vol=20 %, du 15/09/2026 au 02/09/2027).

Les tests couvrent notamment Black-Scholes avec q, les moments des branches,
les dates de détachement, la volatilité nulle et les identités suivantes :

\[
F_T=S_0e^{(r-q)T}-D e^{(r-q)(T-t_{ex})},\qquad
C(K=0)=e^{-rT}F_T,\qquad
C-P=e^{-rT}(F_T-K).
\]

Le terme D ne s'applique que si la date est dans (date de calcul, maturité].
Le spot à la date de calcul est supposé déjà ex-dividende si le détachement a
eu lieu ce jour-là ou auparavant. À maturité, le cash est détaché avant le payoff.
Le contrôle du call de strike nul est aussi recommandé page 21 du cours.
Ton ancien test `K=0 => prix=spot` doit donc changer lorsqu'il y a des dividendes.

Pour l'étude demandée **page 16**, `exemple_dividendes.py` fixe r = 0,
S=100, K=101, vol=20 %, N=200 et compare :

| Marché | Call européen | Call américain | Put européen | Put américain |
|---|---:|---:|---:|---:|
| Sans dividende | 7.380539 | 7.380539 | 8.380539 | 8.380539 |
| Rendement continu q=2 % | 6.412402 | 6.568026 | 9.322687 | 9.322687 |
| Cash D=3 €, ex-date 15/03/2027 | 6.021756 | 6.363722 | 10.021756 | 10.021756 |

Dans ces exemples, les dividendes diminuent le call européen et augmentent le
put européen. Le call américain peut valoir davantage que l'européen grâce à
l'exercice anticipé. Les puts américain et européen sont égaux ici à taux nul.
Les scénarios q=2 % et D=3 € ne correspondent pas au même flux de dividendes.

## Limites à expliquer dans le rendu

- Un seul dividende cash déterministe est géré ; pas un calendrier de plusieurs
  paiements ni un dividende discret proportionnel au spot.
- L'exercice américain est testé aux dates du maillage. L'exercice immédiatement
  avant un détachement situé entre deux dates n'est donc qu'approché ; étudier la
  stabilité en N, surtout pour les calls américains avec cash.
- Le cours précise page 11 que les probabilités trinômiales ne sont pas garanties
  positives. Le code signale un maillage incompatible au lieu de masquer une vraie
  probabilité négative. Il ne corrige que des dépassements d'arrondi de 1e-12.
- Avec de très grands N, des états extrêmement bas peuvent devenir incompatibles
  avec un montant cash fixe : E <= 0 ou probabilités invalides. Cette version n'ajoute
  pas d'élagage. L'exemple D=3 €, ex-date 15/03/2027 est validé à 1 000 pas ; cela
  ne garantit pas tous les nombres de pas ni tous les dividendes.
- Il n'y a pas de benchmark Black-Scholes exact général pour le cash dans ces scripts.
  `convergence.py` refuse donc cette comparaison quand D est actif. Retrancher
  simplement la valeur actuelle de D au spot dans Black-Scholes est une approximation.

Le dessin `debug_arbre.py` est prévu pour 1 à 10 pas ; pour un grand arbre,
utiliser `DebugTree(tree).check()` sans graphique. Après le détachement, une
colonne ne contient plus nécessairement `step+1` nœuds : le débogueur est adapté
à ce changement et aux probabilités locales.
