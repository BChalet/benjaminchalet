# Arbres binomiaux/trinomiaux : valeur, exercice et débogage SVG

Cette version part des deux fichiers **joints le 16 octobre 2026**, `arbre_binomial.py` et `convergence.py`. Les autres fichiers utiles proviennent du dépôt `benjaminchalet`. Les classes et les liens directs de la collègue sont conservés. Les ajouts sont commentés en français dans le code.

Le document de référence est **Dauphine-M2_2026-27_tree_1d_v2.pdf**, extrait du ZIP joint. Les numéros ci-dessous sont ceux imprimés sur les pages du PDF (deux diapositives par page).

- Pages 7–11 : paramétrage binomial et trois probabilités au détachement cash.
- Page 18 : pas de dictionnaire, tableau ou liste pour accéder aux nœuds.
- Pages 21–23 : contrôles de prix, exercice, moyenne actualisée et probabilités par date.
- Pages 34–36 : probabilités d'atteinte, élagage et transitions monomiales.

## Démarrage

Python 3.10 ou plus récent, NumPy et Matplotlib suffisent. Aucun Graphviz ni exécutable externe n'est nécessaire.

```bash
python -m pip install -r requirements.txt
python exemple_complet.py --output-dir resultats
python -m unittest -v test_dividendes test_profils test_debug_arbre
```

`exemple_complet.py` produit les courbes européennes et américaines, leurs données CSV, deux arbres SVG (complet/élagué), et une comparaison du pruning à 1000 pas **sans cash**, afin d’isoler l’élagage binomial. Les graphiques sont enregistrés sans ouvrir de fenêtre. Ajouter `--show` pour les afficher. Pour un essai plus rapide :

```bash
python exemple_complet.py --steps 100 --benchmark-steps 200 --output-dir resultats_rapides
```

Sur un serveur sans écran et avec un dossier personnel non accessible en écriture :

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/arbres-mpl XDG_CACHE_HOME=/tmp/arbres-cache python exemple_complet.py --output-dir resultats
```

Les scripts régénèrent les résultats d'illustration dans le dossier choisi avec `--output-dir`.

## Où lire le code

| Fichier | Rôle |
|---|---|
| `arbre_binomial.py` | Données, construction du sous-jacent, pricing et profils numériques |
| `profils_option.py` | Courbes des pages 22–23 et export des résultats en CSV |
| `debug_arbre.py` | Vérification des liens, masses et moments ; export SVG |
| `exemple_complet.py` | Reproduction des exemples et comparaison des seuils de pruning |
| `convergence.py` | Comparaison européenne à Black-Scholes, sans cash, reprise du fichier joint |
| `test_dividendes.py` | Contrôles existants : Excel, moments, dates, dividendes et parité |
| `test_profils.py` | Contrôles des profils et du premier arrêt |
| `test_debug_arbre.py` | Contrôles du débogueur, corruptions volontaires et SVG |
| `exemple_dividendes.py` | Étude du dividende à taux nul, conservée du dépôt |

## 1. Comprendre la structure : un graphe de nœuds

`Market` décrit le marché : spot S0, taux annuel r, volatilité sigma, rendement continu q, éventuellement un dividende cash D et sa date. `Option` décrit le payoff (call ou put), le strike K, la maturité et le mode d'exercice. `Model` décrit les choix numériques : date initiale, nombre de pas N et seuil de pruning.

Un `Node` représente un état du sous-jacent à une date. `above` et `below` relient deux voisins de la même colonne. `next_up`, `next_mid`, `next_down` relient le parent à ses enfants à la date suivante. Un enfant recombiné peut recevoir plusieurs branches. Il faut donc additionner toutes leurs contributions, et non écraser la précédente.

Un `TrunkNode` est aussi un `Node`, avec `previous_trunk` et `next_trunk` pour changer de date, ainsi que `column_size` pour connaître le nombre réel de nœuds. Le tronc sert de point d'entrée dans la colonne ; il n'est pas forcément le nœud le plus haut.

`iter_columns()` suit `next_trunk`. `iter_column(trunk)` monte par `above` jusqu'au sommet puis redescend par `below`. Les générateurs ne recopient pas les nœuds dans un conteneur. Les listes/tuples de résultats CSV et de coordonnées graphiques ne contiennent que des **scalaires** : ils ne servent pas à accéder aux nœuds.

## 2. Pourquoi construire puis remonter

Le pas vaut `deltaT = (maturité-date_initiale).days / (365*N)`. Le facteur d'actualisation est `df = exp(-r*deltaT)` et la croissance du sous-jacent est `exp((r-q)*deltaT)`. **Le rendement q réduit la dérive, pas le taux d'actualisation.**

`build()` avance de la racine à la maturité. Hors cash, chaque parent a deux enfants. Le tronc monte aux pas impairs puis descend aux pas pairs ; les autres nœuds réutilisent les enfants de leurs voisins pour recombiner. Au pas contenant le cash, trois enfants permettent d'ajuster la somme des probabilités, la moyenne et la variance du cours.

Le nom historique `sqrtAlpha` désigne le multiplicateur binomial autour du forward. Dans les diapositives binomiales, ce multiplicateur est noté alpha. Le champ `tree.alpha = sqrtAlpha**2` représente, lui, le rapport entre deux voisins d'une même colonne. Cette différence de notation est expliquée dans `compute_sqrt_alpha()`.

La masse `node.proba` vaut 1 à la racine. Pour chaque branche parent → enfant :

```python
child.proba += parent.proba * probabilite_de_transition
```

Elle est calculée à la construction, indépendamment de l'exercice de l'option. Une fois le sous-jacent construit, `price()` initialise les valeurs terminales au payoff, puis revient date par date vers la racine :

```python
continuation = df * (p_up * V_up + p_mid * V_mid + p_down * V_down)
```

Une européenne vaut cette continuation. Une américaine peut recevoir le payoff immédiatement. Le moteur mémorise `continuation_value`, `option_value` et `exercise_now` pour rendre la décision explicable dans le débogueur.

Le strike, le type call/put et le mode européen/américain peuvent être changés puis repricés sur le même arbre. Pour changer le marché, la date de calcul, la maturité, N ou le seuil, créer un nouvel arbre : la construction dépend de ces paramètres.

## 3. Convention exacte d'exercice

Avant maturité, `exercise_now` est vrai uniquement pour une américaine dont le payoff est positif et dépasse la continuation de plus de :

```python
1e-12 * max(1.0, abs(payoff), abs(continuation))
```

Dans la bande d'égalité numérique, on conserve la continuation et l'option reste vivante. La valeur stockée suit cette stratégie ; son écart local au maximum mathématique est au plus la tolérance. Cela évite de déclarer des exercices artificiels lorsque les deux valeurs sont égales à l'arrondi près. Cette convention peut modifier le choix d'une date d'exercice entre stratégies économiquement équivalentes.

À maturité, exercice signifie **payoff strictement positif**, pour les européennes comme pour les américaines. Les options sans payoff expirent sans exercice. Ainsi, une option hors de la monnaie n'est jamais comptée comme exercée parce que ses valeurs d'exercice et de continuation valent toutes deux zéro.

## 4. La courbe de valeur demandée dans le PDF

Les pages 22–23 demandent la moyenne par date, exprimée dans le numéraire initial. Si pi(i,n) est `node.proba` et V(i,n) la valeur calculée en remontant l'arbre :

\[
M_i=e^{-rt_i}\sum_n \pi(i,n)V(i,n).
\]

C'est le champ `discounted_mean_value`. Pour une européenne, cette courbe est constante et égale au prix initial, à l'arrondi près. C'est un contrôle de cohérence très utile, y compris avec pruning : il vérifie le prix **du modèle construit**.

Pour une américaine, la moyenne brute actualisée est une supermartingale : elle décroît lorsqu'on donne de la valeur au droit d'exercice. Elle n'intègre pas les flux déjà payés aux détenteurs qui auraient exercé. Il ne faut donc pas attendre une courbe plate.

Une deuxième courbe additionne la valeur des options encore vivantes et les flux déjà payés, tous actualisés. Elle reste égale au prix initial. Elle est exposée sous `total_value` et sert à contrôler simultanément le pricing et les statistiques d'exercice.

## 5. Deux probabilités d'exercice différentes

La courbe du cours est la somme des masses des nœuds où exercer est optimal à chaque date : `exercise_region_probability`. Elle utilise `proba`, sans supprimer les trajectoires déjà exercées. Une trajectoire peut se trouver dans la région d'exercice à plusieurs dates ; **la somme temporelle de cette courbe peut dépasser 100 %**. Ce n'est pas la distribution d'une unique date d'exercice.

Pour obtenir la probabilité de **premier** exercice, chaque nœud possède une seconde masse, `alive_proba`, initialement nulle sauf à la racine. `exercise_profile()` :

1. Appelle `price()` pour disposer de décisions à jour.
2. Réinitialise toutes les masses vivantes, puis met celle de la racine à 1.
3. À chaque date, additionne les masses des nœuds exercés.
4. Absorbe ces masses : elles ne sont pas transmises aux enfants.
5. Transmet les masses des nœuds conservés avec les probabilités locales.

`first_exercise_probability` est cette masse absorbée par date. `cumulative_exercise_probability` en est la somme. `survival_probability` est la masse non exercée **après** la décision de la date ; au dernier pas, elle désigne l'expiration sans paiement.

Toutes ces probabilités sont **risque-neutres**, associées au modèle de valorisation. Elles ne constituent pas une prévision statistique de l'exercice sur le marché réel. Les probabilités ne sont jamais actualisées ; seuls les montants monétaires le sont.

## 6. Tous les champs d'une ligne du profil

`tree.exercise_profile()` renvoie un tuple de `ExerciseProfileRow`, avec une ligne par date, racine et maturité comprises.

| Champ | Sens |
|---|---|
| `step`, `time_years`, `date` | Indice, temps en années ACT/365 et horodatage du maillage |
| `discounted_mean_value` | Moyenne du PDF avec les probabilités brutes |
| `exercise_region_probability` | Masse brute de la zone d'exercice du jour |
| `first_exercise_probability` | Masse exerçant pour la première fois ce jour |
| `cumulative_exercise_probability` | Masse cumulée exercée, paiement du jour inclus |
| `survival_probability` | Masse restant sans exercice après la décision du jour |
| `discounted_alive_value` | Valeur des contrats vivants avant la décision du jour |
| `discounted_exercise_cashflow` | Paiement du jour, actualisé à la date initiale |
| `cumulative_discounted_cashflow` | Paiements actualisés cumulés, jour inclus |
| `total_value` | Valeur vivante avant décision + paiements des dates strictement antérieures |

Pourquoi exclure le flux du jour de `total_value` ? Avant la décision, la valeur vivante contient déjà le payoff du nœud exercé. Ajouter ce même paiement une seconde fois le compterait deux fois.

Les dates intermédiaires peuvent contenir une heure car la durée calendaire ne se divise pas toujours en N jours entiers. Les fichiers CSV conservent cet horodatage. Les probabilités sont enregistrées entre 0 et 1, même lorsque l'axe du graphique les affiche en pourcentage.

Exemple minimal :

```python
import datetime as dt
from arbre_binomial import Market, Model, Option, Tree
from profils_option import OptionProfiles

model = Model(dt.date(2026, 9, 15), 200, pruning_threshold=1e-7)
option = Option(101, dt.date(2027, 9, 2), "Put", "American")
market = Market(100, 0.07, 0.30, dividend_amount=2,
                ex_dividend_date=dt.date(2027, 3, 15))
tree = Tree(model, option, market).build()
rows = tree.exercise_profile()
print("Prix :", tree.root.option_value)
print("Probabilité totale d'exercice :", rows[-1].cumulative_exercise_probability)
OptionProfiles(tree).plot("resultats_personnels", show=False)
```

## 7. Comprendre le pruning fourni

Le pruning de la collègue est conservé : il concerne uniquement les bords des transitions **binomiales**. Si la masse d'un bord est inférieure au seuil, on ne crée pas l'enfant extérieur. Le parent rejoint l'enfant intérieur déjà créé, dans `next_mid`, avec probabilité 1. `node.is_pruned` rend ce choix explicite. Un chemin déterministe dû à sigma=0 n'est pas marqué comme pruning.

Cette redirection conserve la masse : chaque colonne totalise encore 1. Elle change toutefois la moyenne et la variance conditionnelles du nœud. Le débogueur vérifie donc les moments théoriques sur les transitions ordinaires et mesure les biais sur les transitions élaguées. Il contrôle toujours les liens, les probabilités entrantes et la conservation de masse partout.

Un seuil de 1e-7 ne garantit pas une erreur de prix inférieure à 1e-7 : le montant du payoff et le nombre de coupes interviennent aussi. `comparaison_pruning.csv` compare plusieurs seuils au même N. L'écart à l'arbre complet mesure l'effet de l'élagage ; il ne mesure pas toute l'erreur par rapport au modèle continu.

La comparaison à 1000 pas se fait sans cash, avec S0=100, K=101, r=7 %, sigma=30 % et un put américain. Le cas cash de sigma=30 % des profils est valide à 200 pas mais peut produire des probabilités trinômiales négatives aux états extrêmes du graphe complet à 1000 pas ; le benchmark ne force pas ces probabilités.

Les SVG d'illustration utilisent 16 pas et un seuil plus grand (1e-3) pour rendre les coupes visibles. Cette valeur est pédagogique. Le pas trinomial du dividende cash reste construit en entier, comme dans le fichier joint.

## 8. Lire et utiliser le débogueur SVG

```python
from debug_arbre import DebugTree

debugger = DebugTree(tree)
errors = debugger.check()         # Contrôles sans dessin : adapté aux grands N.
# Pour le dessin, utiliser un petit arbre afin que les étiquettes restent lisibles.
small_tree = Tree(Model(model.pricing_date, 16, 1e-3), option, market).build()
DebugTree(small_tree).export_svg("mon_arbre.svg", show_neighbors=True)
```

Ouvrir le SVG dans un navigateur, zoomer et survoler les nœuds pour leurs détails. Le dessin représente les branches réellement présentes, leurs probabilités et les décisions du moteur. Sa légende distingue exercice, élagage et tronc. Des liens de voisinage peuvent être ajoutés pour inspecter la construction. Une limite en nombre de nœuds évite de produire un fichier illisible ou trop lourd ; les contrôles numériques restent utilisables sur les grands arbres.

Les contrôles recherchent notamment les ruptures/cycles de liens, les enfants dans une mauvaise colonne, les probabilités invalides, les masses entrantes incohérentes et les erreurs de moments hors élagage. Un contrôle sans erreur établit ces invariants ; il ne prouve pas à lui seul l'exactitude d'un modèle financier.

Sans pruning effectif, le call européen K=0 doit égaler l'espérance terminale actualisée, dividendes inclus (page 21). Avec pruning, la référence théorique est informative : une différence peut provenir de la redirection connue. Le débogueur ne doit pas l'annoncer comme une mauvaise construction.

## 9. Dividendes et limites conservées

Un seul cash déterministe est géré. Son détachement est affecté à l'intervalle `(t_i,t_{i+1}]` contenant sa date. Le cash est capitalisé jusqu'à la fin du pas. La variance utilisée est l'approximation intra-pas du cours, pas la variance exacte d'un GBM avec saut à une date intérieure quelconque. Le modèle autorise aussi q, avec la dérive r-q.

À date initiale, un dividende déjà détaché est supposé inclus dans le spot ex-dividende. À maturité, le cash est détaché avant le payoff. Un cash postérieur à la maturité est sans effet. Un forward local non positif ou des probabilités trinômiales incompatibles déclenchent une erreur explicite ; le modèle ne tronque pas le cash pour masquer le problème.

L'exercice américain se fait aux dates du maillage. Une décision immédiatement avant un cash situé entre deux dates n'est donc qu'approchée. Il faut étudier la stabilité en N, en particulier pour un call américain à dividende.

Black-Scholes avec q est une référence pour les européennes sans cash. `convergence.py` refuse le cash : retrancher simplement sa valeur actuelle au spot serait une approximation. Les limites K=0 et sigma=0 sont traitées explicitement. Les interprétations générales de convergence sont conservées ; les anciens commentaires contenant des résultats figés ont été remplacés par des explications et les résultats doivent être mesurés sur la machine utilisée.

## 10. Ce qui est vérifié

Les tests existants sont conservés. Les tests ajoutés comparent aussi les premiers exercices à une énumération indépendante des chemins sur un petit arbre : cela détecte le double comptage que manquerait une simple somme de probabilités locales.

Les invariants utiles à retenir à la relecture sont :

- Somme des masses brutes de chaque colonne = 1.
- Cumul de premier exercice + masse non exercée après décision = 1.
- Somme des paiements de premier exercice actualisés = prix initial.
- Valeur vivante actualisée + flux antérieurs actualisés = prix initial à toute date.
- Moyenne brute actualisée européenne constante ; américaine non croissante.

Ces égalités s'appliquent au graphe effectivement construit, complet ou élagué, à l'arrondi numérique près. La comparaison de précision entre les deux graphes reste une étude séparée.

## Résultats vérifiés le 16/10/2026

**34 tests passent** (15 existants, 8 sur les profils, 11 sur le débogueur). Les six SVG de l'exemple complet ont aussi été analysés comme XML. Les courbes PNG ont été inspectées visuellement. Les fonctions et tracés de convergence ont été exécutés sur une petite série de N, ainsi que les limites K=0 et sigma=0.

Pour le put avec cash D=2, S0=100, K=101, r=7 %, sigma=30 %, N=200 et seuil 1e-7, du 15/09/2026 au 02/09/2027 :

| Mode | Prix | Probabilité cumulée d'exercice | Expiration sans paiement |
|---|---:|---:|---:|
| Européen | 9.6613057688 | 50.3310 % | 49.6690 % |
| Américain | 10.4014880890 | 53.8364 % | 46.1636 % |

La somme des paiements actualisés reconstitue le prix et la courbe `total_value` reste constante : l'écart maximal mesuré sur les deux profils est inférieur à 8e-14. Il s'agit d'un contrôle numérique du modèle discret, pas d'une mesure de son erreur par rapport au marché.

Comparaison à N=1000, **sans cash**, avec les autres paramètres du put ci-dessus :

| Seuil | Nœuds | Prix américain | Écart au graphe complet | Temps build + price observé |
|---:|---:|---:|---:|---:|
| 0 | 501501 | 9.5832126015 | 0 | 14.792 s |
| 1e-7 | 105407 | 9.5832127179 | +1.16e-7 | 1.143 s |
| 1e-4 | 70809 | 9.5833469910 | +1.34e-4 | 0.653 s |

Ces temps proviennent d'une exécution dans l'environnement cloud de travail et peuvent varier selon la machine et sa charge. La réduction du nombre de nœuds est plus directement reproductible. Un seuil plus agressif peut gagner du temps en augmentant l'erreur ; ces chiffres ne constituent pas une garantie pour d'autres options ou marchés.
