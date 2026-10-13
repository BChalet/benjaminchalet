import numpy as np
import datetime as dt
import gc
from time import perf_counter
from statistics import NormalDist
from pathlib import Path
import matplotlib.pyplot as plt

# On importe les classes du fichier arbre_binomial.py
from arbre_binomial import Market, Model, Option, Tree

# CONVERGENCE

class Convergence:

    def __init__(self,market: Market,option: Option,pricing_date: dt.date) -> None:
        if option.exercise_type.lower() != "european":
            raise ValueError("La comparaison à Black-Scholes concerne une option européenne.")
        self.market = market
        self.option = option
        self.pricing_date = pricing_date
        self.results = None
        self.output = Path(__file__).resolve().parent

    def black_price(self,strike: float) -> float:
        # Black-Scholes sans dividende, équivalent à Black avec F = S * exp(rT).
        S = self.market.spot
        r = self.market.rate
        sigma = self.market.vol
        T = (self.option.maturity - self.pricing_date).days / 365.0
        if T <= 0 or S <= 0 or strike <= 0 or sigma < 0:
            raise ValueError("Il faut T, spot et strike positifs, et une volatilité >= 0.")

        if sigma == 0:
            forward = S * np.exp(r*T)
            if self.option.type.lower() == "call":
                return np.exp(-r*T) * max(forward - strike,0)
            return np.exp(-r*T) * max(strike - forward,0)

        d1 = (np.log(S/strike) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        normal = NormalDist()

        if self.option.type.lower() == "call":
            return S*normal.cdf(d1) - strike*np.exp(-r*T)*normal.cdf(d2)
        elif self.option.type.lower() == "put":
            return strike*np.exp(-r*T)*normal.cdf(-d2) - S*normal.cdf(-d1)
        else:
            raise ValueError("Option type must be 'Call' or 'Put'")

    def run(self,steps: list[int]) -> None:
        black = self.black_price(self.option.strike)
        rows = []
        print("Calcul de la convergence en cours...")

        for n in steps:
            # Libérer les anciens arbres avant de chronométrer le suivant.
            gc.collect()
            start = perf_counter()
            model = Model(self.pricing_date,n)
            tree = Tree(model,self.option,self.market)
            tree.build()
            price = tree.price()
            elapsed = perf_counter() - start

            rows.append((n,price,price-black,elapsed))
            del tree

        gc.collect()
        self.results = np.array(rows)
        # Conserver les résultats dans un fichier, sans afficher tout le tableau.
        np.savetxt(self.output / "convergence_resultats.txt",self.results,
            fmt=["%d","%.10f","%.10f","%.6f"],
            header="N  prix_arbre  ecart_arbre_moins_black  temps_secondes")

        print(f"Prix Black-Scholes : {black:.10f}")
        print(f"Prix arbre à {n} pas : {price:.10f}")
        print(f"Écart : {price-black:.10f}")

    def plot(self) -> None:
        n = self.results[:,0]
        prices = self.results[:,1]
        errors = self.results[:,2]
        times = self.results[:,3]
        black = self.black_price(self.option.strike)

        fig,axes = plt.subplots(3,1,figsize=(10,12),layout="constrained")

        # 1. Le prix se rapproche-t-il du prix Black-Scholes ?
        axes[0].plot(n,prices,".",label="Prix de l'arbre")
        axes[0].axhline(black,color="orange",label="Black-Scholes")
        axes[0].set(title="1. Convergence du prix",xlabel="Nombre de pas N",ylabel="Prix de l'option")
        axes[0].legend()

        # 2. L'écart se rapproche-t-il de zéro ?
        axes[1].plot(n,errors,".")
        axes[1].axhline(0,color="black",linewidth=1)
        axes[1].set(title="2. Écart entre l'arbre et Black-Scholes",xlabel="Nombre de pas N",ylabel="Prix arbre − prix Black")

        # 3. Le temps augmente-t-il comme N² ?
        # On mesure construction + pricing, une fois par taille d'arbre.
        axes[2].loglog(n,times,".",label="Temps mesuré")
        axes[2].loglog(n,times[-1]*(n/n[-1])**2,"--",label="Repère proportionnel à N²")
        axes[2].set(title="3. Temps de calcul en log-log",xlabel="Nombre de pas N (log)",ylabel="Temps en secondes (log)")
        axes[2].legend()

        for ax in axes:
            ax.grid(True,alpha=0.25)
        fig.savefig(self.output / "convergence.png",dpi=160)

    # Calculer les prix pour plusieurs strikes sur un même arbre.
    def strike_prices(self,strikes: np.ndarray,n: int) -> list[float]:
        option = Option(self.option.strike,self.option.maturity,
                        self.option.type,self.option.exercise_type)
        tree = Tree(Model(self.pricing_date,n),option,self.market)
        tree.build()
        prices = []
        for k in strikes:
            option.strike = k
            prices.append(tree.price())
        # L'option d'origine n'a pas été modifiée.
        del tree
        gc.collect()
        return prices

    def plot_strike(self) -> None:
        # Assignment 1 / page 28 : faire varier K avec un nombre de pas fixé.
        # Une option séparée permet de conserver le strike de l'exemple initial.
        strikes = np.linspace(80,120,161)
        black_prices = [self.black_price(k) for k in strikes]
        fig,axes = plt.subplots(1,2,figsize=(13,5),layout="constrained")
        fig.suptitle("Effet du strike — mêmes marché, dates et option européenne",fontsize=15)
        axes[0].plot(strikes,black_prices,color="black",label="Black-Scholes")

        for n in (10,100):
            prices = self.strike_prices(strikes,n)
            axes[0].plot(strikes,prices,label=f"Arbre N = {n}")
            axes[1].plot(strikes,np.array(prices)-black_prices,label=f"N = {n}")

        axes[0].set(title="Prix en fonction du strike",xlabel="Strike K",ylabel="Prix")
        axes[1].set(title="Écart en fonction du strike",xlabel="Strike K",ylabel="Prix arbre − prix Black")
        axes[1].axhline(0,color="black",linewidth=1)
        for ax in axes:
            ax.legend()
            ax.grid(True,alpha=0.25)
        fig.savefig(self.output / "convergence_strike.png",dpi=160)


# MAIN

def main() -> None:

    # Paramètres de l'exemple : à modifier ici pour un autre cas.
    pricing_date = dt.date(2026, 9, 15)
    maturity = dt.date(2027, 9, 2)

    option = Option(
        strike=101,
        maturity=maturity,
        type="Call",
        exercise_type="European"
    )

    market = Market(
        spot=100,
        rate=0.03,
        vol=0.20
    )

    study = Convergence(market,option,pricing_date)

    # Tous les N de 1 à 100, puis des couples voisins : 149/150, 199/200...
    # Garder des pas pairs ET impairs pour voir les oscillations.
    steps = list(range(1,101))
    for n in range(150,1001,50):
        steps.extend([n-1,n])

    study.run(steps)
    study.plot()
    study.plot_strike()
    print("Graphiques et résultats enregistrés dans le dossier du script.")
    plt.show()


if __name__ == "__main__":
    main()

#note sur les résultats
#augmenter le nombre de pas améliore globalement la précision du prix, 
# avec des oscillations autour de Black-Scholes, 
# mais augmente fortement le coût de calcul. 
# Les deux premiers graphiques montrent le gain de précision ; le troisième montre son coût.