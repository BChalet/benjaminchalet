"""Convergence de l'arbre vers Black-Scholes avec rendement continu q.

Le benchmark Black-Scholes n'est valable ici que sans dividende cash actif.
Les listes et tableaux de ce fichier contiennent des résultats numériques ;
les nœuds de l'arbre restent parcourus par leurs liens dans arbre_binomial.py.
"""

from __future__ import annotations

import datetime as dt
import gc
from math import exp, isfinite, log, sqrt
from numbers import Integral
from pathlib import Path
from statistics import NormalDist
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from arbre_binomial import Market, Model, Option, Tree


class Convergence:

    def __init__(self, market: Market, option: Option, pricing_date: dt.date) -> None:
        if option.exercise_type.lower() != "european":
            raise ValueError("La comparaison à Black-Scholes concerne une option européenne.")
        self.market = market
        self.option = option
        self.pricing_date = pricing_date
        self.results = None
        self.output = Path(__file__).resolve().parent

    def black_price(self, strike: float) -> float:
        """Black-Scholes pour un rendement de dividende continu q.

        Retrancher la valeur actuelle d'un dividende cash au spot dans cette
        formule serait une approximation : ce fichier ne l'utilise pas comme
        référence exacte de convergence.
        """
        dividend_date = self.market.ex_dividend_date
        if (
            self.market.dividend_amount != 0.0
            and dividend_date is not None
            and self.pricing_date < dividend_date <= self.option.maturity
        ):
            raise ValueError(
                "Black-Scholes avec q n'est pas une référence exacte pour un "
                "dividende cash actif. Utiliser les prix de l'arbre et une étude "
                "de stabilité en N, ou annuler dividend_amount pour ce benchmark."
            )

        spot = self.market.spot
        rate = self.market.rate
        sigma = self.market.vol
        q = self.market.dividend_yield
        maturity = (self.option.maturity - self.pricing_date).days / 365.0
        option_type = self.option.type.lower()

        if option_type not in ("call", "put"):
            raise ValueError("Option type must be 'Call' or 'Put'")
        if not all(isfinite(value) for value in (spot, rate, sigma, q, strike)):
            raise ValueError("Les paramètres de Black-Scholes doivent être finis.")
        if maturity < 0.0 or spot <= 0.0 or strike < 0.0 or sigma < 0.0:
            raise ValueError(
                "Il faut une échéance non passée, un spot > 0, un strike >= 0 "
                "et une volatilité >= 0."
            )
        if maturity == 0.0:
            if option_type == "call":
                return max(spot - strike, 0.0)
            return max(strike - spot, 0.0)

        discount = exp(-rate * maturity)
        discounted_spot = spot * exp(-q * maturity)
        # Évite log(S/K) lorsque K = 0.
        if strike == 0.0:
            return discounted_spot if option_type == "call" else 0.0
        if sigma == 0.0:
            forward = spot * exp((rate - q) * maturity)
            if option_type == "call":
                return discount * max(forward - strike, 0.0)
            return discount * max(strike - forward, 0.0)

        standard_deviation = sigma * sqrt(maturity)
        d1 = (
            log(spot / strike) + (rate - q + sigma**2 / 2.0) * maturity
        ) / standard_deviation
        d2 = d1 - standard_deviation
        normal = NormalDist()
        if option_type == "call":
            return discounted_spot * normal.cdf(d1) - strike * discount * normal.cdf(d2)
        return strike * discount * normal.cdf(-d2) - discounted_spot * normal.cdf(-d1)

    def run(self, steps: list[int]) -> None:
        sizes = list(steps)
        if not sizes:
            raise ValueError("Fournir au moins un nombre de pas pour la convergence.")
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in sizes):
            raise ValueError("Chaque nombre de pas doit être un entier strictement positif.")
        black = self.black_price(self.option.strike)
        rows = []
        print("Calcul de la convergence en cours...")

        for n in sizes:
            # Libérer les anciens arbres avant de chronométrer le suivant.
            gc.collect()
            start = perf_counter()
            tree = Tree(Model(self.pricing_date, int(n)), self.option, self.market)
            tree.build()
            price = tree.price()
            elapsed = perf_counter() - start
            rows.append((n, price, price - black, elapsed))
            del tree

        gc.collect()
        self.results = np.array(rows, dtype=float)
        np.savetxt(
            self.output / "convergence_resultats.txt", self.results,
            fmt=["%d", "%.10f", "%.10f", "%.6f"],
            header="N  prix_arbre  ecart_arbre_moins_black  temps_secondes",
        )
        print(f"Prix Black-Scholes avec q : {black:.10f}")
        print(f"Prix arbre à {n} pas : {price:.10f}")
        print(f"Écart : {price - black:.10f}")

    def plot(self) -> None:
        if self.results is None:
            raise ValueError("Exécuter run(steps) avant de tracer la convergence.")
        n = self.results[:, 0]
        prices = self.results[:, 1]
        errors = self.results[:, 2]
        times = self.results[:, 3]
        black = self.black_price(self.option.strike)
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), layout="constrained")

        axes[0].plot(n, prices, ".", label="Prix de l'arbre")
        axes[0].axhline(black, color="orange", label="Black-Scholes avec q")
        axes[0].set(title="1. Convergence du prix", xlabel="Nombre de pas N", ylabel="Prix de l'option")
        axes[0].legend()

        axes[1].plot(n, errors, ".")
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set(title="2. Écart entre l'arbre et Black-Scholes", xlabel="Nombre de pas N", ylabel="Prix arbre − prix Black")

        axes[2].loglog(n, times, ".", label="Temps mesuré")
        axes[2].loglog(n, times[-1] * (n / n[-1])**2, "--", label="Repère proportionnel à N²")
        axes[2].set(title="3. Temps de calcul en log-log", xlabel="Nombre de pas N (log)", ylabel="Temps en secondes (log)")
        axes[2].legend()
        for ax in axes:
            ax.grid(True, alpha=0.25)
        fig.savefig(self.output / "convergence.png", dpi=160)

    def strike_prices(self, strikes: np.ndarray, n: int) -> list[float]:
        """Réutiliser un arbre pour plusieurs strikes, y compris avec cash."""
        option = Option(
            self.option.strike, self.option.maturity,
            self.option.type, self.option.exercise_type,
        )
        tree = Tree(Model(self.pricing_date, n), option, self.market)
        tree.build()
        prices = []
        for strike in strikes:
            option.strike = float(strike)
            prices.append(tree.price())
        del tree
        gc.collect()
        return prices

    def plot_strike(self) -> None:
        # Assignment 1 : faire varier K avec un nombre de pas fixé.
        strikes = np.linspace(80, 120, 161)
        black_prices = [self.black_price(k) for k in strikes]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        fig.suptitle("Effet du strike — mêmes marché, dates et option européenne", fontsize=15)
        axes[0].plot(strikes, black_prices, color="black", label="Black-Scholes avec q")
        for n in (10, 100):
            prices = self.strike_prices(strikes, n)
            axes[0].plot(strikes, prices, label=f"Arbre N = {n}")
            axes[1].plot(strikes, np.array(prices) - black_prices, label=f"N = {n}")
        axes[0].set(title="Prix en fonction du strike", xlabel="Strike K", ylabel="Prix")
        axes[1].set(title="Écart en fonction du strike", xlabel="Strike K", ylabel="Prix arbre − prix Black")
        axes[1].axhline(0, color="black", linewidth=1)
        for ax in axes:
            ax.legend()
            ax.grid(True, alpha=0.25)
        fig.savefig(self.output / "convergence_strike.png", dpi=160)


def main() -> None:
    pricing_date = dt.date(2026, 9, 15)
    maturity = dt.date(2027, 9, 2)
    option = Option(strike=101, maturity=maturity, type="Call", exercise_type="European")
    market = Market(
        spot=100,
        rate=0.03,
        vol=0.20,
        dividend_yield=0.02,  # Rendement continu q ; mettre 0 pour l'exemple initial.
        dividend_amount=0.0,  # Le benchmark BS exige l'absence de cash actif.
        ex_dividend_date=None,
    )
    study = Convergence(market, option, pricing_date)
    # Garder des pas pairs et impairs pour voir les oscillations.
    steps = list(range(1, 101))
    for n in range(150, 1001, 50):
        steps.extend([n - 1, n])
    study.run(steps)
    study.plot()
    study.plot_strike()
    print("Graphiques et résultats enregistrés dans le dossier du script.")
    plt.show()


if __name__ == "__main__":
    main()
