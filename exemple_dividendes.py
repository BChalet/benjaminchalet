"""Étude demandée par le cours : effet des dividendes lorsque r = 0.

Lancer : python exemple_dividendes.py
Sorties : impact_dividendes.csv et impact_dividendes.png à côté du script.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt

from arbre_binomial import Market, Model, Option, Tree


def main() -> None:
    pricing_date = dt.date(2026, 9, 15)
    maturity = dt.date(2027, 9, 2)
    ex_date = dt.date(2027, 3, 15)
    nb_steps = 200
    # Ces listes contiennent des paramètres/résultats, jamais des nœuds.
    cases = (
        ("Sans dividende", Market(100, 0.0, 0.20)),
        ("Continu : q = 2 %", Market(100, 0.0, 0.20, dividend_yield=0.02)),
        ("Cash : D = 3 €", Market(100, 0.0, 0.20, dividend_amount=3.0,
                                ex_dividend_date=ex_date)),
    )
    rows = []
    print(f"S=100, K=101, r=0, vol=20 %, N={nb_steps}")
    print(f"Calcul : {pricing_date}; maturité : {maturity}; ex-date cash : {ex_date}\n")
    print(f"{'Marché':<23} {'Call EU':>11} {'Call AM':>11} {'Put EU':>11} {'Put AM':>11}")
    for label, market in cases:
        option = Option(101, maturity, "Call", "European")
        tree = Tree(Model(pricing_date, nb_steps), option, market).build()
        # Revaloriser les quatre contrats sur le même arbre de sous-jacent.
        call_eu = tree.price()
        option.exercise_type = "American"
        call_am = tree.price()
        option.type = "Put"
        option.exercise_type = "European"
        put_eu = tree.price()
        option.exercise_type = "American"
        put_am = tree.price()
        rows.append((label, call_eu, call_am, put_eu, put_am))
        print(f"{label:<23} {call_eu:11.6f} {call_am:11.6f} {put_eu:11.6f} {put_am:11.6f}")

    output = Path(__file__).resolve().parent
    with (output / "impact_dividendes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("marche", "call_europeen", "call_americain", "put_europeen", "put_americain"))
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    labels = [row[0] for row in rows]
    positions = range(len(rows))
    for ax, eu_index, am_index, title in ((axes[0], 1, 2, "Call"), (axes[1], 3, 4, "Put")):
        ax.bar([i-0.18 for i in positions], [row[eu_index] for row in rows],
               width=0.36, label="Européen", color="#2563eb")
        ax.bar([i+0.18 for i in positions], [row[am_index] for row in rows],
               width=0.36, label="Américain", color="#e87924")
        ax.set(title=title, ylabel="Prix de l'option (€)", xticks=list(positions), xticklabels=labels)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend()
    fig.suptitle("Effet des dividendes à taux nul — N = 200, S = 100, K = 101")
    fig.savefig(output / "impact_dividendes.png", dpi=160)
    plt.close(fig)
    print("\nRésultats et graphique enregistrés à côté du script.")
    print("Les montants 2 % et 3 € sont deux scénarios, pas des dividendes équivalents.")


if __name__ == "__main__":
    main()
