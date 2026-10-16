"""Interface Streamlit du projet d'arbres binomiaux et trinomiaux.

Lancer avec : python -m streamlit run app.py

Organisation pour la relecture :
1. Paramètres et formulaire : une saisie n'est appliquée qu'à la validation.
2. Fonctions d'affichage : elles lisent seulement des résultats scalaires.
3. Une fonction par page : les études longues ont un bouton explicite.
4. main() choisit la page ; les autres pages ne sont pas calculées en avance.

Le moteur et ses formules restent dans arbre_binomial.py. interface_calculs
adapte ses résultats à l'écran ; interface_exports prépare les téléchargements.
Ni les caches ni session_state ne contiennent de collections de nœuds.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
import math
import os
import re
from typing import Callable
import xml.etree.ElementTree as ET

# Un serveur web ne doit pas ouvrir de fenêtre Matplotlib, même sous Windows.
# Définir le backend AVANT d'importer les modules de calcul qui utilisent pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from interface_calculs import (
    Parameters, calculate_summary, check_tree, convergence_study, debug_tree,
    dividend_study, has_active_cash, pruning_study, rate_study, strike_study,
)
from interface_exports import profile_archive, scenario_json


PAGES = (
    "Prix & comparaison", "Valeur & exercice", "Arbre & diagnostics",
    "Convergence", "Pruning", "Dividendes & taux", "Guide",
)
COLORS = ("#2563eb", "#0891b2", "#f59e0b", "#8b5cf6", "#059669")
GITHUB = "https://github.com/BChalet/benjaminchalet"
DATE_MIN, DATE_MAX = dt.date(1900, 1, 1), dt.date(2100, 12, 31)


def preset(name: str) -> Parameters:
    """Trois exemples valides, modifiables ensuite champ par champ."""
    base = Parameters(
        pricing_date=dt.date(2026, 9, 15), maturity=dt.date(2027, 9, 2),
        spot=100.0, strike=101.0, rate=0.07, vol=0.30, dividend_yield=0.0,
        dividend_amount=2.0, ex_dividend_date=dt.date(2027, 3, 15),
        nb_steps=200, pruning_threshold=1e-7, option_type="Put", exercise_type="American",
    )
    if name == "Call européen / Black-Scholes":
        return replace(base, rate=0.03, vol=0.20, dividend_amount=0.0,
                       option_type="Call", exercise_type="European", pruning_threshold=0.0)
    if name == "Call américain / taux négatif":
        return replace(base, rate=-0.01, dividend_amount=0.0, option_type="Call")
    return base


def populate_form(params: Parameters) -> None:
    """Mettre des valeurs dans les widgets AVANT leur création au rerun."""
    for name in ("spot", "strike", "nb_steps", "pruning_threshold", "pricing_date", "maturity"):
        st.session_state["param_" + name] = getattr(params, name)
    for name in ("rate", "vol", "dividend_yield"):
        st.session_state["param_" + name] = 100.0 * getattr(params, name)
    st.session_state["param_dividend_amount"] = params.dividend_amount
    st.session_state["param_cash_enabled"] = params.dividend_amount > 0
    st.session_state["param_ex_dividend_date"] = params.ex_dividend_date or params.pricing_date
    st.session_state["param_option_type"] = params.option_type
    st.session_state["param_exercise_type"] = (
        "Américaine" if params.exercise_type == "American" else "Européenne"
    )


def change_preset() -> None:
    populate_form(preset(st.session_state["preset_name"]))


@st.cache_data(max_entries=5, ttl=1800, show_spinner=False)
def cached_summary(params: Parameters) -> dict:
    """Mémoriser des nombres/DataFrames, jamais un Tree et ses cycles."""
    return calculate_summary(params)


@st.cache_data(max_entries=3, ttl=1800, show_spinner=False)
def cached_archive(params: Parameters) -> bytes:
    return profile_archive(params)


def sidebar() -> tuple[Parameters, str]:
    """Un formulaire évite de reconstruire l'arbre à chaque chiffre tapé."""
    if "active_parameters" not in st.session_state:
        st.session_state["active_parameters"] = preset("Put américain / dividende cash")
        populate_form(st.session_state["active_parameters"])
    with st.sidebar:
        st.markdown("### Arbres d’options")
        st.caption("Laboratoire de valorisation · M2 Dauphine")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        st.selectbox("Exemple de départ", (
            "Put américain / dividende cash", "Call européen / Black-Scholes",
            "Call américain / taux négatif",
        ), key="preset_name", on_change=change_preset)
        st.caption("Choisis un exemple, ajuste les champs, puis applique le scénario.")
        with st.form("scenario_form"):
            with st.expander("Marché", expanded=True):
                st.number_input("Spot S₀ (€)", min_value=0.01, key="param_spot", format="%.2f")
                st.number_input("Taux annuel r (%)", min_value=-100.0, max_value=100.0,
                                key="param_rate", step=0.5, format="%.2f",
                                help="Un taux négatif est accepté. 3 signifie 3 % par an.")
                st.number_input("Volatilité σ (%)", min_value=0.0, max_value=300.0,
                                key="param_vol", step=1.0, format="%.2f")
                st.number_input("Rendement continu q (%)", min_value=0.0, max_value=100.0,
                                key="param_dividend_yield", step=0.5, format="%.2f")
            with st.expander("Option & dates", expanded=True):
                st.selectbox("Type d’option", ("Call", "Put"), key="param_option_type",
                             help="Call : droit d’acheter. Put : droit de vendre.")
                st.selectbox("Exercice", ("Européenne", "Américaine"), key="param_exercise_type")
                st.number_input("Strike K (€)", min_value=0.0, key="param_strike", format="%.2f")
                st.date_input("Date de calcul", key="param_pricing_date", min_value=DATE_MIN,
                              max_value=DATE_MAX, format="DD/MM/YYYY")
                st.date_input("Maturité", key="param_maturity", min_value=DATE_MIN,
                              max_value=DATE_MAX, format="DD/MM/YYYY")
            with st.expander("Dividende cash", expanded=False):
                st.checkbox("Activer le dividende cash", key="param_cash_enabled")
                st.number_input("Montant D (€)", min_value=0.0, key="param_dividend_amount",
                                step=0.5, format="%.2f")
                st.date_input("Date de détachement", key="param_ex_dividend_date",
                              min_value=DATE_MIN, max_value=DATE_MAX, format="DD/MM/YYYY")
                st.caption("Un seul cash est géré. Le rendement continu q peut aussi être actif.")
            with st.expander("Maillage & pruning", expanded=False):
                st.number_input("Nombre de pas N", min_value=1, max_value=2000,
                                key="param_nb_steps", step=10)
                st.number_input("Seuil de pruning", min_value=0.0, max_value=1.0,
                                key="param_pruning_threshold", step=1e-7, format="%.8g",
                                help="0 désactive l’élagage. Exemple : 0.0000001 = 1e-7.")
                st.caption("Les grands N demandent plus de temps et de mémoire.")
            apply = st.form_submit_button("Appliquer le scénario", type="primary",
                                          width="stretch")
        if apply:
            candidate = Parameters(
                pricing_date=st.session_state["param_pricing_date"],
                maturity=st.session_state["param_maturity"],
                spot=float(st.session_state["param_spot"]), strike=float(st.session_state["param_strike"]),
                rate=st.session_state["param_rate"] / 100.0,
                vol=st.session_state["param_vol"] / 100.0,
                dividend_yield=st.session_state["param_dividend_yield"] / 100.0,
                dividend_amount=(float(st.session_state["param_dividend_amount"])
                                 if st.session_state["param_cash_enabled"] else 0.0),
                ex_dividend_date=(st.session_state["param_ex_dividend_date"]
                                  if st.session_state["param_cash_enabled"] else None),
                nb_steps=int(st.session_state["param_nb_steps"]),
                pruning_threshold=float(st.session_state["param_pruning_threshold"]),
                option_type=st.session_state["param_option_type"],
                exercise_type=("American" if st.session_state["param_exercise_type"] == "Américaine"
                               else "European"),
            )
            try:
                candidate.validate()
                st.session_state["active_parameters"] = candidate
                st.success("Scénario appliqué.")
            except ValueError as exc:
                st.error(str(exc))
        active = st.session_state["active_parameters"]
        st.download_button("Exporter les paramètres JSON", scenario_json(active),
                           "parametres.json", "application/json", width="stretch")
        st.link_button("Code et documentation", GITHUB, width="stretch")
    return active, page


def plot(fig: go.Figure, key: str) -> None:
    """Même présentation et export vectoriel pour tous les graphiques."""
    fig.update_layout(
        template="plotly_white", colorway=COLORS,
        margin=dict(l=15, r=15, t=45, b=20), hovermode="x unified",
        font=dict(family="Arial, sans-serif", color="#203047"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, key=key, width="stretch",
                    config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})


def line_figure(frame: pd.DataFrame, x: str, series: dict[str, str],
                xlabel: str, ylabel: str, *, probability: bool = False) -> go.Figure:
    fig = go.Figure()
    for column, label in series.items():
        fig.add_trace(go.Scatter(x=frame[x], y=frame[column], name=label, mode="lines+markers",
                                  marker=dict(size=4), line=dict(width=2)))
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel, tickformat=".0%" if probability else None)
    return fig


def csv_download(frame: pd.DataFrame, filename: str, key: str) -> None:
    # UTF-8 avec BOM facilite l'ouverture des accents dans Excel sous Windows.
    st.download_button("Télécharger les données CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                       filename, "text/csv", key=key)


def show_errors(exc: Exception) -> None:
    st.error("Ce calcul n’a pas pu être effectué avec les paramètres choisis.")
    st.info(str(exc))


def study_result(key: str, signature: tuple, compute: Callable, label: str):
    """Ne montrer que les résultats correspondant aux réglages affichés.

    Les boutons de téléchargement et le zoom provoquent des reruns Streamlit.
    session_state conserve donc les résultats numériques du dernier calcul.
    Une autre combinaison de paramètres invalide seulement l'affichage de ce
    résultat ; elle ne lance pas une étude coûteuse à l'insu de l'utilisateur.
    """
    if st.button(label, type="primary", key=key + "_button"):
        try:
            with st.spinner("Calcul en cours…"):
                value = compute()
            st.session_state[key] = (signature, value)
        except (ValueError, ArithmeticError, RuntimeError, OSError) as exc:
            st.session_state.pop(key, None)
            show_errors(exc)
    saved = st.session_state.get(key)
    if saved is not None and saved[0] == signature:
        return saved[1]
    st.caption("Lance le calcul pour ces réglages. Les études ne se recalculent pas à chaque saisie.")
    return None


def valid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Conserver les échecs dans le CSV et les rendre visibles, sans prix fictif."""
    if "status" not in frame:
        return frame
    failed = frame[frame["status"] != "ok"]
    if not failed.empty:
        st.warning(f"{len(failed)} scénario(s) non représentable(s) sur ce maillage.")
        with st.expander("Voir les scénarios et leurs messages"):
            st.dataframe(failed, hide_index=True, width="stretch")
    return frame[frame["status"] == "ok"]


def number_list(text: str, *, integers: bool = False, maximum: int = 20) -> list:
    """Accepter virgules, points-virgules ou espaces comme séparateurs."""
    tokens = [token for token in re.split(r"[,;\s]+", text.strip()) if token]
    if not 1 <= len(tokens) <= maximum:
        raise ValueError(f"Saisir entre 1 et {maximum} valeurs, séparées par des virgules.")
    try:
        values = [int(token) if integers else float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("Utiliser des nombres, avec un point décimal et sans symbole %.") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Les valeurs doivent être finies.")
    return list(dict.fromkeys(values))


def summary(params: Parameters) -> dict | None:
    try:
        with st.spinner("Valorisation du scénario…"):
            return cached_summary(params)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        show_errors(exc)
        return None


def price_page(params: Parameters) -> None:
    st.subheader("Un scénario, quatre contrats")
    st.write("Compare calls et puts, européens et américains, sur le même sous-jacent.")
    result = summary(params)
    if result is None:
        return
    table = result["comparison"].copy()
    same_kind = table[table["option_type"] == params.option_type].set_index("exercise_type")["price"]
    premium = float(same_kind["American"] - same_kind["European"])
    cols = st.columns(4)
    cols[0].metric("Prix du contrat choisi", f"{result['price']:.6f} €")
    cols[1].metric("Prime américaine", f"{premium:.6f} €")
    cols[2].metric("Nœuds construits", f"{result['node_count']:,}".replace(",", " "))
    cols[3].metric("Temps de la synthèse", f"{result['elapsed_seconds']:.3f} s")
    st.caption("La prime compare américain − européen pour le call ou le put sélectionné. "
               "Le temps inclut la construction, les quatre valorisations et le profil.")
    names = {"European": "européen", "American": "américain"}
    table["Contrat"] = table.apply(lambda row: row.option_type + " " + names[row.exercise_type], axis=1)
    colors = [COLORS[0] if row.option_type == params.option_type and row.exercise_type == params.exercise_type
              else "#cbd5e1" for row in table.itertuples()]
    fig = go.Figure(go.Bar(x=table["Contrat"], y=table["price"], marker_color=colors,
                           text=table["price"].map(lambda value: f"{value:.4f} €"), textposition="outside"))
    fig.update_yaxes(title="Prix (€)", rangemode="tozero")
    plot(fig, "comparison_chart")
    if result["black_scholes"] is not None:
        black = result["black_scholes"]
        st.info(f"Black-Scholes européen : {black:.8f} € · Écart arbre − Black-Scholes : "
                f"{result['price'] - black:+.3g} €.")
    else:
        st.caption(result["black_scholes_reason"])
    st.dataframe(table[["Contrat", "price"]].rename(columns={"price": "Prix (€)"}),
                 hide_index=True, width="stretch")
    csv_download(result["comparison"], "comparaison_contrats.csv", "comparison_csv")


def profile_page(params: Parameters) -> None:
    st.subheader("Valeur et exercice, date par date")
    st.caption("Les probabilités sont risque-neutres : elles servent à valoriser le contrat.")
    result = summary(params)
    if result is None:
        return
    frame = result["profile"]
    first, second, third = st.columns(3)
    first.metric("Premier exercice avant maturité", f"{result['early_exercise_probability']:.2%}")
    second.metric("Premier exercice à maturité", f"{result['maturity_exercise_probability']:.2%}")
    third.metric("Expiration sans paiement", f"{result['expiration_probability']:.2%}")
    values, exercises, data = st.tabs(("Valeur moyenne", "Exercice", "Données & exports"))
    with values:
        st.write("La courbe demandée dans le PDF est la moyenne actualisée de la valeur "
                 "sur tous les nœuds de chaque date.")
        fig = line_figure(frame, "time_years", {"discounted_mean_value": "Moyenne du PDF"},
                          "Temps depuis la date de calcul (années)", "Valeur actualisée (€)")
        fig.add_hline(y=result["price"], line_dash="dash", line_color="#64748b",
                      annotation_text="Prix initial")
        plot(fig, "mean_value_chart")
        st.caption("Européenne : moyenne constante. Américaine : moyenne non croissante, "
                   "car ce premier tracé ignore les paiements déjà intervenus.")
        conservation = frame.assign(past_cashflows=(frame["cumulative_discounted_cashflow"]
                                                    - frame["discounted_exercise_cashflow"]))
        plot(line_figure(conservation, "time_years", {
            "discounted_alive_value": "Contrats vivants, avant décision",
            "past_cashflows": "Paiements strictement antérieurs",
            "total_value": "Somme conservée",
        }, "Temps (années)", "Valeur actualisée (€)"), "conservation_chart")
        st.caption(f"Écart maximal de conservation : {result['value_error']:.3g} € · "
                   f"Écart maximal de masse : {result['mass_error']:.3g}.")
    with exercises:
        st.write("La zone d’exercice décrit l’arbre entier. Pour compter chaque contrat "
                 "une seule fois, utilise la distribution du premier exercice.")
        early, terminal = frame.iloc[:-1], frame.iloc[-1]
        fig = line_figure(early, "time_years", {
            "exercise_region_probability": "Zone d’exercice anticipé",
        }, "Temps (années)", "Probabilité", probability=True)
        fig.add_trace(go.Scatter(x=[terminal.time_years], y=[terminal.exercise_region_probability],
                                  name="À maturité : payoff positif", mode="markers",
                                  marker=dict(size=10, symbol="diamond")))
        plot(fig, "region_chart")
        st.caption("Ne pas sommer cette courbe dans le temps : un chemin peut contribuer à plusieurs dates.")
        fig = go.Figure(go.Bar(x=early.time_years, y=early.first_exercise_probability,
                               name="Premier exercice anticipé", marker_color=COLORS[2]))
        fig.add_trace(go.Scatter(x=[terminal.time_years], y=[terminal.first_exercise_probability],
                                  mode="markers", name="Premier exercice à maturité",
                                  marker=dict(size=11, symbol="diamond", color=COLORS[3])))
        fig.update_xaxes(title="Temps (années)")
        fig.update_yaxes(title="Probabilité par pas", tickformat=".1%")
        plot(fig, "first_exercise_chart")
        fig = line_figure(frame, "time_years", {
            "cumulative_exercise_probability": "Exercice cumulé",
        }, "Temps (années)", "Probabilité", probability=True)
        fig.add_trace(go.Scatter(x=early.time_years, y=early.survival_probability,
                                  name="Encore vivant après décision, avant T", mode="lines"))
        fig.add_trace(go.Scatter(x=[terminal.time_years], y=[terminal.survival_probability],
                                  name="À T : expiration sans paiement", mode="markers",
                                  marker=dict(size=11, symbol="x", color="#dc2626")))
        fig.update_yaxes(range=[0, 1.03])
        plot(fig, "cumulative_chart")
    with data:
        st.dataframe(frame, hide_index=True, width="stretch")
        csv_download(frame, "profils_option.csv", "profile_csv")
        archive = study_result("profile_archive", (params,), lambda: cached_archive(params),
                               "Préparer les graphiques PNG/SVG")
        if archive is not None:
            st.download_button("Télécharger les graphiques et données ZIP", archive,
                               "profils_option.zip", "application/zip")


def svg_view(svg: str, zoom: float) -> None:
    """Redimensionner le canevas SVG en gardant son viewBox vectoriel intact.

    Le dessin peut être plus large que l'écran : l'iframe fournit des barres
    de défilement et le curseur change seulement l'échelle d'affichage. Le
    téléchargement conserve le SVG complet avec ses infobulles d'origine.
    """
    root = ET.fromstring(svg)
    root.set("width", str(float(root.get("width", "1000")) * zoom))
    root.set("height", str(float(root.get("height", "600")) * zoom))
    # Préserver le namespace SVG par défaut lors de la sérialisation ElementTree.
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    markup = ET.tostring(root, encoding="unicode")
    html = ('<style>body{margin:0;background:#f8fafc}.viewport{overflow:auto;height:690px;'
            'border:1px solid #dbe3ee;border-radius:10px}</style>'
            '<div class="viewport">' + markup + '</div>')
    # API récente, avec repli pour les versions 1.51 antérieures à st.iframe.
    # Le HTML provient exclusivement du débogueur et des nombres validés ;
    # aucun fichier HTML téléversé n'est inséré dans cet iframe.
    if hasattr(st, "iframe"):
        st.iframe(html, height=710, alt="Arbre SVG avec nœuds et branches")
    else:
        components.html(html, height=710, scrolling=False)


def debug_page(params: Parameters) -> None:
    st.subheader("Inspecter la construction de l’arbre")
    st.write("Survole les nœuds pour lire le payoff, la continuation et les moments. "
             "Les branches orange signalent une redirection de pruning.")
    a, b, c = st.columns(3)
    steps = int(a.number_input("Pas du dessin", 1, 40, 16, key="debug_steps"))
    threshold = float(b.number_input("Seuil du dessin", 0.0, 1.0, 1e-3,
                                      format="%.8g", key="debug_threshold"))
    color_label = c.selectbox("Couleur des nœuds", ("Valeur de l’option", "Sous-jacent", "Probabilité"))
    color = {"Valeur de l’option": "option", "Sous-jacent": "spot", "Probabilité": "probability"}[color_label]
    neighbors = st.checkbox("Afficher les liens entre voisins", value=False)
    compare = st.checkbox("Comparer avec l’arbre sans pruning", value=True)
    st.caption("Le nombre de pas et le seuil ci-dessus sont propres au dessin. "
               "Le marché, l’option et les dates restent ceux du scénario appliqué.")
    drawing_params = replace(params, pruning_threshold=threshold)

    def compute():
        results = {"Dessin choisi": debug_tree(drawing_params, steps, color, neighbors)}
        if compare and threshold > 0:
            results["Sans pruning"] = debug_tree(replace(drawing_params, pruning_threshold=0.0),
                                                  steps, color, neighbors)
        return results

    result = study_result("debug_result", (drawing_params, steps, color, neighbors, compare),
                          compute, "Générer et contrôler les arbres")
    if result is not None:
        zoom = st.slider("Zoom du dessin (%)", 10, 150, 35, step=5)
        for tab, (name, diagnostic) in zip(st.tabs(list(result)), result.items()):
            with tab:
                cols = st.columns(3)
                cols[0].metric("Nœuds", diagnostic["node_count"])
                cols[1].metric("Transitions élaguées", diagnostic["pruned_count"])
                cols[2].metric("Erreurs détectées", diagnostic["errors"])
                if diagnostic["svg"]:
                    svg_view(diagnostic["svg"], zoom / 100.0)
                    st.download_button("Télécharger le SVG complet", diagnostic["svg"],
                                       "arbre_" + ("complet" if name == "Sans pruning" else "choisi") + ".svg",
                                       "image/svg+xml", key="svg_" + name)
                else:
                    st.error("Le dessin est suspendu : les contrôles ont trouvé une incohérence. Voir le rapport.")
                with st.expander("Rapport des contrôles"):
                    st.code(diagnostic["report"], language="text")
    st.divider()
    st.write(f"Contrôler aussi le scénario principal à N = {params.nb_steps}, sans dessiner tous les nœuds.")
    check = study_result("main_check", (params,), lambda: check_tree(params),
                         "Contrôler le scénario principal")
    if check is not None:
        if check["errors"] == 0:
            st.success(f"Aucune erreur de construction détectée sur {check['node_count']} nœuds.")
        else:
            st.error(f"{check['errors']} erreur(s) détectée(s).")
        st.code(check["report"], language="text")


def reference_params(params: Parameters) -> Parameters | None:
    """Une comparaison Black-Scholes doit expliciter tout changement de contrat."""
    st.caption("Comparaison européenne, avec rendement continu q ; pruning désactivé pour isoler N.")
    if has_active_cash(params):
        ignore_cash = st.checkbox("Faire cette étude sans le dividende cash", value=False,
                                  key="convergence_ignore_cash")
        if not ignore_cash:
            st.info("Black-Scholes n’est pas une référence exacte avec un cash actif. "
                    "Coche la case pour étudier explicitement le scénario sans cash.")
            return None
        return replace(params, dividend_amount=0.0, ex_dividend_date=None)
    return params


def convergence_page(params: Parameters) -> None:
    st.subheader("Convergence vers Black-Scholes")
    reference = reference_params(params)
    if reference is None:
        return
    mode = st.radio("Étude", ("Nombre de pas N", "Strike K"), horizontal=True, key="convergence_mode")
    if mode == "Nombre de pas N":
        text = st.text_input("N à comparer (20 valeurs maximum)", "5,10,20,50,100,200,400")
        result = study_result("convergence_result", (reference, text),
                              lambda: convergence_study(reference, number_list(text, integers=True)),
                              "Calculer la convergence en N")
        if result is None:
            return
        good = valid_rows(result)
        if not good.empty:
            good = good.sort_values("nb_steps")
            plot(line_figure(good, "nb_steps", {"price": "Arbre", "reference": "Black-Scholes"},
                             "Nombre de pas N", "Prix (€)"), "convergence_prices")
            a, b = st.columns(2)
            with a:
                plot(line_figure(good, "nb_steps", {"error": "Écart brut"}, "N", "Arbre − Black (€)"),
                     "convergence_error")
            with b:
                plot(line_figure(good, "nb_steps", {"scaled_error": "Écart × N"}, "N", "Écart × N"),
                     "convergence_scaled")
            fig = line_figure(good, "nb_steps", {"seconds": "Temps observé"}, "N", "Secondes")
            last = good.iloc[-1]
            fig.add_trace(go.Scatter(x=good.nb_steps,
                                      y=last.seconds * (good.nb_steps / last.nb_steps) ** 2,
                                      name="Repère N²", line=dict(dash="dash")))
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")
            plot(fig, "convergence_time")
        st.dataframe(result, hide_index=True, width="stretch")
        csv_download(result, "convergence.csv", "convergence_csv")
    else:
        left, middle, right = st.columns(3)
        minimum = float(left.number_input("Strike minimum", min_value=0.0, value=max(0.0, params.strike * 0.8)))
        maximum = float(middle.number_input("Strike maximum", min_value=0.0, value=max(1.0, params.strike * 1.2)))
        count = int(right.number_input("Nombre de strikes", 2, 81, 31))
        text = st.text_input("N à comparer (4 valeurs maximum)", "10,100", key="strike_steps")

        def compute():
            if maximum <= minimum:
                raise ValueError("Le strike maximum doit être supérieur au minimum.")
            return strike_study(reference, np.linspace(minimum, maximum, count).tolist(),
                                number_list(text, integers=True, maximum=4))

        result = study_result("strike_result", (reference, minimum, maximum, count, text), compute,
                              "Calculer l’écart selon le strike")
        if result is not None:
            fig = go.Figure()
            for n, group in result.groupby("nb_steps"):
                fig.add_trace(go.Scatter(x=group.strike, y=group.error, name=f"N = {n}", mode="lines"))
            fig.add_hline(y=0, line_color="#94a3b8")
            fig.update_xaxes(title="Strike K (€)")
            fig.update_yaxes(title="Arbre − Black-Scholes (€)")
            plot(fig, "strike_error")
            st.dataframe(result, hide_index=True, width="stretch")
            csv_download(result, "convergence_strike.csv", "strike_csv")


def pruning_page(params: Parameters) -> None:
    st.subheader("Mesurer l’effet du pruning")
    st.write("Compare les seuils au même nombre de pas. Le seuil 0 fournit la référence sans élagage.")
    steps = int(st.number_input("N de la comparaison", 1, 2000, min(params.nb_steps, 1000), key="pruning_steps"))
    text = st.text_input("Seuils à comparer", "0,1e-7,1e-5,1e-3")
    without_cash = st.checkbox("Faire cette comparaison sans cash", value=False, key="pruning_without_cash")
    target = replace(params, dividend_amount=0.0, ex_dividend_date=None) if without_cash else params
    result = study_result("pruning_result", (target, text, steps),
                          lambda: pruning_study(target, number_list(text), steps), "Comparer les seuils")
    if result is None:
        return
    good = valid_rows(result)
    if not good.empty:
        display = good.assign(label=good.threshold.map(lambda value: f"{value:g}"))
        a, b = st.columns(2)
        with a:
            fig = go.Figure(go.Bar(x=display.label, y=display.node_count, marker_color=COLORS[0]))
            fig.update_xaxes(title="Seuil de pruning", type="category")
            fig.update_yaxes(title="Nombre de nœuds")
            plot(fig, "pruning_nodes")
        with b:
            fig = go.Figure(go.Bar(x=display.label, y=display.seconds, marker_color=COLORS[1]))
            fig.update_xaxes(title="Seuil de pruning", type="category")
            fig.update_yaxes(title="Construction + pricing (s)")
            plot(fig, "pruning_times")
        fig = go.Figure(go.Bar(x=display.label, y=display.price_gap, marker_color=COLORS[2]))
        fig.update_xaxes(title="Seuil de pruning", type="category")
        fig.update_yaxes(title="Écart de prix au seuil 0 (€)", tickformat=".2e")
        plot(fig, "pruning_gap")
    st.caption("Un seuil sur la probabilité ne garantit pas une erreur de prix maximale. "
               "Si l’arbre complet échoue, aucun écart de référence n’est inventé. Les temps dépendent de la machine.")
    st.dataframe(result, hide_index=True, width="stretch")
    csv_download(result, "comparaison_pruning.csv", "pruning_csv")


def four_price_chart(frame: pd.DataFrame, x: str, xlabel: str, key: str) -> None:
    good = valid_rows(frame)
    if good.empty:
        return
    plot(line_figure(good, x, {
        "call_european": "Call européen", "call_american": "Call américain",
        "put_european": "Put européen", "put_american": "Put américain",
    }, xlabel, "Prix (€)"), key)


def dividend_rate_page(params: Parameters) -> None:
    st.subheader("Dividendes, taux et exercice anticipé")
    dividends, rates = st.tabs(("Effet des dividendes", "Américain − européen / taux"))
    with dividends:
        rate_zero = st.checkbox("Fixer le taux à 0 % pour cette étude (cas du PDF)", value=True)
        target = replace(params, rate=0.0) if rate_zero else params
        mode = st.radio("Paramètre étudié", ("Cash D", "Rendement continu q"), horizontal=True, key="dividend_mode")
        if mode == "Cash D":
            text = st.text_input("Montants cash en euros", "0,1,2,3,4")
            # Même si le cash du formulaire est désactivé, l'étude doit
            # disposer d'une date explicite pour ses montants positifs.
            date = st.date_input("Détachement de l’étude", value=params.ex_dividend_date or
                                 (params.pricing_date + (params.maturity - params.pricing_date) // 2),
                                 min_value=DATE_MIN, max_value=DATE_MAX, format="DD/MM/YYYY")
            target = replace(target, ex_dividend_date=date)
            st.caption(f"q reste fixé à {target.dividend_yield:.2%}. Le montant 0 signifie sans cash.")
            if not target.pricing_date < date <= target.maturity:
                st.info("Ce détachement est hors de la vie du contrat : les montants cash étudiés seront sans effet.")
            result = study_result("dividend_result", (target, text),
                                  lambda: dividend_study(target, number_list(text, maximum=15)),
                                  "Comparer les dividendes cash")
            x, xlabel = "dividend_amount", "Dividende cash D (€)"
        else:
            text = st.text_input("Rendements continus en %", "0,1,2,3,4", key="yield_list")
            target = replace(target, dividend_amount=0.0, ex_dividend_date=None)
            st.caption("Cette étude fait varier q, sans dividende cash.")

            def compute():
                rows = []
                for q_percent in number_list(text, maximum=15):
                    row = dividend_study(replace(target, dividend_yield=q_percent / 100.0), [0.0]).iloc[0].to_dict()
                    row["yield_percent"] = q_percent
                    rows.append(row)
                return pd.DataFrame(rows)

            result = study_result("yield_result", (target, text), compute, "Comparer les rendements continus")
            x, xlabel = "yield_percent", "Rendement continu q (%)"
        if result is not None:
            four_price_chart(result, x, xlabel, "dividend_chart")
            st.dataframe(result, hide_index=True, width="stretch")
            csv_download(result, "impact_dividendes.csv", "dividend_csv")
    with rates:
        text = st.text_input("Taux annuels à comparer, en %", "-3,-1,0,1,3,5,7")
        result = study_result("rate_result", (params, text),
                              lambda: rate_study(params, [value / 100.0 for value in number_list(text, maximum=15)]),
                              "Comparer les taux et les primes américaines")
        if result is not None:
            frame = result.assign(rate_percent=result.rate * 100.0)
            four_price_chart(frame, "rate_percent", "Taux annuel r (%)", "rate_prices")
            good = valid_rows(frame)
            if not good.empty:
                plot(line_figure(good, "rate_percent", {
                    "call_premium": "Prime américaine du call", "put_premium": "Prime américaine du put",
                }, "Taux annuel r (%)", "Américain − européen (€)"), "rate_premium")
            st.dataframe(frame, hide_index=True, width="stretch")
            csv_download(frame, "impact_taux.csv", "rate_csv")


def guide_page(params: Parameters) -> None:
    st.subheader("Prendre en main le laboratoire")
    st.write("1. Choisis un exemple dans la colonne de gauche. 2. Modifie ses paramètres et "
             "applique le scénario. 3. Choisis une page et lance les études qui t’intéressent.")
    st.dataframe(pd.DataFrame([
        ("Prix & comparaison", "Prix du contrat, quatre prix call/put européen/américain et référence Black-Scholes."),
        ("Valeur & exercice", "Moyenne actualisée, premier exercice, cumul, survie et conservation de valeur."),
        ("Arbre & diagnostics", "SVG avec zoom, liens, exercice, pruning et contrôles numériques."),
        ("Convergence", "Écart à Black-Scholes selon N ou le strike, ordre d’erreur et temps."),
        ("Pruning", "Prix, nombre de nœuds et durée selon le seuil d’élagage."),
        ("Dividendes & taux", "Effet du cash, du rendement continu et des taux positifs/négatifs."),
    ], columns=["Page", "Ce qu’elle permet d’étudier"]), hide_index=True, width="stretch")
    with st.expander("Unités et conventions", expanded=True):
        st.write("Les taux et la volatilité sont saisis en pourcentage : 3 signifie 3 %. "
                 "Les montants sont en euros. Le temps est calculé en ACT/365. "
                 "Une américaine peut exercer aux dates du maillage ; une européenne à maturité seulement.")
        st.write("À maturité, un payoff strictement positif compte comme exercice. "
                 "Avant maturité, une égalité numérique conduit à conserver le contrat.")
    with st.expander("Dividendes et limites du modèle"):
        st.write("Le moteur gère un rendement continu et un seul cash déterministe. "
                 "Le pas de détachement utilise trois branches ; les autres pas sont binomiaux. "
                 "La variance du pas cash utilise l’approximation du cours.")
        st.write("Certains paramètres cash produisent un forward non positif ou des probabilités "
                 "trinômiales négatives. L’écran explique alors l’échec ; il ne force pas un prix. "
                 "Pour Black-Scholes, utilise un scénario européen sans cash actif.")
    with st.expander("Ce que mesure le pruning"):
        st.write("Une coupe redirige la masse du bord vers un enfant intérieur avec probabilité 1. "
                 "La masse est conservée mais les moments locaux changent. Compare plusieurs seuils "
                 "au même N pour mesurer l’effet sur le prix ; aucun seuil ne garantit seul une précision.")
    with st.expander("Lancer sur Windows / VS Code"):
        st.write("Utilise le terminal avec le Python dans lequel les dépendances sont installées.")
        st.code('& "$env:LOCALAPPDATA\\Programs\\Python\\Python313\\python.exe" -m pip install -r requirements.txt\n'
                '& "$env:LOCALAPPDATA\\Programs\\Python\\Python313\\python.exe" -m streamlit run app.py',
                language="powershell")
        st.write("Le navigateur ouvre normalement http://localhost:8501. "
                 "Garde le terminal ouvert ; Ctrl+C arrête le serveur. Les lanceurs Windows fournis "
                 "effectuent aussi ces étapes.")
    st.caption("Les grecs et le choix automatique de N selon une précision ne font pas encore partie du moteur.")
    st.link_button("Lire le guide détaillé sur GitHub", GITHUB + "#readme")


def main() -> None:
    st.set_page_config(page_title="Arbres d’options · Laboratoire", page_icon="📈", layout="wide")
    st.markdown("""<style>
    .block-container{max-width:1500px;padding-top:2rem;padding-bottom:3rem}
    [data-testid="stMetric"]{background:#f8fafc;border:1px solid #e2e8f0;
      border-radius:12px;padding:16px 18px}
    .hero{background:#10223a;color:#fff;padding:28px 32px;border-radius:16px;margin-bottom:18px}
    .hero .eyebrow{font-size:12px;letter-spacing:.12em;color:#8ac8ff;text-transform:uppercase}
    .hero h1{font-size:32px;line-height:1.2;margin:8px 0 10px;color:#fff}
    .hero p{font-size:15px;margin:0;color:#d7e3f1}
    </style>""", unsafe_allow_html=True)
    params, page = sidebar()
    st.markdown('<div class="hero"><div class="eyebrow">Binomial · Trinomial · Exercice optimal</div>'
                '<h1>Explore ton arbre d’options</h1>'
                '<p>Modifie un scénario, observe le prix et comprends chaque décision d’exercice.</p></div>',
                unsafe_allow_html=True)
    exercise = "américain" if params.exercise_type == "American" else "européen"
    st.caption(f"SCÉNARIO APPLIQUÉ · {params.option_type} {exercise} · S₀ {params.spot:g} € · "
               f"K {params.strike:g} € · r {params.rate:.2%} · σ {params.vol:.2%} · "
               f"q {params.dividend_yield:.2%} · D {params.dividend_amount:g} € · "
               f"N {params.nb_steps} · {params.pricing_date:%d/%m/%Y} → {params.maturity:%d/%m/%Y}")
    if params.dividend_amount > 0 and not has_active_cash(params):
        st.caption("Le dividende cash est hors de la vie du contrat : il n’intervient pas dans cette valorisation.")
    pages = dict(zip(PAGES, (price_page, profile_page, debug_page, convergence_page,
                             pruning_page, dividend_rate_page, guide_page)))
    pages[page](params)


if __name__ == "__main__":
    main()
