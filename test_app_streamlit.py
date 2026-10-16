"""Parcours de l'interface avec le navigateur simulé officiel de Streamlit.

Ces tests vérifient le branchement écran/calcul : scénario appliqué après un
formulaire, navigation, lancement des études et téléchargements. Les formules
financières ont leurs propres tests indépendants dans les autres fichiers.
Lancement : python -m unittest -v test_app_streamlit
"""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().with_name("app.py")


class StreamlitTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(APP), default_timeout=60).run()

    def clean(self, app):
        self.assertEqual([item.message for item in app.exception], [])

    def navigate(self, app, page):
        app.sidebar.radio[0].set_value(page).run()
        self.clean(app)

    def test_default_scenario_and_all_pages(self):
        app = self.app()
        self.clean(app)
        self.assertEqual(app.number_input(key="param_spot").value, 100.0)
        self.assertAlmostEqual(app.number_input(key="param_rate").value, 7.0)
        self.assertEqual(app.metric[0].value, "10.401488 €")
        for page in app.sidebar.radio[0].options:
            with self.subTest(page=page):
                self.navigate(app, page)

    def test_preset_and_form_apply_actual_parameters(self):
        app = self.app()
        app.selectbox(key="preset_name").set_value("Call européen / Black-Scholes").run()
        # Choisir l'exemple remplit le formulaire, sans changer le prix actif.
        self.assertEqual(app.session_state["active_parameters"].option_type, "Put")
        app.number_input(key="param_nb_steps").set_value(2)
        submit = next(button for button in app.button if button.label == "Appliquer le scénario")
        submit.click().run()
        self.clean(app)
        self.assertEqual(app.session_state["active_parameters"].option_type, "Call")
        self.assertEqual(app.session_state["active_parameters"].nb_steps, 2)
        self.assertAlmostEqual(float(app.metric[0].value.split()[0]), 8.303144, places=6)
        # Une maturité invalide affiche un message et garde le dernier scénario.
        app.date_input(key="param_maturity").set_value(app.date_input(key="param_pricing_date").value)
        next(button for button in app.button if button.label == "Appliquer le scénario").click().run()
        self.clean(app)
        self.assertTrue(app.error)
        self.assertEqual(app.session_state["active_parameters"].nb_steps, 2)

    def test_debug_controls_and_profile_download(self):
        app = self.app()
        self.navigate(app, "Arbre & diagnostics")
        app.button(key="debug_result_button").click().run()
        self.clean(app)
        self.assertEqual([metric.value for metric in app.metric if metric.label == "Erreurs détectées"], ["0", "0"])
        self.assertEqual(len(app.get("download_button")), 3)  # Paramètres + deux SVG.
        app.slider[0].set_value(50).run()
        self.clean(app)
        self.assertEqual(app.metric[0].value, "140")
        self.navigate(app, "Valeur & exercice")
        app.button(key="profile_archive_button").click().run()
        self.clean(app)
        self.assertEqual(len(app.get("download_button")), 3)  # Paramètres, CSV, ZIP.

    def test_studies_buttons_and_stale_results(self):
        app = self.app()
        self.navigate(app, "Convergence")
        self.assertTrue(app.info)  # La référence est indisponible avec cash actif.
        app.checkbox(key="convergence_ignore_cash").check().run()
        app.text_input[0].set_value("2,5,10")
        app.button(key="convergence_result_button").click().run()
        self.clean(app)
        self.assertEqual(len(app.dataframe), 1)
        app.text_input[0].set_value("3,6").run()
        self.assertEqual(len(app.dataframe), 0)  # Ne pas afficher l'ancienne série.
        app.radio(key="convergence_mode").set_value("Strike K").run()
        app.button(key="strike_result_button").click().run()
        self.clean(app)
        self.assertEqual(len(app.dataframe), 1)
        self.navigate(app, "Pruning")
        app.button(key="pruning_result_button").click().run()
        self.clean(app)
        self.assertEqual(len(app.dataframe), 1)
        self.navigate(app, "Dividendes & taux")
        app.button(key="dividend_result_button").click().run()
        self.clean(app)
        app.button(key="rate_result_button").click().run()
        self.clean(app)
        self.assertEqual(len(app.dataframe), 2)
        app.radio(key="dividend_mode").set_value("Rendement continu q").run()
        app.button(key="yield_result_button").click().run()
        self.clean(app)
        self.assertEqual(len(app.dataframe), 2)


if __name__ == "__main__":
    unittest.main()
