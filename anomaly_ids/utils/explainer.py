# Implementing SHAP to explain features
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd


class IDSExplainer:
    """
        This class explains pipeline predictions using SHAP and LIME
        The features used will be the original features(used before autoencoder's encoding)
    """

    def __init__(self, pipeline, feature_names, X_train_background):
        self.pipeline = pipeline
        self.feature_names = feature_names
        # SHAP explainer — _predict_fn returns 1-D intrusion probability
        self.shap_explainer = shap.KernelExplainer(
            model = self._predict_fn,
            data = shap.sample(X_train_background, 100)
        )

        # LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data = X_train_background,
            feature_names = feature_names,
            class_names = ['Normal', 'Intrusion'],
            mode = 'classification'
        )

    def _predict_fn(self, X):
        """
            Bridge function for SHAP.
            Returns only the intrusion probability (1 value per row) so that
            KernelExplainer produces a simple 2-D shap_values array instead
            of a list-of-arrays, avoiding index-out-of-bounds on shap_values[1].
        """
        both = self.pipeline._predict_proba_transformed(X)  # shape (n, 2)
        return both[:, 1]  # intrusion probability only, shape (n,)

    def _predict_fn_lime(self, X):
        """
            Bridge function for LIME.
            LIME needs both class probabilities [P(normal), P(intrusion)].
        """
        return self.pipeline._predict_proba_transformed(X)  # shape (n, 2)

    def explain_shap(self, X_sample, top_k = 10):
        """
            SHAP explanation for one or more samples.
            Returns top_k most important features with their SHAP values.
        """
        shap_values = self.shap_explainer.shap_values(X_sample, nsamples = 100)
        # _predict_fn returns a 1-D output (intrusion prob only), so
        # KernelExplainer returns shap_values as a 2-D array: (n_samples, n_features).
        # Row 0 = first (only) sample.
        row = shap_values[0]  # shape: (n_features,)
        top_indices = np.argsort(np.abs(row))[::-1][:int(top_k)]
        return [
            {
                "feature" : self.feature_names[i],
                "shap_value" : float(row[i]),
                "direction" : "toward_intrusion" if row[i] > 0 else "toward_normal"
            }
            for i in top_indices
        ]

    def explain_lime(self, X_sample, top_k = 10):
        """
            LIME Explanantion for a single sample point
            Returns top_k features with their local importance weights
        """
        explanation = self.lime_explainer.explain_instance(
            data_row = X_sample[0],
            predict_fn = self._predict_fn_lime,  # LIME needs [P(normal), P(intrusion)]
            num_features = top_k
        )
        return [
            {
                "feature" : f,
                "weight" : float(w),
                "direction" : "toward_intrusion" if w > 0 else "toward_normal"
            }
            for f,w in explanation.as_list()
        ]
