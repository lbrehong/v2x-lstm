"Ce code transfère les poids du modèle TensorFlow vers le modèle PyTorch afin de garantir que les deux modèles commencent avec les mêmes poids et le même état initial. "
import numpy as np
import torch


def build_tf_model_with_seed(model_type, timesteps, features, seed):
    """Build an untrained Keras model with a given seed, for reproducible transplantation."""
    from learning.seed import set_tf_seed
    from learning.model import build_model
    set_tf_seed(seed)
    return build_model(model_type, timesteps, features)


def extract_tf_weights(keras_model, model_type):
    """
    Extract per-layer weights from a Keras model built by learning.model.build_model.

    Returns:
        Dict with keys 'rnn', 'dense', 'latency_head', 'pdr_head', each
        holding the layer's get_weights() list.
    """
    import keras

    rnn_layer = next(layer for layer in keras_model.layers
                      if isinstance(layer, (keras.layers.LSTM, keras.layers.GRU, keras.layers.SimpleRNN)))
    dense_layer = next(layer for layer in keras_model.layers
                        if isinstance(layer, keras.layers.Dense) and layer.name not in ("latency_ms", "pdr"))
    latency_layer = keras_model.get_layer("latency_ms")
    pdr_layer = keras_model.get_layer("pdr")

    if model_type == "gru" and not rnn_layer.reset_after:
        raise ValueError(
            "GRU layer has reset_after=False; the Keras->PyTorch weight mapping "
            "in this module assumes reset_after=True (the TF2/Keras3 default). "
            "Rebuild the model with a recent Keras/TF version before transplanting."
        )

    return {
        "rnn": rnn_layer.get_weights(),
        "dense": dense_layer.get_weights(),
        "latency_head": latency_layer.get_weights(),
        "pdr_head": pdr_layer.get_weights(),
    }


def _swap_update_reset_blocks(arr):
    """c'est pour gru puisqu'il a un ordre différents des poids entre pytorch et tensorflow"""
    z, r, h = np.split(arr, 3, axis=-1)
    return np.concatenate([r, z, h], axis=-1)


def _to_torch_layout(tf_weights, model_type):
    """
    Convert extracted Keras weights into the exact PyTorch parameter layout
    and gate order, keyed by the RATPredictor parameter names they map to.
    Pure conversion, no model is touched - reused by both
    load_tf_weights_into_torch (to copy) and weight_rel_error (to compare).
    """
    layout = {}
    if model_type in ("lstm", "rnn"):
        kernel, recurrent_kernel, bias = tf_weights["rnn"]
        layout["rnn.weight_ih_l0"] = kernel.T.astype(np.float32)
        layout["rnn.weight_hh_l0"] = recurrent_kernel.T.astype(np.float32)
        layout["rnn.bias_ih_l0"] = bias.astype(np.float32)
        layout["rnn.bias_hh_l0"] = np.zeros_like(bias, dtype=np.float32)
    elif model_type == "gru":
        kernel, recurrent_kernel, bias = tf_weights["rnn"]
        kernel = _swap_update_reset_blocks(kernel)
        recurrent_kernel = _swap_update_reset_blocks(recurrent_kernel)
        bias = _swap_update_reset_blocks(bias)
        layout["rnn.weight_ih_l0"] = kernel.T.astype(np.float32)
        layout["rnn.weight_hh_l0"] = recurrent_kernel.T.astype(np.float32)
        layout["rnn.bias_ih_l0"] = bias[0].astype(np.float32)
        layout["rnn.bias_hh_l0"] = bias[1].astype(np.float32)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    dense_kernel, dense_bias = tf_weights["dense"]
    layout["dense.weight"] = dense_kernel.T.astype(np.float32)
    layout["dense.bias"] = dense_bias.astype(np.float32)

    lat_kernel, lat_bias = tf_weights["latency_head"]
    layout["latency_head.weight"] = lat_kernel.T.astype(np.float32)
    layout["latency_head.bias"] = lat_bias.astype(np.float32)

    pdr_kernel, pdr_bias = tf_weights["pdr_head"]
    layout["pdr_head.weight"] = pdr_kernel.T.astype(np.float32)
    layout["pdr_head.bias"] = pdr_bias.astype(np.float32)
    return layout


def load_tf_weights_into_torch(torch_model, tf_weights, model_type):
    """Copy extracted Keras weights into an equivalent RATPredictor's parameters, in place."""
    layout = _to_torch_layout(tf_weights, model_type)
    params = dict(torch_model.named_parameters())
    with torch.no_grad():
        for name, value in layout.items():
            params[name].copy_(torch.from_numpy(value))


def transplant(tf_model, torch_model, model_type):
    """Extract weights from tf_model and load them into torch_model, in place."""
    weights = extract_tf_weights(tf_model, model_type)
    load_tf_weights_into_torch(torch_model, weights, model_type)


def weight_rel_error(tf_model, torch_model, model_type, eps=1e-8):
    """
    Compare les poids actuels d'un modèle TensorFlow avec ceux d'un modèle
    PyTorch, groupe de paramètres par groupe de paramètres, sans modifier
    aucun des deux modèles. Cette fonction est utilisée pour représenter
    l'évolution de la divergence des poids au cours de l'entraînement lorsque
    les deux modèles ont commencé avec les mêmes poids transférés, puis ont
    été entraînés indépendamment.

    Pour les LSTM/SimpleRNN, les biais `bias_ih_l0` et `bias_hh_l0` sont
    fonctionnellement redondants (seule leur somme intervient dans les
    équations des portes). Ainsi, l'optimiseur peut déplacer une partie de
    la valeur entre ces deux biais pendant l'entraînement, même si lors du
    transfert initial, toute la valeur est placée dans `bias_ih_l0` et
    `bias_hh_l0` est initialisé à zéro.

    Comparer séparément ces deux paramètres avec leur répartition initiale
    (qui n'est valable qu'immédiatement après le transfert) pourrait donc
    indiquer une divergence importante mais qui n'a pas de réelle
    signification. Ces deux paramètres sont donc comparés comme une seule
    quantité : leur somme.

    Pour le GRU, les deux biais ne sont pas redondants de cette manière
    (car ils interviennent différemment dans la formule de la porte de
    réinitialisation) et sont donc comparés séparément, tels quels.

    L'agrégation utilise la MÉDIANE plutôt que la moyenne sur les éléments
    de chaque paramètre. En particulier, les vecteurs de biais contiennent
    souvent plusieurs composantes très proches de zéro. Dans ce cas, la
    formule :

        abs(tf - pt) / max(abs(tf), eps)

    peut produire une très grande valeur pour une différence absolue
    négligeable. Un seul élément de ce type peut alors fortement influencer
    la moyenne et donner l'impression qu'une couche correctement transférée
    a fortement divergé.

    La médiane est plus robuste à ce phénomène tout en permettant de détecter
    une véritable divergence généralisée des paramètres.

    Retourne :
        Un dictionnaire {nom_du_paramètre : erreur_relative_médiane},
        calculée sur les groupes de paramètres correspondants.
    """
    tf_weights = extract_tf_weights(tf_model, model_type)
    layout = _to_torch_layout(tf_weights, model_type)
    params = dict(torch_model.named_parameters())

    errors = {}
    skip = set()
    if model_type in ("lstm", "rnn"):
        tf_bias = layout["rnn.bias_ih_l0"]
        pt_bias_sum = (params["rnn.bias_ih_l0"] + params["rnn.bias_hh_l0"]).detach().cpu().numpy()
        denom = np.maximum(np.abs(tf_bias), eps)
        errors["rnn.bias_ih_l0+bias_hh_l0"] = float(np.median(np.abs(tf_bias - pt_bias_sum) / denom))
        skip = {"rnn.bias_ih_l0", "rnn.bias_hh_l0"}

    for name, tf_value in layout.items():
        if name in skip:
            continue
        pt_value = params[name].detach().cpu().numpy()
        denom = np.maximum(np.abs(tf_value), eps)
        errors[name] = float(np.median(np.abs(tf_value - pt_value) / denom))
    return errors
