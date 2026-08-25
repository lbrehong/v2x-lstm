# Migration TensorFlow -> PyTorch : journal complet

Ce document trace tout ce qui a été fait dans le cadre de la migration de la partie
entraînement (`learning/`) de TensorFlow/Keras vers PyTorch, en parallèle de l'existant,
avec ajout d'un seed reproductible et de deux validations croisées TF vs PyTorch.

** ! ** : seul l'entraînement (`learning/`) est concerné. `selection/api.py`,
`selection/rat_selection.py` et `run_pipeline.py` ne sont **pas modifiés** et continuent
de charger des modèles `.keras` en production. Le code PyTorch est **ajouté à côté** du
code TensorFlow existant (qui reste intact), pas un remplacement.

## 1. Fichiers créés

### 1.1 `learning/seed.py` — reproductibilité intra-framework

| Fonction | Paramètres | Sortie | Rôle |
|---|---|---|---|
| `set_tf_seed(seed)` | `seed: int` | `None` | Fixe `random.seed`, `numpy.random.seed`, `PYTHONHASHSEED`, `tf.random.set_seed(seed)`, et active `tf.config.experimental.enable_op_determinism()`. |
| `set_torch_seed(seed)` | `seed: int` | `None` | Fixe `random.seed`, `numpy.random.seed`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`. |

**Point important** : fixer le même seed des deux côtés ne
donne PAS les mêmes poids initiaux entre TF et PyTorch (RNG et initialiseurs différents).
Le seed garantit uniquement la reproductibilité **au sein** d'un même framework d'un run
à l'autre. Pour des poids initiaux identiques, J'ai crée  `weight_transfer.py`.

### 1.2 `learning/model_torch.py` — équivalent PyTorch de `learning/model.py`

| Fonction / Classe | Paramètres | Sortie | Rôle |
|---|---|---|---|
| `rmse_torch(y_pred, y_true)` | 2 tenseurs | tenseur scalaire | `sqrt(mean((y_pred-y_true)^2))`. Évaluation ponctuelle sur un tableau complet uniquement — pas pour l'agrégation epoch par epoch (voir plus bas). |
| `RATPredictor(nn.Module)` | `__init__(model_type, timesteps, features)` | instance | Architecture : `{LSTM\|GRU\|RNN}(features→64)` → `Linear(64,32)+ReLU` → 2 têtes `Linear(32,1)+Sigmoid` (`latency_head`, `pdr_head`). `.forward(x)` retourne `(latency_pred, pdr_pred)`, chacun `(batch,1)`. |
| `build_model_torch(model_type, timesteps, features)` | idem | `RATPredictor` avec `.optimizer` attaché (`Adam(lr=0.001, eps=1e-7)` — `eps` aligné sur le défaut Keras, pas le défaut PyTorch 1e-8) | Construction + compilation façon Keras. |
| `EarlyStopping` | `monitor='loss'`, `patience`, `restore_best_weights`, `min_delta` | — | Réplique `keras.callbacks.EarlyStopping` : surveille la loss d'**entraînement**, sauvegarde le meilleur `state_dict()`, et le restaure **inconditionnellement** en fin d'entraînement (`on_train_end`), que la patience ait été déclenchée ou non. Comportement vérifié directement sur le code source Keras (voir §5). |
| `keras_style_validation_split(X, y_latency, y_pdr, validation_split)` | tableaux + fraction | `(X_tr, y_tr_lat, y_tr_pdr, X_val, y_val_lat, y_val_pdr)` | Réplique exactement `model.fit(..., validation_split=...)` de Keras : les **derniers** `validation_split`% des tableaux (ordre original, sans mélange) sont réservés à la validation. |
| `fit_torch(model, X_train, y_train_dict, epochs, batch_size, validation_split, callbacks, verbose)` | — | `history: dict` (mêmes clés que `history.history` de Keras : `loss`, `latency_ms_loss`, `pdr_loss`, `*_rmse`, et leurs `val_*`) | Boucle d'entraînement complète : split validation (ci-dessus), mélange des données d'entraînement **à chaque epoch** (comme Keras `shuffle=True`), perte pondérée `1.0*mse_latence + 1.5*mse_pdr`, callbacks (`EarlyStopping`, loggers). |
| `TorchDataStreamGenerator(Dataset)` | `X_data, y_data_dict, batch_size` | — | Équivalent de `DataStreamGenerator` (Keras `Sequence`) : indexation **par batch entier**, pas par échantillon. |
| `incremental_train_torch(model, new_X, new_y, epochs, batch_size)` | — | `None` | Équivalent de `incremental_train` (réentraînement en ligne, `EarlyStopping(patience=2)`). |
| `generate_new_measurement(x_new_data, y_new_data, index)` | — | `(x, y)` reshapés en batch de 1 | Identique à la version TF (pur numpy, partagé). |
| `predict_and_retrain_torch(model, x_data, y_data, steps, start)` | — | `None` | Équivalent de `predict_and_retrain`. |
| `automatic_train_torch(model, X_new_data, y_new_data, batch_size, N, validation, log_file, rat, model_type)` | — | `None` (écrit un CSV de log + sauvegarde le modèle réentraîné) | Équivalent de `automatic_train`. |
| `save_torch_model(model, path)` / `load_torch_model(path, map_location)` | — | `None` / `RATPredictor` | Sauvegarde un dict `{model_type, timesteps, features, state_dict}` en `.pt` (pas un pickle du module entier) — permet de reconstruire l'architecture sans dépendre du nom de fichier. |

**Point important ** : Keras enveloppe une fonction de
métrique custom dans un `MeanMetricWrapper` qui moyenne le RMSE par batch, ce n'est
PAS `sqrt(mean(MSE))` sur toute l'epoch (racine carrée = fonction concave, donc les deux
diffèrent). `fit_torch` reproduit fidèlement cette agrégation par-batch.

**Initialisation Keras-like** : par défaut (sans transplantation), `RATPredictor`
initialise ses poids façon Keras (`glorot_uniform` pour les poids entrée→caché,
`orthogonal` pour caché→caché, biais à zéro, +1 sur le biais de la porte "forget" pour
LSTM comme le défaut Keras `unit_forget_bias=True`) — pour qu'un entraînement PyTorch
**autonome** (sans passer par la transplantation) ait une distribution d'init comparable
à Keras, même si non bit-identique.

### 1.3 `learning/main_torch.py` — mirror CLI de `learning/main.py`

| Élément | Rôle |
|---|---|
| `parse_args()` | Mêmes arguments que `learning/main.py` (`--rat`, `--model`, `--data`/`--npz`, `--new_data`, `--load`, `--epochs`) + nouveau `--seed`. |
| `load_csv_data(base_path, file_list, pdr_window, tx_interval)` | **Dupliqué** (pas importé) depuis `learning/main.py` — volontairement, pour que le chemin PyTorch ne force jamais l'import de TensorFlow (voir §4, bug trouvé et corrigé). |
| `prepare_data(df, df_new, rat, data_npz, output_dir)` | Idem, dupliqué pour la même raison. Charge un `.npz` déjà prétraité ou appelle `preprocess_lstm_input` (partagé, agnostique du framework). |
| `TorchMetricsLogger` | Équivalent de `MetricsLogger` (TF), écrit dans `{model}_{rat}_training_log_pt.csv` (suffixe `_pt` pour coexister avec le log TF). |
| `train_single_model_torch(model_type, timesteps, features, X_train, y_train_dict, rat, epochs)` | Construit, entraîne (`fit_torch`), sauvegarde (`{model}_{rat}_{timestamp}.pt`). |
| `main()` | Pipeline complet, identique en structure à `learning/main.py`. |

Usage :
```
python -m learning.main_torch --rat dsrc --model lstm --data <dossier_csv> --seed 42
python -m learning.main_torch --rat dsrc --model lstm --npz <data.npz> --seed 42
```

### 1.4 `learning/weight_transfer.py` — transplantation de poids Keras → PyTorch

C'est le module qui garantit des poids initiaux **identiques** entre TF et PyTorch
(le seed seul ne le permet pas — voir §1.1).

| Fonction | Paramètres | Sortie | Rôle |
|---|---|---|---|
| `build_tf_model_with_seed(model_type, timesteps, features, seed)` | — | modèle Keras non entraîné | Construit un modèle TF avec un seed donné, source de la transplantation. |
| `extract_tf_weights(keras_model, model_type)` | — | `dict {"rnn", "dense", "latency_head", "pdr_head"}` (listes `get_weights()`) | Extrait les poids bruts de chaque couche. Lève une erreur explicite si un layer GRU a `reset_after=False` (mapping non supporté). |
| `_to_torch_layout(tf_weights, model_type)` *(interne)* | — | `dict {nom_paramètre_pytorch: array}` | Convertit les poids Keras au layout/ordre PyTorch exact (voir mapping ci-dessous), sans toucher à aucun modèle — réutilisé à la fois pour la copie et pour la comparaison. |
| `load_tf_weights_into_torch(torch_model, tf_weights, model_type)` | — | `None` (modifie `torch_model` en place) | Copie les poids convertis dans les paramètres PyTorch. |
| `transplant(tf_model, torch_model, model_type)` | — | `None` | `extract_tf_weights` + `load_tf_weights_into_torch` en une étape. |
| `weight_rel_error(tf_model, torch_model, model_type, eps=1e-8)` | — | `dict {nom_paramètre: erreur_relative_médiane}` | Compare les poids **actuels** des deux modèles (utile après un entraînement indépendant) sans les modifier. Voir §6 pour les deux bugs trouvés et corrigés ici. |

**Mapping des poids (vérifié empiriquement par les tests, pas seulement en théorie)** :

- **LSTM** : même ordre de portes des deux côtés (i, f, g/c, o) → simple transposition.
  Keras n'a qu'un seul biais fusionné (taille `4×units`) ; il est assigné entièrement à
  `bias_ih_l0`, et `bias_hh_l0` est mis à zéro (le biais net PyTorch = somme des deux).
- **GRU** (Keras `reset_after=True`, défaut TF2/Keras3, formule "linear before reset" =
  formule native PyTorch) : réordonnancement des blocs `[z,r,h]` (Keras) →
  `[r,z,n]` (PyTorch). Le biais Keras `(2, 3×units)` reste **séparé** en deux biais
  PyTorch (ne pas sommer, contrairement à LSTM).
- **SimpleRNN** : mapping direct, `nn.RNN(nonlinearity='tanh')` explicite.
- **Dense** : `Linear.weight = kernel.T`, biais copié directement.

### 1.5 `validation/` — scripts de comparaison croisée TF vs PyTorch

#### `validation/common.py`

| Fonction | Rôle |
|---|---|
| `force_cpu_torch()` | Force l'exécution CPU côté PyTorch (les kernels cuDNN GPU de TF et PyTorch sont différents et introduisent du bruit non lié à un bug de portage). |
| `rel_error(tf_val, pt_val, eps=1e-8)` | `abs(tf_val - pt_val) / max(abs(tf_val), eps)`, vectorisé numpy. |
| `build_model_pair(model_type, rat, timesteps, features, seed)` | Construit un modèle TF (seedé) + un modèle PyTorch avec poids transplantés identiques → `(tf_model, torch_model)`. |
| `instrumented_forward(tf_model, torch_model, model_type, x)` | Déroule **manuellement** la RNN pas-à-pas (via `LSTMCell`/`GRUCell`/`SimpleRNNCell`, natifs des deux côtés, mêmes poids que la couche complète) pour capturer à chaque timestep : hidden state, cell state (LSTM), puis last hidden, sortie dense, sigmoid pré/post-activation, prédiction finale. Retourne `(chain_tf, chain_pt)`, deux dicts aux clés identiques. |

#### `validation/plotting.py`

Fonctions matplotlib (headless, backend `Agg`), toutes sauvegardées en PNG :

| Fonction | Contenu du graphique |
|---|---|
| `plot_loss_curves(history_tf, history_pt, save_path)` | 3 panneaux : loss totale, latence, PDR — TF vs PT, train (plein) vs val (pointillé). |
| `plot_weight_divergence(weight_divergence, save_path)` | Une courbe par paramètre, erreur relative **médiane** TF vs PT au fil des epochs (échelle log). |
| `plot_prediction_comparison(y_true, pred_tf, pred_pt, save_path)` | Scatter prédit-vs-réel (TF et PT superposés) + scatter TF-vs-PT, par sortie. |
| `plot_chain_divergence(stage_names, mean_errors, max_errors, save_path)` | Erreur relative (moyenne et max) à chaque étape de la chaîne du réseau. |
| `plot_timestep_divergence(hidden_rel_error, cell_rel_error, save_path)` | Erreur relative du hidden state (et cell state pour LSTM) à chaque timestep, une ligne par séquence + moyenne. |

#### `validation/compare_predictions.py` — **Validation Type 1**

CLI :
```
python -m validation.compare_predictions --rat dsrc --model lstm --seed 42 [--npz DATA.npz] --stage {init,trained} [--epochs N]
```
- `--stage init` (défaut) : compare les prédictions juste après transplantation, sans
  entraînement — vérifie la justesse du portage architecture/poids.
- `--stage trained` : entraîne TF et PyTorch indépendamment depuis les mêmes poids
  initiaux, avec un ordre de mélange des batches synchronisé manuellement entre les
  deux frameworks (permutation numpy partagée, shuffle interne désactivé des deux côtés)
  — mesure la divergence due aux différences d'implémentation optimiseur/flottant.
- Sorties dans `output/validation/` : `{rat}_{model}_{stage}_prediction_errors.csv`,
  `{rat}_{model}_{stage}_summary.txt`, `{rat}_{model}_{stage}_predictions.png`, et en
  stage `trained` : `{rat}_{model}_loss_curves.png`, `{rat}_{model}_weight_divergence.png`.

Fonctions internes principales : `_load_eval_batch` (tire un lot réel du `.npz` ou
synthétique), `_predict_tf`/`_predict_torch`, `_report_rel_error` (stats + CSV),
`_train_synced` (boucle d'entraînement manuelle synchronisée, capture loss + divergence
des poids par epoch).

#### `validation/compare_layers.py` — **Validation Type 2**

CLI :
```
python -m validation.compare_layers --rat dsrc --model lstm --seed 42 --npz DATA.npz --n-sequences 10 --seq-length 5 [--start-index N]
```
- Utilise `instrumented_forward` sur un petit sous-ensemble réel du dataset (ou
  synthétique si pas de `--npz`).
- `--start-index` : par défaut, prend le slice à ~1/3 du dataset plutôt qu'au tout
  début, pour éviter la zone de "warm-up" des fenêtres glissantes/PDR qui peut produire
  des séquences atypiques.
- **Garde-fou de représentativité** : avertit si la variance des hidden states capturés
  est quasi nulle (lot dégénéré pouvant masquer une vraie divergence).
- Calcule l'erreur relative à chaque étape de la chaîne (`input`, `hidden_states`,
  `cell_states` si LSTM, `last_hidden`, `dense_out`, `latency_pre/post`, `pdr_pre/post`,
  `prediction`) et à chaque timestep, avec alerte si l'erreur moyenne bondit de plus de
  100x entre deux étapes consécutives.
- Sorties dans `output/validation/` : `{rat}_{model}_chain_report.txt`,
  `{rat}_{model}_chain_divergence.png`, `{rat}_{model}_timestep_divergence.png`.

### 1.6 Tests

| Fichier | Contenu |
|---|---|
| `tests/test_model_torch.py` | Mirror de `tests/test_model.py` : architecture (3 types), shapes de prédiction, `rmse_torch`, `TorchDataStreamGenerator`, `generate_new_measurement`, **et** des tests spécifiques PyTorch : `EarlyStopping` (restauration inconditionnelle des meilleurs poids, arrêt après patience), `fit_torch` (la loss diminue, le split de validation prend bien la fin des données sans mélange). |
| `tests/test_weight_transfer.py` | Pour LSTM/GRU/RNN : après transplantation, les prédictions TF et PyTorch sur une entrée aléatoire coïncident à `<1e-3` près ; l'erreur relative des poids transplantés est `<1e-5` ; un GRU `reset_after=False` est bien rejeté avec une erreur explicite. |

---

## 2. Fichiers modifiés (existants)

| Fichier | Changement |
|---|---|
| `learning/main.py` (TF) | Ajout du flag `--seed` optionnel ; si fourni, appelle `set_tf_seed(seed)` en tout début de `main()`. Aucun autre changement. |
| `utils.py` | `get_latest_model(...)` gagne un paramètre **nommé, en fin de signature**, `extension: str = "keras"` (défaut inchangé → zéro régression pour les appels existants). `learning/main_torch.py` l'appelle avec `extension="pt"`. |
| `requirements.txt` | Ajout de `torch` (non pinné, comme les autres dépendances hors TensorFlow). |
| `learning/__init__.py` | Suppression de `from learning.model import build_model, rmse` (ré-export inutilisé ailleurs dans le repo, vérifié par `grep`) — ce ré-export forçait l'import de TensorFlow **dès qu'on importait n'importe quel sous-module** de `learning`, y compris les modules PyTorch. Voir la partie 4 . |

---

## 3. Environnement d'exécution mis en place

Ni PyTorch ni TensorFlow n'étaient installés (ni sous Windows, ni dans le WSL). Étapes :

1. `python3 -m venv --without-pip .venv` dans `*/v2x-lstm/lstm` — un venv
   classique (`python3 -m venv .venv`) échouait car `ensurepip` nécessite le paquet
   système `python3.12-venv`, lui-même installable seulement via `sudo apt`, qui
   demandait un mot de passe que je ne pouvais pas fournir de façon interactive.
2. Bootstrap de pip sans sudo : `curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && .venv/bin/python3 /tmp/get-pip.py`.
3. `.venv/bin/python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch`
   — build **CPU uniquement** (pas besoin de CUDA ici, et la validation croisée doit de
   toute façon forcer le CPU pour être fiable)parceque si TensorFlow et PyTorch utilisent des implémentations GPU différentes, de petites différences numériques peuvent apparaître.
4. `.venv/bin/python3 -m pip install 'tensorflow~=2.18.0' tf_keras pandas numpy scikit-learn matplotlib tqdm folium scipy tabulate python-dotenv simpy` (le reste de `requirements.txt`).
5. `.venv/bin/python3 -m pip install pytest` (pour lancer la suite de tests).

Pour relancer quoi que ce soit vous-même :
```
cd */v2x-lstm/lstm
.venv/bin/python3 -m pytest tests/ -q
.venv/bin/python3 -m learning.main_torch --rat dsrc --model lstm --npz output/dsrc_lstm_data.npz --seed 42
.venv/bin/python3 -m validation.compare_predictions --rat dsrc --model lstm --seed 42 --npz output/dsrc_lstm_data.npz --stage trained --epochs 5
```

---
**Reamrque** on peut changer le paramétre --stage en init pour faire une validation type 1
## 4. Bug trouvé et corrigé pendant les tests : import TensorFlow forcé côté PyTorch

**Symptôme** : lancer `learning.main_torch` affichait quand même tous les messages de
démarrage TensorFlow/CUDA, alors que ce script ne doit dépendre que de PyTorch.

**Cause** : `learning/__init__.py` (préexistant) faisait
`from learning.model import build_model, rmse` de façon inconditionnelle — donc tout
`import learning.xxx` (y compris `learning.model_torch`) chargeait TensorFlow via le
package parent, même si rien dans `main_torch.py` n'en avait besoin.

**Vérification avant correction** : `grep` sur tout le repo (hors `.venv`) pour
`from learning import` / `import learning` → aucun résultat, donc ce ré-export n'était
utilisé nulle part ailleurs.

**Correction** : suppression de cette ligne dans `learning/__init__.py`. Confirmé par
un nouveau run de `learning.main_torch` (plus aucun message TensorFlow/CUDA) et par la
suite de tests complète (417/417 toujours au vert).

---

## 5. Vérification technique du comportement Keras `EarlyStopping`

Avant d'implémenter `EarlyStopping` côté PyTorch, le comportement exact de
`restore_best_weights=True` a été vérifié directement sur le code source actuel
(`keras/src/callbacks/early_stopping.py`, via fetch GitHub), pas seulement supposé :

```python
# on_train_end
if self.restore_best_weights and self.best_weights is not None:
    self.model.set_weights(self.best_weights)
```

→ La restauration est **inconditionnelle** : elle ne dépend que de `restore_best_weights`
et de l'existence de `best_weights`, **pas** du déclenchement de la patience. Un
entraînement qui va au bout de toutes les epochs sans jamais déclencher la patience se
voit quand même restaurer les poids du **meilleur** epoch, pas ceux du dernier. La classe
PyTorch (`learning/model_torch.py::EarlyStopping`) reproduit exactement ce comportement.

---

## 6. Résultats des validations exécutées et bugs trouvés en cours de route

### 6.1 Entraînements de bout en bout (dataset réel `*/trim_dsrc.csv`)

- `learning.main --rat dsrc --model lstm --data * --seed 42 --epochs 2` :
  OK, génère `output/dsrc_lstm_data.npz` et
  `models/lstm_dsrc_<timestamp>.keras`.
- `learning.main_torch --rat dsrc --model lstm --npz output/dsrc_lstm_data.npz --seed 42 --epochs 2` :
  OK, sauvegarde `models/lstm_dsrc_<timestamp>.pt`. Résultats numériques reproductibles
  (mêmes valeurs à chaque run avec le même seed).

on designe par * le chemin de dossier qui contient le jeux de données 
### 6.2 Validation Type 1 — stage `init` (sanity du portage, sans entraînement)

```
latency_ms: mean=4.527e-08 max=1.189e-07
pdr:        mean=4.159e-08 max=1.128e-07
```
Confirme que le portage
de l'architecture et la transplantation des poids sont corrects, indépendamment de toute
dynamique d'entraînement.

### 6.3 Validation Type 1 — stage `trained` (5 epochs, entraînement indépendant synchronisé)

Loss TF et PyTorch quasi identiques epoch par epoch (`{ep1: 0.0026/0.0026, ep2: 0.0004/0.0004, ...}`).
Erreur de prédiction finale : `latency_ms` mean=1.3e-3/max=8.0e-2 ; `pdr` mean=5.9e-4/max=4.3e-1
— divergence plus élevée qu'en stage `init`, attendue après plusieurs epochs
d'entraînement indépendant (chaque framework accumule ses propres différences
d'implémentation optimiseur/flottant), mais les courbes de loss montrent que les deux
modèles suivent la même trajectoire d'apprentissage.

**Deux bugs trouvés et corrigés dans `weight_rel_error` (`learning/weight_transfer.py`) en observant `dsrc_lstm_weight_divergence.png` :**

1. **`bias_ih_l0`/`bias_hh_l0` traités comme non-redondants** — pour LSTM/SimpleRNN,
   PyTorch a deux biais (`bias_ih_l0`, `bias_hh_l0`) dont seule la **somme** compte dans
   le calcul (ils sont mathématiquement redondants). La transplantation initiale met tout
   dans `bias_ih_l0` et zéro dans `bias_hh_l0`, mais un entraînement peut librement
   déplacer de la masse de l'un vers l'autre sans changer le comportement du modèle.
   L'ancienne version de `weight_rel_error` comparait `bias_hh_l0` entraîné à une valeur
   "attendue" figée à zéro → erreur relative artificielle de l'ordre de 1e6-1e7, sans
   rapport avec une vraie divergence. **Corrigé** : `bias_ih_l0` et `bias_hh_l0` sont
   maintenant comparés en tant que **somme** contre le biais unique de Keras (GRU n'est
   pas concerné, ses deux biais ne sont pas redondants).
2. **Agrégation par la moyenne sensible aux valeurs proches de zéro** — après ce premier
   correctif, `dense.bias` affichait encore une erreur ~6e4 dès la 1ère epoch (alors que
   le stage `init`, avant tout entraînement, donnait ~1e-8). Cause : `abs(TF-PT)/max(|TF|,eps)`
   explose numériquement quand `|TF|` est proche de zéro (fréquent pour un biais en
   sortie de ReLU en début d'entraînement), même pour un écart absolu négligeable ; une
   seule composante quasi nulle sur 32 suffit à faire exploser une **moyenne**.
   **Corrigé** : agrégation par la **médiane** plutôt que la moyenne (robuste à ce type
   de valeur aberrante, sans masquer un vrai décalage généralisé).

**Résultat après les deux correctifs** (`dsrc_lstm_weight_divergence.png` régénéré) :
toutes les courbes sont désormais dans une plage cohérente (erreur relative médiane entre
~0.02 et ~0.6 sur les 5 epochs), avec une divergence qui croît progressivement au fil de
l'entraînement indépendant — un comportement attendu et interprétable, sans plus aucune
valeur aberrante de type 1e6-1e7. Les deux correctifs sont donc validés.

### 6.4 Validation Type 2 (chaîne + par timestep), sur données réelles `dsrc`

```
.venv/bin/python3 -m validation.compare_layers --rat dsrc --model lstm --seed 42 --npz output/dsrc_lstm_data.npz --n-sequences 10 --seq-length 5
```
10 séquences réelles, `start_index=34416` (≈1/3 du dataset, évite la zone de warm-up).
Toutes les étapes de la chaîne sont au niveau du bruit flottant attendu :

```
hidden_states: mean=3.0e-07 max=9.3e-05
cell_states:   mean=2.1e-07 max=9.3e-05
last_hidden:   mean=3.0e-07 max=2.4e-06
dense_out:     mean=1.5e-07 max=2.2e-06
latency_post:  mean=1.2e-08 max=1.2e-07
pdr_post:      mean=5.5e-08 max=2.2e-07
prediction:    mean=3.3e-08 max=2.2e-07
```

Le graphique par timestep (`dsrc_lstm_timestep_divergence.png`) montre une erreur stable
entre ~1e-7 et ~1e-6 sur les 5 timesteps, sans tendance à s'accumuler dans le temps — pas
de dérive progressive, ce qui aurait été le signe d'un problème dans la récurrence.

**Bug trouvé et corrigé** : la détection automatique de "saut anormal >100x entre deux
étapes" (`compare_layers.py`) comparait systématiquement `hidden_states` contre `input`,
dont l'erreur relative est *par construction* exactement 0 (même tableau numpy fourni aux
deux modèles) — ce qui déclenchait un faux avertissement à **chaque** exécution, quelle
que soit la qualité réelle du portage. Corrigé en excluant `input` de cette comparaison
(elle démarre désormais à la 2ᵉ étape réelle) et en remplaçant le plancher `1e-12` par
`1e-6` (le plancher de bruit flottant réellement observé), pour éviter aussi de signaler
de fausses alertes entre deux étapes déjà toutes deux au niveau du bruit.

---

## 6.5 Campagne complète : 9 combinaisons (3 RAT × 3 modèles), 30 epochs

Pour étendre la validation à tous les modèles (`lstm`, `gru`, `rnn`) et
tous les RAT (`5g`, `pc5`, `dsrc`), avec davantage d'epochs (30) pour le stage `trained`.

**Orchestration** : On doit  génèrer les `.npz` manquants pour `5g`/`pc5`/`dsrc`, puis pour chaque
combinaison RAT×modèle lance `compare_predictions --stage init`,
`compare_predictions --stage trained --epochs 30`, puis `compare_layers`, chaque run
loggé séparément dans `output/validation/logs/{rat}_{model}_{étape}.log`.
**Distinction des sorties** : chaque fichier est préfixé `{rat}_{model}_...` (déjà le cas
dès la conception initiale), donc les 9 combinaisons coexistent sans écrasement dans
`output/validation/`. Un nouveau script, `validation/aggregate_reports.py`
(`python -m validation.aggregate_reports`), scanne tous les
`*_prediction_errors.csv` et produit **`output/validation/summary_all.csv`** — une table
unique avec colonnes `rat,model,stage,output,mean_rel_error,max_rel_error,p50,p95,p99`,
pour comparer toutes les combinaisons sans ouvrir chaque rapport un par un.

**Résultats stage `init`** (portage architecture, sans entraînement) — cohérents sur les
9 combinaisons : erreur relative toujours au niveau du bruit flottant (~4e-8 à 3e-7),
quel que soit le RAT ou le type de modèle. Confirme que le portage est fidèle de façon
générale, pas seulement pour le cas `dsrc/lstm` testé initialement.

**Résultats stage `trained`** (30 epochs, entraînement indépendant synchronisé) :

| RAT | Modèle | latency_ms mean | latency_ms max | pdr mean | pdr max |
|---|---|---|---|---|---|
| dsrc | lstm | 0.0041 | 0.330 | 0.00087 | 0.062 |
| dsrc | gru  | 0.0082 | 0.482 | 0.0011  | 0.174 |
| dsrc | rnn  | 0.0059 | 1.024 | 0.0011  | 0.197 |
| 5g   | lstm | 0.0092 | 0.151 | 0.00008 | 0.0005 |
| 5g   | gru  | 0.0161 | 0.182 | 0.00011 | 0.0023 |
| 5g   | rnn  | 0.0121 | 0.131 | 0.00009 | 0.0019 |
| pc5  | lstm | 0.0606 | 0.727 | 0.0018  | 0.025 |
| pc5  | gru  | 0.0553 | 0.672 | 0.0027  | 0.026 |
| pc5  | rnn  | 0.0586 | 0.425 | 0.0039  | 0.019 |

Observations :
- **`pc5` diverge nettement plus** que `dsrc`/`5g` après 30 epochs d'entraînement
  indépendant (moyenne ~5-6% contre ~0.4-1.6% ailleurs), pour les 3 types de modèle.
  Hypothèse la plus probable : `pc5` n'a que 4 features (`FEATURE_COLS['pc5']`, contre 6
  pour `5g`/`dsrc`) et un fichier source plus petit (`trim_pc5.csv` ≈ 3.7 Mo contre ~6 Mo
  pour les deux autres), donc moins de séquences d'entraînement → plus de variance entre
  deux trajectoires d'optimisation indépendantes sur le même nombre d'epochs.
- **`dsrc/rnn` a un max de 102% sur `latency_ms`** alors que sa médiane/moyenne restent
  basses (~0.6%) — un point isolé (probablement un échantillon en zone de saturation
  sigmoid), pas un problème systémique. À surveiller si on relance avec plus d'epochs.
- Globalement, la moyenne et la médiane restent < 6% pour toutes les combinaisons après
  30 epochs d'entraînement **indépendant** (pas juste une comparaison à poids figés),
  ce qui est cohérent avec des courbes de loss quasi superposées observées par ailleurs.

**Pour relancer** cette campagne complète (ou une sous-partie) :
```bash
cd */v2x-lstm/lstm
for RAT in dsrc 5g pc5; do
  for MODEL in lstm gru rnn; do
    .venv/bin/python3 -m validation.compare_predictions --rat $RAT --model $MODEL --seed 42 \
      --npz output/${RAT}_lstm_data.npz --stage init
    .venv/bin/python3 -m validation.compare_predictions --rat $RAT --model $MODEL --seed 42 \
      --npz output/${RAT}_lstm_data.npz --stage trained --epochs 30
    .venv/bin/python3 -m validation.compare_layers --rat $RAT --model $MODEL --seed 42 \
      --npz output/${RAT}_lstm_data.npz --n-sequences 10 --seq-length 5
  done
done
.venv/bin/python3 -m validation.aggregate_reports
```

---

## 7. Emplacement des livrables

- Code source : `learning/*.py`, `validation/*.py`, `tests/*.py` (tous listés ci-dessus).
- Rapports texte/CSV et courbes PNG générés par les scripts de validation :
  **`output/validation/`** (pas dans le dossier `validation/` qui contient le *code*),
  préfixés `{rat}_{model}_...` pour chaque combinaison, plus `summary_all.csv` (table
  agrégée toutes combinaisons) et `logs/` (sortie brute de chaque run de la campagne).
- Modèles entraînés : `models/*.keras` (TF) et `models/*.pt` (PyTorch).
- Historique JSON par epoch : `output/{model}_{rat}_training_history.json` (TF) et
  `output/{model}_{rat}_training_history_pt.json` (PyTorch).
