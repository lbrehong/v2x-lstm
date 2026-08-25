# Commandes de vérification de la migration TF → PyTorch

Ce document liste, dans l'ordre logique, toutes les commandes nécessaires pour
installer l'environnement, faire tourner les tests, entraîner les deux
versions du modèle, et exécuter les validations croisées TF vs PyTorch
décrites dans [`MIGRATION_TF_TO_PYTORCH.md`](MIGRATION_TF_TO_PYTORCH.md).

Toutes les commandes supposent d'être dans le dossier du projet :

```bash
cd ~/v2x-lstm/lstm
```

---

## 1. Installation de l'environnement

À exécuter une seule fois (si `.venv` n'existe pas déjà).

```bash
# 1. Créer un venv sans dépendre d'ensurepip
python3 -m venv --without-pip .venv

# 2. Bootstrap de pip sans sudo
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python3 /tmp/get-pip.py

# 3. PyTorch (build CPU uniquement)
.venv/bin/python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch

# 4. TensorFlow + reste des dépendances du projet
.venv/bin/python3 -m pip install 'tensorflow~=2.18.0' tf_keras pandas numpy scikit-learn matplotlib tqdm folium scipy tabulate python-dotenv simpy

# 5. pytest (pour la suite de tests)
.venv/bin/python3 -m pip install pytest
```

---

## 2. Suite de tests (TF + PyTorch)

```bash
.venv/bin/python3 -m pytest tests/ -q
```

Pour ne lancer que les tests liés à la migration PyTorch :

```bash
.venv/bin/python3 -m pytest tests/test_model_torch.py tests/test_weight_transfer.py -q
```

---

## 3. Entraînement de bout en bout

### 3.1 Côté TensorFlow (référence)

```bash
.venv/bin/python3 -m learning.main --rat dsrc --model lstm --data ~/full_new/full_new --seed 42 --epochs 2
```

Génère `output/dsrc_lstm_data.npz` et `models/lstm_dsrc_<timestamp>.keras`.

### 3.2 Côté PyTorch (à partir des données déjà prétraitées)

```bash
.venv/bin/python3 -m learning.main_torch --rat dsrc --model lstm --npz output/dsrc_lstm_data.npz --seed 42 --epochs 2
```

Génère `models/lstm_dsrc_<timestamp>.pt`.

### 3.3 Depuis des CSV bruts (alternative à `--npz`)

```bash
.venv/bin/python3 -m learning.main_torch --rat dsrc --model lstm --data <dossier_csv> --seed 42
```

---

## 4. Validation Type 1 — comparaison des prédictions finales

Script : `validation/compare_predictions.py`.

### 4.1 Stage `init` — sanity du portage (sans entraînement)

Vérifie que l'architecture et la transplantation des poids sont correctes
(erreur attendue : bruit flottant, `~1e-7` à `~1e-6`).

```bash
.venv/bin/python3 -m validation.compare_predictions \
  --rat dsrc --model lstm --seed 42 --npz output/dsrc_lstm_data.npz --stage init
```

### 4.2 Stage `trained` — entraînement indépendant synchronisé

Vérifie que TF et PyTorch convergent de façon comparable (courbes de loss,
divergence des poids et des prédictions au fil des epochs).

```bash
.venv/bin/python3 -m validation.compare_predictions \
  --rat dsrc --model lstm --seed 42 --npz output/dsrc_lstm_data.npz \
  --stage trained --epochs 5
```

Sorties dans `output/validation/` :
`{rat}_{model}_{stage}_prediction_errors.csv`,
`{rat}_{model}_{stage}_summary.txt`,
`{rat}_{model}_{stage}_predictions.png`,
et en stage `trained` : `{rat}_{model}_loss_curves.png`,
`{rat}_{model}_weight_divergence.png`.

---

## 5. Validation Type 2 — comparaison couche par couche

Script : `validation/compare_layers.py`. Compare les activations internes
(hidden/cell states, sorties denses, prédictions) à chaque étape du réseau et
à chaque timestep, sur un petit échantillon réel.

```bash
.venv/bin/python3 -m validation.compare_layers \
  --rat dsrc --model lstm --seed 42 --npz output/dsrc_lstm_data.npz \
  --n-sequences 10 --seq-length 5
```

Sorties dans `output/validation/` :
`{rat}_{model}_chain_report.txt`,
`{rat}_{model}_chain_divergence.png`,
`{rat}_{model}_timestep_divergence.png`.

---

## 6. Campagne complète (toutes combinaisons RAT × modèle)

Boucle sur les 3 RAT (`dsrc`, `5g`, `pc5`) et les 3 types de modèle
(`lstm`, `gru`, `rnn`), avec 30 epochs pour le stage `trained`. Chaque run est
loggé séparément dans `output/validation/logs/`.

```bash
for RAT in dsrc 5g pc5; do
  for MODEL in lstm gru rnn; do
    .venv/bin/python3 -m validation.compare_predictions --rat $RAT --model $MODEL --seed 42 \
      --npz output/${RAT}_lstm_data.npz --stage init \
      2>&1 | tee output/validation/logs/${RAT}_${MODEL}_init.log

    .venv/bin/python3 -m validation.compare_predictions --rat $RAT --model $MODEL --seed 42 \
      --npz output/${RAT}_lstm_data.npz --stage trained --epochs 30 \
      2>&1 | tee output/validation/logs/${RAT}_${MODEL}_trained.log

    .venv/bin/python3 -m validation.compare_layers --rat $RAT --model $MODEL --seed 42 \
      --npz output/${RAT}_lstm_data.npz --n-sequences 10 --seq-length 5 \
      2>&1 | tee output/validation/logs/${RAT}_${MODEL}_layers.log
  done
done
```

Si les `.npz` de `5g`/`pc5` n'existent pas encore, les générer d'abord avec
`learning.main` (§3.1) pour chaque RAT.

### 6.1 Agrégation des résultats

Une fois la campagne terminée, produire la table récapitulative
`output/validation/summary_all.csv` (une ligne par combinaison RAT/modèle/stage/sortie) :

```bash
.venv/bin/python3 -m validation.aggregate_reports
```

---

## 7. Checklist rapide (avant de considérer la migration validée)

1. `pytest tests/ -q` → tout au vert.
2. `main.py` (TF) et `main_torch.py` (PyTorch) tournent tous les deux sans
   erreur sur les mêmes données.
3. `compare_predictions --stage init` → erreur relative au niveau du bruit
   flottant (`~1e-7`–`1e-6`) pour toutes les combinaisons RAT/modèle testées.
4. `compare_predictions --stage trained` → courbes de loss TF/PyTorch quasi
   superposées, divergence des poids qui croît progressivement (pas de saut
   brutal type `1e6`).
5. `compare_layers` → erreur stable au niveau du bruit flottant à chaque
   étape de la chaîne, sans dérive progressive dans le temps.
6. `aggregate_reports` → `summary_all.csv` sans valeur aberrante inexpliquée.
