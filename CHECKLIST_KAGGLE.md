# Checklist de publication Kaggle (règle 6b)

La règle 6b impose que le partage public du code de compétition se fasse **sur Kaggle**
(notebooks ou forum de la compétition). Voici les étapes, dans l'ordre. Budget : ~2-3 h
(surtout des uploads).

## Décision préalable : publier MAINTENANT ou préparer-sans-publier ?

- Si la compétition exige le code **pour valider la soumission** → publier maintenant.
- Si le code n'est demandé **qu'aux gagnants après la clôture** → tout préparer en privé
  (datasets et notebooks en visibilité *Private*), et basculer en *Public* le dernier
  jour ou sur demande des organisateurs. **Recommandé** : ne pas offrir ta recette aux
  concurrents pendant que le leaderboard est ouvert.

Les étapes ci-dessous sont identiques dans les deux cas — seule la visibilité change.

## Étape 1 — Datasets Kaggle (les gros fichiers)

Créer 2 datasets Kaggle (kaggle.com → Datasets → New Dataset) :

**`afrivoices-model-v5t3`** (~2.5 Go)
- Contenu : le dossier `baobab-asr-v5-t3` du Drive : `model.safetensors`, `config.json`,
  `vocab.json`, `tokenizer_config.json`, `preprocessor_config.json`, `added_tokens.json`.
- ⚠️ Le `vocab.json` est critique (mapping CTC).
- Licence à sélectionner : Apache 2.0.

**`afrivoices-kenlm-v2`** (~1.7 Go)
- Contenu : `lm_v2/` : les 6 `.bin` + les 6 `.txt` (corpus, nécessaires aux unigrams
  du décodeur).
- Licence : Apache 2.0 (mentionner Wikipedia CC-BY-SA pour les corpus dans la description).

Astuce upload : depuis Colab, `pip install kaggle`, placer `kaggle.json` (API token),
puis `kaggle datasets create -p dossier/` — plus fiable que l'upload navigateur pour
plusieurs Go.

## Étape 2 — Notebooks Kaggle (le code)

Importer les 6 notebooks (kaggle.com → Code → New Notebook → File → Import) :
1. `afrivoices_asr_train.ipynb` — base v2 + KenLM + inférence
2. `afrivoices_retrain_v3.ipynb`
3. `afrivoices_retrain_v4_propre.ipynb`
4. `afrivoices_retrain_v5_tranches.ipynb` — cœur du système (+ §10 décodage)
5. `afrivoices_kenlm_v2.ipynb`
6. `afrivoices_retrain_v6_anvke.ipynb` (si v6 aboutit)

Pour chacun : titre clair (« AfriVoices — v5 training by slices »), attacher à la
compétition (Add-ons → Competition), coller en 1ère cellule markdown le résumé du
README correspondant. Pas besoin qu'ils soient exécutables sur Kaggle (GPU/Drive
différents) : ils documentent la méthode — le préciser en tête.

## Étape 3 — Post forum de la compétition

Un post « Solution write-up (0.42) » : architecture (2 étages), tableau de progression
des scores, liens vers les notebooks et datasets, conformité edge (RTF/RAM), licences.
Reprendre le README — c'est son rôle.

## Étape 4 — Vérifications finales

- [ ] LICENSE : remplacer `[Nom du participant / de l'équipe]` par ton nom + année
- [ ] Aucun token HF / clé API dans les notebooks publiés (chercher `hf_`)
- [ ] Les sorties de cellules sensibles (chemins Drive privés) : acceptables, mais
      vérifier qu'aucun secret n'y traîne
- [ ] Datasets et notebooks liés entre eux (descriptions croisées)
- [ ] La soumission retenue sur Kaggle correspond bien au modèle publié

## Étape 5 — (si v6 aboutit) mise à jour

- Nouveau dataset `afrivoices-model-v6` ou nouvelle version du dataset v5-t3
  (Kaggle versionne les datasets)
- Mettre à jour le write-up avec le nouveau score
- Ne JAMAIS faire cette bascule dans les dernières 24 h sans re-tester l'inférence
