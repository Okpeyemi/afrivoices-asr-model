# GUIDE DE REPRODUCTION

Ce guide s'adresse aux organisateurs/auditeurs. Deux chemins selon l'objectif :
**vérifier la soumission** (30-60 min, recommandé) ou **réentraîner de zéro** (30 h+ GPU).

---

## Chemin A — Vérifier la soumission finale (recommandé, ~45 min sur GPU)

Le fichier soumis (`submission_v9_2_polB.csv`, leaderboard **0.39477**) se régénère avec
UN seul notebook : **`notebooks/3_inference_soumission/afrivoices_soumission_par_type.ipynb`**
(le « générateur final » : routage par (langue × type), troncature anti-padding, décodage
direct pleine longueur, reprise automatique). La démarche qui mène de 0.40283 à 0.39477
est documentée dans `RAPPORT_afrivoices.md`, **Partie II**.

### Prérequis
1. **Les poids du modèle et les modèles de langue** (fournis à part — voir section
   « Accès aux artefacts » ci-dessous) :
   - `baobab-asr-v9-2/` : modèle acoustique final (w2v-BERT + CTC, ~2.5 Go)
   - `lm_v2/` : 6 fichiers KenLM `.bin` + 6 corpus `.txt` (~1.7 Go)
2. **Le jeu de test** : celui de la compétition (fourni par les organisateurs sur Kaggle,
   ou `digitalumuganda/anv-test-data-nt` sur Hugging Face — le notebook le télécharge
   automatiquement s'il ne le trouve pas localement).
3. Un environnement GPU (Colab T4 suffit) — sur CPU l'inférence fonctionne aussi mais
   prend plusieurs heures.

### Étapes
1. Placer `baobab-asr-v9-2/` et `lm_v2/` dans un dossier de base (par défaut le notebook
   attend `Drive/afrivoices/` sur Colab ; ajuster la variable `BASE` en §2 sinon).
2. Ouvrir `afrivoices_soumission_par_type.ipynb`. Config §2 (déjà préréglée sur la
   soumission finale) : `MODEL_NAME="baobab-asr-v9-2"`, `LM_SUBDIR="lm_v2"`, et la table
   `AB` = **(α, β) = (0.5, 0.0)** pour les six langues et les deux types. Pour reproduire
   la soumission de contrôle (**0.39923**), remettre toute la table `AB` à (0.7, 0.5).
3. Exécuter §1 (installe kenlm/pyctcdecode — sur image Colab récente, faire
   `pip install --no-deps pyctcdecode pygtrie` puis kenlm, sinon numpy est rétrogradé et
   l'image casse —, **redémarrer le runtime**, relancer §1), puis §2 → §6 dans l'ordre.
4. Sortie : `submission_v9_2_polB.csv` — 41 733 lignes, colonnes `id,language,transcription`
   (asserts de conformité intégrés en §6 ; reprise automatique via le fichier `_partiel`).

> Décomposition (facultatif) : `afrivoices_soumission_greedy.ipynb` régénère la
> transcription **sans modèle de langue** (≈0.45), pour isoler la contribution du KenLM.

### Validation edge
Les rapports de validation matérielle datés sont dans `validation/` (un par soumission,
générés par `notebooks/4_analyses/afrivoices_validation_materielle.ipynb`, session CPU) ;
`HARDWARE_VALIDATION.md` en détaille la méthodologie. Mesures (config finale) :
0.606 Md paramètres ; RAM plancher 1.46 Go, pic 6.71 Go en décodage direct ≤ 60 s puis
repli fenêtré à 1.80 Go au-delà ; RTF CPU 4 threads 0.75 agrégé / 1.48 max (direct),
0.69 en repli ; latence projetée ≈ 333 h sur les 467 h d'audio du test.

---

## Chemin B — Réentraîner de zéro (~30 h+ de GPU A100)

### Accès aux données d'entraînement
- **Swahili** : `DigitalUmuganda/Afrivoice_Swahili` (Hugging Face, gated — demander
  l'accès).
- **5 langues kényanes** : dépôts `Anv-ke/{kikuyu, Dholuo, Kalenjin, Maasai, Somali}`
  (Hugging Face, gated) — lien officiel posté par les organisateurs dans le forum de la
  compétition. Nos notebooks les lisent depuis un montage Google Drive (dossier
  `afrivoices/anvke/` contenant un raccourci par langue) pour la résilience aux coupures ;
  la lecture directe `hf://` fonctionne aussi (variante dans
  `1_entrainement/afrivoices_retrain_v6_anvke.ipynb`).

### Ordre des notebooks (chaque étape part du modèle de la précédente)
| # | Notebook (dossier `1_entrainement/`) | Produit |
|---|---|---|
| 1 | `afrivoices_asr_train.ipynb` | modèle de base v2 (+ premier KenLM) |
| 2 | `afrivoices_retrain_v3.ipynb` | v3 |
| 3 | `afrivoices_retrain_v4_propre.ipynb` | v4 |
| 4 | `afrivoices_retrain_v5_tranches.ipynb` | v5-t0 → v5-t3 (1 session par tranche, régler `TRANCHE`) |
| 5 | `afrivoices_retrain_v6_anvke.ipynb` | v6-t0 → v6-t1 (données officielles, éval = dev officiel) |
| 6 | `afrivoices_retrain_v8_drive.ipynb` | v7-soup (cellule fusion en §10) puis v8-cible |
| 7 | `afrivoices_retrain_v9_bicible.ipynb` | v9-bicible (TRANCHE=3) puis v9-2 (TRANCHE=4, probs ciblées kln) |

Modèles de langue : `2_modeles_de_langue/afrivoices_kenlm_v2.ipynb` (session CPU,
indépendant de l'entraînement — peut se faire en parallèle).

### Analyses (facultatif — reproduisent les diagnostics du rapport)
`4_analyses/` : audit d'erreurs par catégorie, grille beta, tests de normalisation,
mesure de la distribution des durées, **banc dev stratifié (langue × type)**
(`afrivoices_p0b_dev_stratifie.ipynb`) et **validation matérielle**
(`afrivoices_validation_materielle.ipynb`). Voir `RAPPORT_afrivoices.md` — Partie I pour
les diagnostics historiques, Partie II pour le banc stratifié et la campagne finale.
`5_explorations/` regroupe les leviers mesurés puis écartés (fenêtrage, hybride, v10
spontané) — voir le `README.md` qu'il contient.

---

## Accès aux artefacts (poids + KenLM)

Les poids finaux (`baobab-asr-v9-2`) et les modèles de langue (`lm_v2`) ne sont pas dans
ce dépôt Git (taille). Ils sont fournis via [**dataset Kaggle privé / lien à insérer au
moment de la remise**] et partagés aux organisateurs sur demande. Contenu exact :
- `baobab-asr-v9-2/` : `model.safetensors`, `config.json`, `vocab.json`,
  `tokenizer_config.json`, `preprocessor_config.json` (+ `added_tokens.json`)
- `lm_v2/` : `{swa,som,kik,luo,kln,mas}.bin` et `.txt`

## Licences
Voir le tableau du `README.md` : base w2v-BERT 2.0 (MIT), données CC-BY-4.0, corpus
Wikipedia CC-BY-SA, code et modèles livrés sous Apache-2.0. Aucun composant à
restriction commerciale. Aucune transcription manuelle du test ; vérification anti-fuite
documentée (0 chevauchement train/test).
