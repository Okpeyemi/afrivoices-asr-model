# AfriVoices East Africa — ASR multilingue edge (6 langues)

Système de reconnaissance vocale automatique (ASR) **unique** couvrant six langues
est-africaines, conçu pour fonctionner **hors-ligne sur du matériel embarqué**
(Raspberry Pi 4 / Android), pour la compétition *AfriVoices East Africa: ASR Hackathon*.

Langues (ISO 639-3) : **swa, som, kik, luo, kln, mas** (swahili, somali, kikuyu, luo,
kalenjin, maasai).

## Résultats (leaderboard public, WER macro)

| Système | WER |
|---|---|
| w2v-BERT v2, décodage glouton | 0.566 |
| v2 + KenLM v1 | 0.469 |
| v3 + KenLM v1 (α 0.7) | 0.455 |
| v5-t3 + KenLM v2 (α 0.5) | 0.42 |
| v9-bicible + KenLM v2 (α 0.7, β 0.5) | 0.41477 |
| **v9-2 (tranche 4, ciblage kln) + KenLM v2** | **0.40283** |

Le score officiel est une **moyenne non pondérée du WER sur les 6 langues** (chaque
langue compte pour 1/6). WER par langue (dev officiel, avec KenLM v2) : swa ~0.10 ·
luo 0.24 · kik 0.31 · som 0.27 · mas 0.46 · kln 0.48.

**CER (taux d'erreur caractère) ≈ 0.094** : le modèle reconnaît correctement ~91 % des
caractères. Le WER-mot résiduel provient surtout de désaccords de segmentation de mots
et de la difficulté acoustique intrinsèque des langues nilotiques (kln/mas).

## Architecture (deux étages indépendants)

1. **Modèle acoustique** : `facebook/w2v-bert-2.0` (MIT, 606M params) + tête CTC,
   vocabulaire de 67 caractères **partagé** entre les 6 langues. Entraînement progressif
   par **tranches** (le dataset complet ~750 Go dépasse le disque local, parcouru en
   tranches de ~60-80k clips, 1 époque chacune) : v4 → v5-t3 → v6 → v7-soup (fusion de
   poids) → v8-cible (ciblage langues faibles) → v9-bicible (kln+mas) → **v9-2** (tranche 4, ciblage kln).
2. **Modèles de langue KenLM v2** : un 5-gram par langue (pyctcdecode), transcriptions
   ×3 + texte externe (Wikipedia sw/so/ki). Hyperparamètres : **α 0.7, β 0.5**.

## Conformité edge

| Critère | Limite | Mesuré | Statut |
|---|---|---|---|
| Paramètres | ≤ 1 Md | 0.606 Md | ✓ |
| RAM (modèle + 1 KenLM, pire cas swa) | ≤ 8 Go | ≈ 1.13 Go | ✓ |
| RTF (CPU 4 threads, pipeline complet) | ≤ 2× | 0.28 moy / 0.74 max | ✓ |
| Hors-ligne, CPU uniquement | requis | oui | ✓ |

## Structure

```
├── README.md / MODEL_CARD.md / HARDWARE_VALIDATION.md / LICENSE / requirements.txt
├── CHECKLIST_KAGGLE.md
├── GUIDE_REPRODUCTION.md   # ← point d'entrée pour les auditeurs
├── RAPPORT_afrivoices.md   # démarche complète (choix, mesures, impasses)
├── notebooks/
│   ├── 1_entrainement/         # chaîne v2 → v9-2 (ordre numéroté dans le guide)
│   ├── 2_modeles_de_langue/    # construction KenLM v2 (v3 exploratoire)
│   ├── 3_inference_soumission/ # ← afrivoices_soumission.ipynb = LE notebook de vérification
│   └── 4_analyses/             # audits, grilles, diagnostics du rapport
├── lm/                 # KenLM v2 (ou lien dataset Kaggle)
└── checkpoints/        # poids v9-bicible (ou lien dataset Kaggle)
```

## Reproduire

**→ Voir `GUIDE_REPRODUCTION.md`** : chemin rapide « vérifier la soumission » (~45 min,
1 seul notebook) et chemin complet « réentraîner de zéro » (ordre des notebooks, accès
aux données gated, artefacts).


1. Environnement : `requirements.txt` (⚠️ Colab : installer kenlm+pyctcdecode PUIS
   redémarrer le runtime ; ne jamais forcer torch ni numpy<2).
2. Entraînement : notebooks de réentraînement dans l'ordre (v4-propre → v5-tranches →
   v6-anvke → v8-drive → v9-bicible), `TRANCHE` = seul réglage entre sessions.
3. KenLM v2 : `afrivoices_kenlm_v2.ipynb` (session CPU).
4. Inférence + soumission : `afrivoices_soumission.ipynb`
   (MODEL_NAME=baobab-asr-v9-2, LM_SUBDIR=lm_v2, α 0.7, β 0.5).

## Données et licences

| Source | Usage | Licence |
|---|---|---|
| DigitalUmuganda/Afrivoice_Swahili | train swa | CC-BY-4.0 |
| Anv-ke/* (train officiel, 5 langues) | train | CC-BY-4.0 |
| Wikipedia sw/so/ki | corpus KenLM v2 | CC-BY-SA |
| facebook/w2v-bert-2.0 (modèle de base) | pré-entraîné | MIT |

Aucune transcription manuelle du test. Filet anti-fuite vérifié (0 chevauchement des
`filename` de train/dev avec les 41 733 ids du test). Aucun composant sous licence
non-commerciale.

## Licence

Code et modèles sous **Apache-2.0** (OSI, sans restriction commerciale — règle 6c).
Partage public sur Kaggle conformément à la règle 6b.
