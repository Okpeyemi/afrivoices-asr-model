# AfriVoices East Africa — ASR multilingue edge (6 langues)

Système de reconnaissance vocale automatique (ASR) **unique** couvrant six langues
est-africaines, conçu pour fonctionner **hors-ligne sur du matériel embarqué**
(Raspberry Pi 4 / Android), pour la compétition *AfriVoices East Africa: ASR Hackathon*.

Langues : **swahili (swa), somali (som), kikuyu (kik), luo (luo), kalenjin (kln),
maasai (mas)**.

## Résultats (leaderboard public)

| Système | WER |
|---|---|
| w2v-BERT v2, décodage glouton | 0.566 |
| v2 + KenLM v1 | 0.469 |
| v3 + KenLM v1 (alpha 0.7) | 0.455 |
| **v5-t3 + KenLM v2 (alpha 0.5, beta 1.0)** | **0.42** |

WER greedy par langue (éval interne, v5-t3) : swa 0.103 · kik 0.281 · som 0.331 ·
kln 0.653 · mas 0.690 · luo 0.714 — macro 0.462. Avec KenLM v2 : macro interne 0.30.

## Architecture (deux étages indépendants)

1. **Modèle acoustique** : `facebook/w2v-bert-2.0` (MIT, 606M paramètres) + tête CTC,
   vocabulaire de 67 caractères **partagé** entre les 6 langues. Entraînement progressif :
   fine-tuning multilingue à *temperature sampling* (T=1.6), puis **entraînement par
   tranches** — le dataset complet (~750 Go, incompatible avec le disque local) est
   parcouru en tranches de ~75k clips (1 époque chacune, données 100 % fraîches),
   chaîne de modèles v4 → v5-t0 → … → v5-t3, arrêt au plateau.
2. **Modèles de langue KenLM v2** : un 5-gram par langue (pyctcdecode), entraîné sur
   transcriptions ×3 + **texte externe** (Wikipedia sw/so/ki — 600k phrases pour le
   swahili), filtré par l'alphabet exact du modèle, élagué (`--prune 0 0 1`).
   Hyperparamètres réglés par grille : **alpha 0.5, beta 1.0**.

## Conformité edge

| Critère | Limite | Mesuré | Statut |
|---|---|---|---|
| Paramètres | ≤ 1 Md | 0.606 Md | ✓ |
| RAM (modèle + 1 KenLM, pire cas swa) | ≤ 8 Go | ≈ 1.13 Go | ✓ |
| RTF (CPU 4 threads, pipeline complet) | ≤ 2× | 0.28 moyen / 0.74 max | ✓ |
| Hors-ligne, CPU uniquement | requis | oui | ✓ |

Détails et méthodologie : `HARDWARE_VALIDATION.md`.

## Structure

```
├── README.md / MODEL_CARD.md / HARDWARE_VALIDATION.md / LICENSE / requirements.txt
├── CHECKLIST_KAGGLE.md            # étapes de publication (règle 6b)
├── notebooks/
│   ├── afrivoices_asr_train.ipynb           # entraînement initial + KenLM + inférence
│   ├── afrivoices_retrain_v3.ipynb          # v2 -> v3
│   ├── afrivoices_retrain_v4_propre.ipynb   # v3 -> v4 (SpecAugment doux, filtrage >20s)
│   ├── afrivoices_retrain_v5_tranches.ipynb # v4 -> v5-t3 (tranches) + décodage §10
│   ├── afrivoices_kenlm_v2.ipynb            # KenLM v2 (texte externe)
│   └── afrivoices_retrain_v6_anvke.ipynb    # tranches sur données officielles Anv-ke
├── lm/          # KenLM v2 : 6 x .bin + corpus (ou lien dataset Kaggle)
└── checkpoints/ # poids v5-t3 (ou lien dataset Kaggle)
```

## Reproduire

1. Environnement : `requirements.txt` (⚠️ sur Colab : installer kenlm+pyctcdecode PUIS
   redémarrer le runtime ; ne jamais forcer torch ni numpy<2).
2. Entraînement : `afrivoices_asr_train.ipynb` (base) puis les notebooks de réentraînement
   dans l'ordre (v3 → v4-propre → v5-tranches, `TRANCHE` = seul réglage entre sessions).
3. KenLM v2 : `afrivoices_kenlm_v2.ipynb` (session CPU).
4. Inférence + soumission : §10 du notebook v5-tranches (grille alpha/beta incluse).

## Données et licences

| Source | Usage | Licence |
|---|---|---|
| DigitalUmuganda/Afrivoice_Swahili | train swa | CC-BY-4.0 |
| MCAA1-MSU/anv_data_ke, puis Anv-ke/* (officiel) | train 5 langues | CC-BY-4.0 |
| Wikipedia sw/so/ki | corpus KenLM v2 | CC-BY-SA |
| facebook/w2v-bert-2.0 (modèle de base) | pré-entraîné | MIT |

Aucune transcription manuelle du test. Aucun composant sous licence non-commerciale.

## Licence

Code et modèles sous **Apache-2.0** (OSI, sans restriction commerciale — règle 6c).
Partage public effectué sur Kaggle conformément à la règle 6b.
