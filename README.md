> **→ Dépôt canonique (code + poids + LM + validation) : https://huggingface.co/Okpeyemi/afrivoices-asr-baobab**

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
| v9-2 (tranche 4, ciblage kln) + KenLM v2 (α 0.7, β 0.5) | 0.40283 |
| v9-2 + troncature anti-padding, décodage direct (« contrôle par type ») | 0.39923 |
| v9-2 **sans** modèle de langue (glouton, décomposition) | ≈ 0.45 |
| **v9-2 + KenLM v2, α 0.5, β 0.0 — FINALE** | **0.39477** |

Le score officiel est une **moyenne non pondérée du WER sur les 6 langues**.
Décomposition du score final : acoustique seule ≈ 0.45 ; contribution du modèle de
langue ≈ −0.055 ; corrections de décodage (campagne post-0.40283) ≈ −0.008.

Le test fournit une colonne `type` par clip (scripted / unscripted) : la part de parole
**spontanée** va de 18.6 % (kik) à 42.2 % (mas), et le swahili est 100 % spontané. Le
WER stratifié par (langue × type) et la calibration dev↔leaderboard sont dans
`RAPPORT_afrivoices.md` (Partie II, §2).

## Architecture (deux étages indépendants)

1. **Modèle acoustique** : `facebook/w2v-bert-2.0` (MIT, 606M params) + tête CTC,
   vocabulaire de 67 caractères **partagé** entre les 6 langues. Entraînement progressif
   par **tranches** (dataset ~750 Go > disque local) : v4 → v5-t3 → v6 → v7-soup →
   v8-cible → v9-bicible → **v9-2**. L'acoustique est gelée à v9-2 ; la campagne finale
   (Partie II du rapport) n'a modifié que la mesure et le décodage.
2. **Modèles de langue KenLM v2** : un 5-gram par langue (pyctcdecode), corpus =
   transcriptions d'entraînement ×3 + texte externe (Wikipedia sw/so/ki, MasakhaNEWS
   luo), recette `lmplz -o 5 --discount_fallback --prune 0 0 1`. Hyperparamètres
   finaux : **α 0.5, β 0.0** (α 0.7 / β 0.5 jusqu'au contrôle ; justification mesurée
   en Partie II §4 — sur un test à dominante spontanée, réduire le poids d'un LM nourri
   de texte écrit/lu rend l'arbitrage à l'acoustique).

## Conformité edge

| Critère | Limite | Mesuré | Statut |
|---|---|---|---|
| Paramètres | ≤ 1 Md | 0.606 Md | ✓ |
| RAM (RSS pic : modèle fp32 + pire KenLM) | ≤ 8 Go | pic 6,71 Go en direct ≤ 60 s ; 1,80 Go via repli fenêtré au-delà (voir validation/) | ✓ |
| RTF (CPU 4 threads, pipeline complet) | ≤ 2× | 0.75 agrégé / 1.48 max (direct ≤ 60 s) ; 0.69 via repli fenêtré — détail dans `validation/` | ✓ |
| Hors-ligne, CPU uniquement | requis | oui | ✓ |

Note d'intégrité de la soumission : 7 clips du test (sur 41 733) ont des bytes audio
illisibles par tout lecteur (soundfile, librosa, repli fichier) ; ils sont transcrits
« _ » par conception (impact ≤ 0.0002 de macro). Ids listés en Partie II §5.

## Structure

```
├── README.md / MODEL_CARD.md / HARDWARE_VALIDATION.md / LICENSE / requirements.txt
├── CHECKLIST_KAGGLE.md
├── GUIDE_REPRODUCTION.md   # ← point d'entrée pour les auditeurs
├── RAPPORT_afrivoices.md   # démarche complète (Partie I : chaîne v2→v9-2 ;
│                           #   Partie II : campagne de mesure post-0.40283)
├── notebooks/
│   ├── 1_entrainement/          # chaîne v2 → v9-2
│   ├── 2_modeles_de_langue/     # construction KenLM v2 (v3 exploratoire)
│   ├── 3_inference_soumission/
│   │   ├── afrivoices_soumission_par_type.ipynb  # ← GÉNÉRATEUR FINAL (0.39477)
│   │   ├── afrivoices_soumission_greedy.ipynb    # décomposition sans LM (≈0.45)
│   │   └── afrivoices_soumission.ipynb           # générateur historique (0.40283)
│   ├── 4_analyses/
│   │   ├── afrivoices_p0b_dev_stratifie.ipynb    # banc stratifié (langue × type)
│   │   └── ...                                    # audits, grilles, diagnostics
│   └── 5_explorations/          # leviers mesurés puis écartés (README dedans)
├── validation/          # 1 rapport de validation matérielle PAR soumission (règlement)
├── lm_v2/               # KenLM v2 : 6 binaires + listes d'unigrams exactes (50k/langue)
└── (racine du dépôt HF) # poids v9-2 : from_pretrained("Okpeyemi/afrivoices-asr-baobab")
```

## Reproduire

**→ Voir `GUIDE_REPRODUCTION.md`.** Chemin rapide « vérifier la soumission finale » :

1. Environnement : `requirements.txt`. ⚠️ Colab récent : installer
   `pip install --no-deps pyctcdecode pygtrie` (pyctcdecode épingle numpy<2 ; l'installer
   avec ses dépendances rétrograde numpy et casse l'image — erreur
   `numpy.dtype size changed`), puis kenlm, puis **redémarrer le runtime**.
2. Entraînement (optionnel, chemin complet) : notebooks 1_entrainement dans l'ordre,
   `TRANCHE` = seul réglage entre sessions.
3. KenLM v2 : `afrivoices_kenlm_v2.ipynb` (session CPU).
4. **Soumission finale** : `afrivoices_soumission_par_type.ipynb` — config par défaut
   = finale (v9-2, lm_v2, α 0.5, β 0.0, troncature anti-padding, décodage direct,
   reprise auto). Remettre (0.7, 0.5) reproduit la soumission de contrôle (0.39923).
5. **Validation matérielle** (exigée pour chaque soumission) :
   `afrivoices_validation_materielle.ipynb` (session CPU) → rapport daté dans
   `validation/` (RAM, RTF stratifié, latence projetée sur les 41 733 clips).

## Données et licences

| Source | Usage | Licence |
|---|---|---|
| DigitalUmuganda/Afrivoice_Swahili | train swa | CC-BY-4.0 |
| Anv-ke/* (train officiel, 5 langues) | train | CC-BY-4.0 |
| Wikipedia sw/so/ki | corpus KenLM v2 | CC-BY-SA |
| masakhane/masakhanews (luo) | corpus KenLM v2 | voir fiche HF |
| facebook/w2v-bert-2.0 (modèle de base) | pré-entraîné | MIT |

Aucune transcription manuelle du test. Filet anti-fuite vérifié (0 chevauchement des
`filename` de train/dev avec les 41 733 ids du test ; locuteurs train/dev disjoints,
vérifié par `recorder_uuid`). Aucun composant sous licence non-commerciale.

## Licence

Code et modèles sous **Apache-2.0** (OSI, sans restriction commerciale — règle 6c).
Partage public sur Kaggle conformément à la règle 6b.
