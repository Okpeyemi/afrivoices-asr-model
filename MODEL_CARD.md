# Model Card — AfriVoices East Africa ASR (v5-t3 + KenLM v2)

## Description

ASR multilingue unique pour six langues est-africaines, déployable hors-ligne sur CPU
embarqué. Modèle acoustique w2v-BERT 2.0 + CTC (vocabulaire de caractères partagé),
rescoring KenLM 5-gram par langue enrichi de texte externe.

- **Type** : CTC caractère + LM n-gram par langue (pyctcdecode)
- **Langues** : swa, som, kik, luo, kln, mas
- **Licence** : Apache-2.0 (base : w2v-bert-2.0, MIT)

## Entraînement

| Étape | Données | Méthode |
|---|---|---|
| v2 (base) | ~13k clips, 6 langues | fine-tuning multilingue, temperature sampling T=1.5 |
| v3 | + langues faibles | fine-tuning doux LR 1e-5 |
| v4 | 60k clips | SpecAugment doux (F=8,T=10), filtrage >20s (jamais de troncature), LR 1e-5, 5 époques |
| v5-t0→t3 | 4 tranches de ~75k clips frais chacune | 1 époque/tranche, chaîne de modèles, arrêt au plateau (gain<0.005) |

KenLM v2 : transcriptions ×3 + Wikipedia (sw 607k / so 91k / ki 11k phrases), filtrage
par l'alphabet exact (67 caractères), lmplz -o 5 --prune 0 0 1. Alpha 0.5 / beta 1.0
(grille sur logits d'éval en cache).

## Performance

| Mesure | Valeur |
|---|---|
| Leaderboard public | **0.42** |
| Macro greedy (éval interne) | 0.462 |
| Macro + KenLM v2 (éval interne) | 0.30 |

Par langue (greedy, éval interne) : swa 0.103 · kik 0.281 · som 0.331 · kln 0.653 ·
mas 0.690 · luo 0.714. Échantillons d'éval petits (45-60/langue) : valeurs bruitées,
les deltas entre versions sont plus fiables que les niveaux absolus.

## Limites et biais connus

- **Disparité entre langues** : swa ≈ 0.10 vs kln/mas/luo > 0.6 (greedy). Fiabilité
  nettement moindre sur les langues à faible ressource.
- **KenLM et mots hors corpus** : le LM peut imposer une forme fréquente erronée sur
  noms propres et mots rares ; le texte externe (Wikipedia) réduit mais n'élimine pas.
- **Registre** : entraîné sur lectures et conversations guidées ; robustesse non garantie
  en environnement très bruité ou parole très spontanée.
- **kln/mas sans texte externe** : aucun corpus public trouvé ; leurs LM restent limités
  aux transcriptions.
- **Orthographe variable** (somali notamment) : une partie du WER reflète des variantes
  d'écriture légitimes plutôt que des erreurs d'écoute.

## Usage prévu

Transcription hors-ligne sur appareil embarqué (CPU, ≤ 8 Go RAM), langue connue par
audio. Non destiné aux usages critiques (médical, juridique).

## Empreinte

Modèle 0.31 Go (fp32) + KenLM swa v2 0.82 Go ≈ **1.13 Go** RAM. RTF CPU 4 threads :
0.28 moyen / 0.74 max. Voir HARDWARE_VALIDATION.md.
