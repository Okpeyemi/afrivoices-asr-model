# Model Card — AfriVoices East Africa ASR (v9-2 + KenLM v2)

## Description

ASR multilingue unique pour six langues est-africaines, déployable hors-ligne sur CPU
embarqué. Modèle acoustique w2v-BERT 2.0 + CTC (vocabulaire de caractères partagé),
rescoring KenLM 5-gram par langue enrichi de texte externe.

- **Type** : CTC caractère + LM n-gram par langue (pyctcdecode)
- **Langues** : swa, som, kik, luo, kln, mas
- **Licence** : Apache-2.0 (base : w2v-bert-2.0, MIT)

## Entraînement (chaîne progressive par tranches)

| Étape | Méthode |
|---|---|
| v2→v4 | fine-tuning multilingue, temperature sampling, SpecAugment doux, filtrage >20s |
| v5 (t0→t3) | 4 tranches de données fraîches (~75k clips), 1 époque/tranche |
| v6 (t0→t1) | tranches sur données officielles Anv-ke (dev officiel comme éval) |
| v7-soup | fusion de poids v5-t3 + v6-t1 (model soup, λ=0.3) |
| v8-cible | tranche ciblée langues faibles (kln/mas/som ~70% du mix) |
| v9-bicible | tranche bi-ciblée kln+mas (64% du mix), départ v8-cible |
| **v9-2** | tranche 4 supplémentaire, ciblage kln renforcé, départ v9-bicible |

KenLM v2 : transcriptions ×3 + Wikipedia (sw 607k / so 91k / ki 11k phrases), filtrage
par l'alphabet exact (67 caractères), lmplz -o 5 --prune 0 0 1. α 0.7 / β 0.5.

## Performance

| Mesure | Valeur |
|---|---|
| Leaderboard public (WER macro) | **0.40283** |
| CER (caractères) | ≈ 0.094 |
| Macro dev + KenLM v2 | ≈ 0.31 |

Par langue (dev officiel + LM) : swa ~0.10 · luo 0.24 · som 0.27 · kik 0.31 · mas 0.46 ·
kln 0.48. Échantillons dev petits (~60/langue) : les deltas entre versions sont plus
fiables que les niveaux absolus. Règle de conversion observée : leaderboard ≈ dev+LM + 0.10.

## Limites et biais connus

- **Disparité entre langues** : swa/luo « faciles » vs kln/mas > 0.45. Fiabilité moindre
  sur les langues nilotiques à faible ressource.
- **Plafond acoustique kln/mas** : ~35 % des erreurs sont des substitutions franches (le
  modèle entend mal), non corrigeables par le LM. Le maasai est particulièrement dur
  (données majoritairement spontanées/unscripted).
- **Segmentation de mots** : CER 0.094 mais WER 0.31+ → une grande part du WER vient de
  frontières de mots en désaccord avec la référence, irrégulières donc non normalisables.
- **kln/mas sans texte externe** : LM limité aux transcriptions (pas de Wikipedia).
- **Orthographe variable (somali)** : une partie du WER reflète des variantes légitimes.

## Usage prévu

Transcription hors-ligne sur appareil embarqué (CPU, ≤ 8 Go RAM), langue connue par
audio. Non destiné aux usages critiques (médical, juridique).

## Empreinte

Modèle 0.31 Go (fp32) + KenLM swa v2 0.82 Go ≈ **1.13 Go** RAM. RTF CPU 4 threads :
0.28 moyen / 0.74 max. Voir HARDWARE_VALIDATION.md.
