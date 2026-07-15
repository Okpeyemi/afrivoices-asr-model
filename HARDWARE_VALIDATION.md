# Rapport de validation hardware — AfriVoices East Africa ASR

Document de conformité aux contraintes de déploiement embarqué (edge) de la compétition —
**méthodologie de mesure**. Les rapports chiffrés datés, un **par soumission**, sont dans
`validation/` (générés par `notebooks/4_analyses/afrivoices_validation_materielle.ipynb`) ;
les valeurs ci-dessous reprennent la configuration finale (v9-2, leaderboard 0.39477).

## Conditions de test

- **Cible** : CPU embarqué (Raspberry Pi 4 / Android), 100 % hors-ligne.
- **Simulation** : exécution **CPU uniquement**, `torch.set_num_threads(4)` pour reproduire
  les 4 cœurs d'un Raspberry Pi 4.
- **Pipeline mesuré** : chaîne complète = modèle acoustique (w2v-BERT 2.0 + CTC) **+
  décodage KenLM par langue**. Le décodage LM est inclus car c'est la partie la plus
  coûteuse sur CPU ; le mesurer garantit un RTF honnête.
- **Audios** : échantillon représentatif de l'ensemble de test (durées variées).

## 1. Latence — Real-Time Factor (RTF)

Le RTF = temps de traitement / durée de l'audio. **Limite de la compétition : RTF ≤ 2×.**
La mesure est faite **par régime déployé** : décodage direct pour les clips ≤ 60 s, repli
fenêtré au-delà (le RTF direct d'un clip de 90 s atteindrait ~2,1× — régime non déployé).

| Régime | Mesure | Valeur |
|---|---|---|
| Direct (≤ 60 s) | RTF agrégé | 0,75 |
| Direct (≤ 60 s) | RTF p95 | 0,99 |
| Direct (≤ 60 s) | RTF max | 1,48 |
| Repli fenêtré (> 60 s, clip de 95 s) | RTF | 0,69 |

**Conforme** : tous les régimes déployés restent ≤ 2×. Le pire cas direct (1,48× sur les
clips spontanés longs) garde une marge, et au-delà de ~60 s le repli fenêtré ramène le RTF
à 0,69. Latence projetée ≈ **333 h** pour les 467 h d'audio du test (CPU 4 threads).

## 2. Empreinte mémoire (RAM)

Mesure d'un **processus Python isolé** chargeant uniquement les artefacts de déploiement
(pour éviter de compter la RAM résiduelle d'une session encombrée).
**Limite de la compétition : RAM ≤ 8 Go.**

| Composant / régime | RAM (Go) |
|---|---|
| Base Python | 0,30 |
| + Modèle acoustique (fp32, CPU) | +0,31 → 0,61 |
| + KenLM v2 (swahili, le plus gros des 6, mmap) | → 1,46 |
| **Plancher (modèle + 1 KenLM v2, avant inférence)** | **1,46** |
| Pic — décodage direct à 60 s (transitoire d'attention T²) | 6,71 |
| Pic — clips > 60 s, repli fenêtré | 1,80 |

**Conforme** : le transitoire d'attention du conformer croît en T². En décodage direct, le
pic atteint 6,71 Go à 60 s (≤ 8 Go) — **seuil de conformité direct ~60 s** ; au-delà, le
générateur bascule automatiquement sur le **repli fenêtré** (fenêtres ≤ 16 s + recollage
des logits), dont le pic retombe à 1,80 Go. Tous les régimes déployés restent ≤ 8 Go.

**Hypothèse de déploiement** : un seul KenLM est chargé à la fois, celui de la langue de
l'audio courant (la langue est fournie dans les données de test). Le swahili étant le plus
volumineux des six modèles de langue (~740 Mo de binaire), ce total représente le **pire
cas mémoire**. Toutes les autres langues consomment moins.

*Mise à jour v2 : le KenLM swahili enrichi (lm_v2) pèse 819 Mo (vs ~740 Mo en v1) ;
le plancher mémoire reste ≈ 1,46 Go, très loin de la limite.*

Une marge supplémentaire existe : le modèle pourrait être quantifié en INT8 (~4× plus
petit, soit ~0,1 Go), mais ce n'est pas nécessaire pour la conformité.

## 3. Autres contraintes

| Critère | Limite | Mesuré | Statut |
|---|---|---|---|
| Paramètres du modèle | ≤ 1 Md | ≈ 0,6 Md | ✓ |
| Exécution hors-ligne | requise | aucune dépendance réseau à l'inférence | ✓ |
| CPU uniquement | requis | testé sans GPU | ✓ |

## 4. Conclusion

Le système respecte l'ensemble des contraintes de déploiement edge :

- RTF ≤ 2× sur tous les régimes déployés (direct ≤ 60 s : 0,75 agrégé / 1,48 max ;
  repli fenêtré : 0,69) ✓
- RAM plancher 1,46 Go, pic 6,71 Go en direct ≤ 60 s puis repli fenêtré à 1,80 Go
  — **≤ 8 Go** ✓
- ≈ 0,6 Md paramètres **≤ 1 Md** ✓
- 100 % hors-ligne, CPU uniquement ✓

Le modèle est déployable sur un Raspberry Pi 4 (8 Go) ou un appareil Android de gamme
moyenne : le basculement automatique direct → repli fenêtré au-delà de ~60 s maintient
l'empreinte sous la limite quelle que soit la durée du clip.

---

*Méthodologie de mesure reproductible dans le notebook principal. Les valeurs RAM sont
issues d'une mesure en processus isolé (psutil RSS) ; les valeurs RTF d'une passe CPU
bridée à 4 threads sur échantillon de test.*
