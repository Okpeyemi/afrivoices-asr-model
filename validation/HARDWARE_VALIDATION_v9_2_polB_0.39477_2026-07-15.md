# Validation matérielle — soumission v9_2_polB_0.39477

*Mesuré le 2026-07-15, session CPU fraîche, pipeline identique à la soumission (modèle, KenLM, (α, β)=(0.5, 0.0), troncature anti-padding, décodage direct avec repli fenêtré intégré au-delà du seuil mémoire).*

## Matériel de mesure
- CPU : Intel(R) Xeon(R) CPU @ 2.20GHz (8 cœurs logiques, **4 threads utilisés** — régime edge)
- RAM totale : 54.8 Go | OS : Linux-6.6.122+-x86_64-with-glibc2.35

## Conformité

| Critère | Limite | Mesuré | Statut |
|---|---|---|---|
| Paramètres | ≤ 1 Md | 0.606 Md | ✓ |
| RAM plancher (modèle fp32 + pire KenLM, avant inférence) | ≤ 8 Go | 1.46 Go | ✓ |
| RAM pic — clips ≤ 60s, décodage direct | ≤ 8 Go | 6.71 Go | ✓ |
| RAM pic — clips > 60s, **repli fenêtré** | ≤ 8 Go | 1.80 Go | ✓ |
| RTF — clips ≤ 60s, direct (agrégé / p95 / max) | ≤ 2× | 0.754 / 0.991 / 1.484 | ✓ |
| RTF — clips > 60s, repli fenêtré (mesuré, 95s) | ≤ 2× | 0.688 | ✓ |

## Mémoire : pic = f(durée du clip) — décodage direct

Le transitoire d'attention du conformer croît en T². Pic « processus propre » = plancher (1.46 Go) + delta transitoire mesuré :

| durée | Δ pic transitoire | pic processus propre | ≤ 8 Go |
|---|---|---|---|
| 10s | 0.06 Go | 1.52 Go | ✓ |
| 20s | 0.57 Go | 2.03 Go | ✓ |
| 30s | 1.29 Go | 2.75 Go | ✓ |
| 45s | 2.95 Go | 4.41 Go | ✓ |
| 60s | 5.25 Go | 6.71 Go | ✓ |
| 90s | 11.84 Go | 13.30 Go | ✗ |

**Seuil de conformité en décodage direct : ~60s.** Au-delà, le générateur bascule automatiquement sur le **repli fenêtré** (fenêtres ≤ 16 s + recollage des logits) : pic 1.80 Go et RTF 0.688 mesurés sur un clip de 95 s — le repli résout simultanément la contrainte mémoire ET la contrainte de vitesse (le RTF direct d'un clip de 90 s atteint 2.13 : régime non déployé). Coût qualité du repli mesuré au préalable (RAPPORT Partie II §3) : appliqué à tous les clips > 20 s il coûterait ≈ +0.007 de macro ; appliqué au-delà du seuil seulement, son impact est inférieur. Sur GPU / machines à grande RAM, le décodage direct s'applique partout (configuration de la soumission).

## Latence d'inférence sur l'intégralité du test
- Durée audio totale : **467.2 h** (estimation sur 601 clips (moy 40.3s ± IC95 28.5 h sur le total))
- Répartition du temps d'audio : 39.3% en clips ≤ 60s (direct, RTF agrégé 0.754), 60.7% en clips > 60s (repli, RTF 0.688)
- **Latence projetée : ≈ 333.4 h** sur CPU 4 threads (41 733 clips, RTF déployé 0.714)

## Détail RTF par (langue × type) — décodage direct, échantillon stratifié n=110
| case | n | durée moy | RTF moy | RTF max |
|---|---|---|---|---|
| kik/scripted | 10 | 5.6s | 0.426 | 0.555 |
| kik/unscripted | 10 | 66.0s | 1.343 | 2.126 |
| kln/scripted | 10 | 5.0s | 0.469 | 0.670 |
| kln/unscripted | 10 | 78.3s | 1.530 | 2.017 |
| luo/scripted | 10 | 13.7s | 0.527 | 0.936 |
| luo/unscripted | 10 | 52.0s | 1.119 | 1.925 |
| mas/scripted | 10 | 6.2s | 0.440 | 0.538 |
| mas/unscripted | 10 | 43.7s | 0.938 | 1.681 |
| som/scripted | 10 | 6.0s | 0.414 | 0.517 |
| som/unscripted | 10 | 57.8s | 1.200 | 1.899 |
| swa/unscripted | 10 | 19.6s | 0.654 | 1.484 |

## Méthode
- Pic RSS échantillonné à 50 ms PENDANT chaque passe (psutil, thread dédié). Session fraîche : plancher = base 0.30 → +modèle 0.61 → +KenLM swa 1.46 Go (mmap, pire cas compté).
- RTF = temps mur (features + forward CPU + beam pyctcdecode) / durée, par clip. Le tableau détaillé documente le décodage direct sur toutes durées ; la conformité s'évalue par régime déployé (direct ≤ 60s, repli au-delà).
- Latence totale = durée audio × RTF pondéré par régime (répartition des durées mesurée sur l'échantillon de durée).
- Chargements : modèle 46 s, décodeur swa 28 s (hors RTF).