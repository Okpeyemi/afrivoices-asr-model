# 5_explorations — leviers mesurés puis écartés

Chaque notebook ici a été exécuté, mesuré, et son verdict est chiffré dans le
RAPPORT (Partie II). Ils sont conservés parce que la démarche — y compris ses
impasses — fait partie de la solution.

| Notebook | Question posée | Verdict |
|---|---|---|
| `afrivoices_p0_devlong_padding_fenetres.ipynb` | Les clips > 20s (29 % du test) perdent-ils au décodage direct ? Le padding en batch coûte-t-il ? | **Direct gagne partout** (le conformer extrapole en longueur) ; padding réel mais bénin (troncature adoptée). |
| `afrivoices_soumission_fenetres.ipynb` (historique) | Fenêtrage 15s/2s + recollage textuel | Perdant : duplications au recollage, projection +0.026. Jamais soumis. |
| `afrivoices_soumission_fenetres_v2.ipynb` | Fenêtrage amélioré (coupes au silence, recollage des logits) | Meilleur que v1, **toujours derrière le direct** (+0.007 projeté). Conservé comme secours OOM dans le générateur final. |
| `afrivoices_soumission_hybride.ipynb` (historique) | Routage par langue entre deux checkpoints (ère v5/v6) | Gagnant à son époque (0.3408 dev), supplanté par la lignée v9. |
| `afrivoices_retrain_v10_unscripted.ipynb` | Réentraîner sur le spontané (85-94 % exclu par le filtre 20s ; alignement forcé → ressource ×25) | Gain dev massif **non transféré** (+0.031 LB) : dev à 2-6 locuteurs/case, sur-ajustement aux voix. Levier réel, instrument insuffisant. |

Règle générale apprise (Partie II §2) : à ces effectifs, seuls les changements
**globaux et corroborés plusieurs fois** se transfèrent au leaderboard.
