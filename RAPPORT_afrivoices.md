# AfriVoices East Africa — ASR Hackathon : rapport de démarche

Système de reconnaissance vocale automatique **unique**, **hors-ligne** et **embarquable**
(CPU, ≤ 8 Go RAM) couvrant six langues est-africaines : swahili (swa), somali (som),
kikuyu (kik), luo (luo), kalenjin (kln), maasai (mas). Ce document retrace la démarche
complète : les choix, leurs raisons, les résultats mesurés, et les impasses — car les
impasses documentées font partie du travail d'ingénierie.

> **Structure.** Ce rapport a deux parties. **Partie I** retrace la construction du
> système jusqu'au plateau de 0,40283 (chaîne acoustique v2 → v9-2, KenLM v1 → v2,
> audits et calibration). **Partie II** documente la campagne de mesure menée après le
> gel de l'acoustique, qui a porté le score à **0,39477** sans réentraînement retenu.

---

# Partie I — Construction du système (jusqu'au plateau 0,40283)

## 1. Contraintes et métrique

**Contraintes edge (éliminatoires)** : un seul modèle ≤ 1 milliard de paramètres, RAM
≤ 8 Go, CPU uniquement, hors-ligne, facteur temps-réel (RTF) ≤ 2×, et licence permissive
sans restriction commerciale (règle 6c).

**Métrique** : le score est la **moyenne non pondérée du WER (Word Error Rate) sur les 6
langues** — chaque langue compte pour 1/6, indépendamment de sa fréquence dans le test.
Conséquence stratégique majeure : les langues faibles pèsent autant que les fortes, donc
tout gain sur kln/mas (WER élevé) vaut autant qu'un gain sur swa (WER bas), et il y a
beaucoup plus à y gagner.

**Répartition du test** : 41 733 clips, 94 fichiers parquet, la langue de chaque clip
étant fournie (ce qui autorise un routage par langue à l'inférence).

---

## 2. Architecture retenue : deux étages indépendants

Le choix de fond a été un système **CTC à caractères + modèle de langue n-gram**, plutôt
qu'un modèle autorégressif. Raison : le CTC fait une seule passe (RTF faible, crucial pour
la contrainte edge), et il permet de brancher un KenLM externe — un levier de correction
puissant et gratuit en calcul.

**Étage 1 — modèle acoustique** : `facebook/w2v-bert-2.0` (licence MIT, 606M paramètres,
pré-entraîné sur 4,5M h / 143+ langues) avec une tête CTC et un **vocabulaire de 67
caractères partagé** entre les six langues. Le partage du vocabulaire permet le transfert
entre langues proches (les trois langues nilotiques kln/luo/mas partagent des sons).

**Étage 2 — modèles de langue KenLM** : un 5-gram par langue, appliqué au décodage via
pyctcdecode. Le LM corrige les séquences de caractères en mots plausibles.

Ces deux étages sont **indépendants** : on peut améliorer l'un sans retoucher l'autre, et
les régler séparément. C'est ce qui a permis d'itérer vite.

---

## 3. Le modèle acoustique : entraînement par tranches

**Problème central** : le dataset complet (~750 Go d'audio) dépasse de loin le disque
local disponible (~235 Go). Impossible de tout charger.

**Solution — l'entraînement par tranches** : parcourir les données en tranches successives
de ~60-80k clips, chacune entraînée pendant une époque, en chaînant les modèles
(chaque tranche part du modèle produit par la précédente). Techniquement, cela a demandé :
- un chargement par paquets de 8 parquets avec purge du cache brut entre paquets (sinon
  saturation disque) ;
- un scan « zéro-copie » (lecture des durées et des labels sans dupliquer l'audio) ;
- `Audio(decode=False)` obligatoire (sinon décodage automatique = explosion mémoire) ;
- un échantillonnage par **température** (T=1.6) pour équilibrer les langues sans écraser
  les rares.

**Réglages validés par l'échec** : les langues faibles ont d'abord souffert d'un
SpecAugment trop agressif et d'une troncature des audios longs (cibles impossibles). La
version qui a fonctionné : SpecAugment doux, **filtrage des audios > 20s** (jamais de
troncature), learning rate 1e-5, batch effectif 16 (4×4). Cette configuration a stabilisé
la vitesse à ~0,40 it/s et le WER à un niveau sain.

---

## 4. Le modèle de langue : KenLM v1 → v2 enrichi

**KenLM v1** : n-grams entraînés sur les seules transcriptions d'entraînement. Gain
immédiat et massif au premier branchement (0,566 → 0,469 au leaderboard).

**KenLM v2 — enrichissement par texte externe** : les transcriptions seules limitent le
vocabulaire du LM. On a ajouté du texte externe filtré par l'alphabet exact du modèle :
Wikipedia swahili (607k phrases), somali (91k), kikuyu (11k), avec les transcriptions
répétées ×3 pour préserver la dominance du registre oral. Résultat : le LM v2 a battu le
v1 sur le dev, et l'optimum des hyperparamètres de décodage (alpha, beta) s'est déplacé —
d'où la nécessité de les re-régler par grille à chaque changement de LM.

**Limite assumée** : kalenjin et maasai n'ont aucun corpus texte public (pas de Wikipedia).
Leur LM reste limité aux transcriptions. Ces deux langues couvrent pourtant une part
importante du score (macro), ce qui a pesé sur le plafond final.

---

## 5. Le raffinement du modèle acoustique (v5 → v9)

Une fois le pipeline stable, la progression a suivi une logique de **ciblage croissant**,
guidée systématiquement par la mesure **avec LM** (jamais par le décodage glouton seul —
leçon répétée : le glouton est trompeur, le KenLM récupère des erreurs que le glouton
montre).

| Version | Méthode | Raison |
|---|---|---|
| v5-t3 | 4 tranches de données fraîches | épuiser le gain du volume |
| v6 | tranches sur les données **officielles Anv-ke** | données mises à jour par les organisateurs ; le split `dev/` officiel est devenu l'éval (plus prédictif du leaderboard) |
| v7-soup | fusion de poids v5-t3 + v6-t1 | « model soup » — un seul modèle héritant des forces des deux, sans réentraînement |
| v8-cible | tranche ciblée langues faibles (kln/mas/som ~70%) | le score macro récompense les langues faibles |
| v9-bicible | tranche bi-ciblée kln+mas (64%) | concentrer sur les deux pires langues |
| v9-2 | tranche 4 supplémentaire, ciblage kln | kln continuait de répondre |

**Découverte clé du ciblage** : v8-cible avait un décodage glouton *mitigé* (kik/luo
légèrement érodés) mais un résultat *avec LM* nettement meilleur (0,3178 sur le dev, record).
Cela a confirmé la règle méthodologique : **toujours décider avec le LM**.

**Model soup** : la fusion de poids (moyenne pondérée λ=0,3·v6t1 + 0,7·v5t3) a produit un
modèle unique meilleur que chacun de ses parents — un gain gratuit, sans entraînement.

---

## 6. Calibration dev ↔ leaderboard

Point méthodologique important : l'éval interne et le leaderboard ne coïncident pas, mais
leur écart est **stable et mesurable**. Après recalage sur les scores réels :

> **leaderboard ≈ (macro dev avec LM) + 0,10**

Cette règle a permis d'estimer chaque modèle sans gaspiller de soumission, et de fixer des
objectifs réalistes (viser 0,31 au leaderboard aurait demandé 0,21 au dev — hors d'atteinte
avec cette architecture).

---

## 7. L'audit d'erreurs : le diagnostic décisif

Plutôt que de continuer à deviner l'origine du WER, un **audit systématique** des erreurs
a été mené : transcription du dev, alignement mot-à-mot avec les références, catégorisation
de chaque écart. Résultats :

- **CER (taux d'erreur caractère) ≈ 0,094** : le modèle reconnaît correctement ~91 % des
  caractères. **Le modèle entend excellemment bien.** Le problème n'est pas acoustique.
- **~39 % des erreurs sont des substitutions « proches »** (1-2 caractères : diacritiques,
  variantes orthographiques) ; **~28 % sont des insertions/suppressions de segmentation**
  (frontières de mots en désaccord avec la référence) ; seulement **~32 % sont des erreurs
  « franches »** (vraie confusion acoustique), concentrées sur kln/mas.

**Conclusion de l'audit** : le WER-mot élevé (malgré un CER excellent) provient surtout
d'un désaccord de **segmentation des mots** et de **variantes orthographiques**, pas d'une
mauvaise écoute. Cela a orienté toute la suite et écarté définitivement le besoin d'un
modèle acoustique plus puissant (omniASR) pour l'essentiel des langues.

---

## 8. Les leviers testés puis écartés (par la mesure)

La rigueur a consisté à tester chaque hypothèse et à l'abandonner dès que la mesure
l'invalidait, plutôt que de s'entêter :

- **Volume de données supplémentaire** → plateau atteint dès v6 (rendements décroissants).
- **Réglage de beta (segmentation)** → grille étendue {0,5 … 5,0} testée ; beta=0,5 reste
  optimal. La segmentation est *irrégulière*, donc non corrigeable par un paramètre global.
- **Normalisation de la prédiction** → toutes les variantes testées *augmentaient* le WER
  (la référence Kaggle n'est pas normalisable à l'aveugle).
- **Registre scripted/unscripted** → vérifié : l'entraînement était déjà bien réparti,
  cohérent avec le dev.
- **KenLM plus costaud (ordre 6, corpus étendu)** → les corpus kln/mas étaient déjà
  saturés (244k et 162k lignes) ; aucun gain possible.
- **Découpage en fenêtres des audios longs** → dégrade le score (coupe au milieu des mots) ;
  la bonne approche serait un découpage aux silences (VAD), non implémenté faute de temps.
- **omniASR-CTC-1B (Meta, Apache-2.0)** → éligible et conforme (RTF CPU 1,12 mesuré), mais
  son zero-shot était *pire* que le modèle fine-tuné (macro 0,66 vs 0,49), et le fine-tuning
  passe par l'écosystème fairseq2 (chantier lourd). Écarté faute de rapport gain/temps
  favorable, d'autant que l'audit avait montré que l'acoustique n'était pas le problème.
- **MMS (Meta)** → **inéligible** : licence CC-BY-NC (non-commercial), qui se propage aux
  modèles fine-tunés (œuvre dérivée). Vérification de licence cruciale.

---

## 9. Le levier des audios longs (dernière piste active)

Une observation fortuite (vérification d'un possible double encodage, finalement inexistant)
a révélé que **le test contient des audios très longs** : jusqu'à 100s, alors que le modèle
n'a jamais vu de clip > 20s à l'entraînement. Mesure de la proportion par langue :

| Langue | % clips > 20s |
|---|---|
| swa | 41,5 % |
| mas | 36,1 % |
| kln | 31,7 % |
| som | 22,9 % |
| luo | 19,6 % |
| kik | 17,8 % |
| **Global** | **29,0 %** |

**29 % du test dépasse 20s**, et les proportions les plus fortes sont sur mas et kln — les
langues les plus faibles. Hypothèse : le modèle décroche sur ces longueurs jamais vues.
**Solution testée** : à l'inférence, découper les clips > 20s en fenêtres de 15s (avec 2s
de recouvrement), transcrire chaque fenêtre dans le régime maîtrisé, puis recoller les
textes en dédupliquant la zone de chevauchement. Levier entièrement **gratuit** (aucun
réentraînement), ciblant précisément les langues faibles.

**Résultat : échec (0,40283 → 0,42737).** Le découpage en fenêtres fixes a *dégradé* le
score. Explication : le modèle a appris à transcrire des unités de sens complètes ; couper
un audio long à intervalles fixes tombe **au milieu des mots et des phrases**. Les mots à
cheval sur deux fenêtres sont massacrés dans les deux, le recollage heuristique échoue à
retrouver le bon point de jointure (fragments de frontière mal transcrits), et le KenLM
travaille sur des débuts/fins de phrase artificiels. Le découpage crée plus d'erreurs de
frontière qu'il n'en résout de longueur.

**Piste future (non implémentée faute de temps)** : découper **aux silences** via une
détection d'activité vocale (VAD), pour couper *entre* les mots et non au milieu. C'est la
bonne façon de traiter les audios longs, mais elle demande d'intégrer un détecteur de
pauses et une gestion de segments — un chantier à part entière.

---

## 10. Résultats (leaderboard public, WER macro)

| Système | Leaderboard |
|---|---|
| v2, décodage glouton | 0,566 |
| v2 + KenLM v1 | 0,469 |
| v3 + KenLM v1 (α 0,7) | 0,455 |
| v5-t3 + KenLM v2 (α 0,5) | 0,42 |
| v9-bicible + KenLM v2 | 0,41477 |
| **v9-2 (tranche 4) + KenLM v2** | **0,40283** |
| v9-2 + découpage en fenêtres | *(en évaluation)* |

Progression totale : **de 0,566 à 0,403**, soit une réduction de ~29 % du WER, à contraintes
edge constantes.

---

## 11. Conformité edge (validée)

| Critère | Limite | Mesuré | Statut |
|---|---|---|---|
| Paramètres | ≤ 1 Md | 0,606 Md | ✓ |
| RAM (modèle + KenLM swahili) | ≤ 8 Go | ≈ 1,13 Go | ✓ |
| RTF (CPU 4 threads) | ≤ 2× | 0,28 moy / 0,74 max | ✓ |
| Hors-ligne, CPU | requis | oui | ✓ |
| Licence | OSI, commercial OK | Apache-2.0 | ✓ |

Toute la pile a été vérifiée sans restriction commerciale (modèle de base MIT, données
CC-BY, code Apache-2.0). Filet anti-fuite validé : 0 chevauchement entre les identifiants
du test et les données d'entraînement/dev.

---

## 12. Enseignements

1. **Décider avec le LM, pas au décodage glouton.** Le glouton a plusieurs fois donné un
   signal trompeur que le KenLM corrigeait.
2. **Mesurer avant de bâtir.** L'audit d'erreurs (CER 0,094) a évité un chantier omniASR
   inutile en prouvant que le problème n'était pas acoustique.
3. **Abandonner par la mesure.** Chaque levier écarté l'a été sur des chiffres, pas sur une
   intuition — ce qui donne la certitude d'avoir exploré l'espace des solutions.
4. **La formule de score dicte la stratégie.** Le macro (1/6 par langue) a justifié le
   ciblage des langues faibles plutôt qu'une optimisation globale.
5. **Les contraintes edge orientent l'architecture** dès le départ (CTC vs autorégressif,
   licence, taille) — un modèle plus performant mais inéligible (MMS) ou trop lourd ne sert
   à rien.

Le plafond atteint (~0,40) s'explique par des facteurs irréductibles à cette architecture :
la segmentation de mots (irrégulière) et la difficulté acoustique intrinsèque des langues
nilotiques à faible ressource. Le système livré reste un ASR edge complet, conforme et
méthodiquement optimisé sur six langues dont trois parmi les moins dotées au monde.

---

# Partie II — Campagne d'optimisation : du 0,40283 au 0,39477 par la mesure

> Cette partie documente la campagne menée après le gel de l'acoustique (v9-2).
> Aucun réentraînement n'a été retenu dans la solution finale : la totalité du gain
> (−0,008) provient de corrections du décodage et, surtout, de la construction puis
> de la **critique honnête** de nos propres instruments de mesure.

## 1. Résumé exécutif

| Soumission | Config | Leaderboard | Delta |
|---|---|---|---|
| Historique | v2-era, décodage glouton | 0,566 | — |
| Référence | v9-2 + KenLM v2, (α, β) = (0,7, 0,5) | 0,40283 | — |
| Contrôle « par type » | + troncature anti-padding, décodage direct | **0,39923** | −0,0036 |
| v10 (réentraînement spontané) | mesuré, **non retenu** | 0,43040 | +0,031 |
| Unigrams 200k sélectifs | mesuré, **non retenu** | 0,40050 | +0,0013 |
| Glouton (décomposition) | v9-2 sans LM | ≈0,45 | +0,05 |
| **Finale** | v9-2 + KenLM v2, **(α, β) = (0,5, 0,0)** | **0,39477** | **−0,0045** |

Décomposition finale du score : acoustique seule ≈ 0,45 ; contribution du modèle de
langue ≈ −0,055 ; corrections de décodage ≈ −0,008. La solution finale est intégralement
construite par l'auteur (modèle, corpus LM, décodeur, générateur de soumission).

## 2. Le banc de mesure — construction, puis limites démontrées

Le test fournit une colonne `type` (scripted/unscripted) par clip. Son exploitation a
permis de mesurer la composition exacte du test (41 733 clips) : la part de parole
spontanée varie de 18,6 % (kik) à 42,2 % (mas), et le swahili est **100 % spontané**
(descriptions d'images par domaine, distribution identique au corpus d'entraînement
swahili). Un dev stratifié par (langue × type), pondéré par ces parts exactes, a donné
pour v9-2 :

| case | scripted | unscripted |
|---|---|---|
| swa (mixte) | 0,060 | — |
| som | 0,243 | 0,625 |
| kik | 0,265 | 0,476 |
| luo | 0,205 | 0,328 |
| kln | 0,495 | 0,713 |
| mas | 0,429 | 0,567 |

Macro pondéré : **0,332**, pour un leaderboard à 0,403. Le résiduel (−0,071) s'explique
par la normalisation du scoring, l'effet locuteurs et l'échantillonnage — pas par la
longueur des clips ni par un bug (section 3).

**Limite structurelle, démontrée à nos dépens** : les cases du dev reposent sur 2 à 6
locuteurs (dev/train disjoints en voix, vérifié par `recorder_uuid`). L'instrument
prédit correctement les changements de *décodage* corroborés, mais pas les changements
de *modèle* : le réentraînement v10 était annoncé neutre (−0,001) et a coûté +0,031 au
leaderboard ; le réglage « unigrams 200k » était annoncé à −0,007 et a rendu +0,001.
Doctrine finale : raisonner en **deltas**, ne soumettre que des effets **corroborés
plusieurs fois** et appliqués **globalement** (sans sélection par case) — c'est ce
protocole qui a produit le gain final (annoncé −0,004, obtenu −0,0045).

## 3. Leviers explorés, mesurés, écartés

**Fenêtrage des clips longs.** 29 % du test dépasse 20 s ; l'hypothèse d'un décrochage
au-delà du régime d'entraînement s'est révélée fausse. Sur un dev-long construit
exprès (clips > 20 s, exclus de toutes les mesures antérieures), le décodage **direct
pleine longueur bat le fenêtrage partout** (ex. kik : 0,455 direct contre 0,775 pour le
recollage textuel et 0,681 pour un recollage amélioré au niveau des logits). Le
conformer extrapole bien en longueur ; découper détruit du contexte. Projection
leaderboard du fenêtrage : +0,007 à +0,026 — écarté sans soumission.

**Padding en batch.** Les soumissions décodaient des logits paddés (frames jamais
entraînées, loss CTC masquée) alors que toutes les mesures dev décodaient clip par
clip. Bug réel mais bénin : 2 clips divergents sur 120, WER inchangé à la 4ᵉ décimale.
La troncature à la vraie longueur (via attention_mask) a été intégrée au générateur ;
le contrôle a rendu −0,0036, cohérent avec un artefact mineur corrigé gratuitement.

**Grilles (α, β) par (langue × type).** Surfaces plates (±0,01 sur la zone α∈[0,3-0,7]),
optima instables à 80 clips/case : la politique « meilleur réglage par case » projetait
−0,004 mais chaque adoption individuelle était sous le bruit. Non soumise telle quelle ;
seule la **direction commune** aux deux grilles (moins de poids au LM) a été retenue —
section 4.

**Réentraînement ciblé spontané (v10).** Découverte majeure en chemin : le filtre
d'entraînement `durée ≤ 20 s`, hérité d'une contrainte disque, excluait **85 à 94 % de
la parole spontanée** (médianes 41-70 s selon la langue) — le modèle n'avait presque
jamais entendu le registre qui compose jusqu'à 42 % du test. Un alignement forcé CTC
(v9-2) suivi d'une découpe aux silences a converti les clips longs en segments ≤ 18 s
avec leur texte (< 1 % de rejets), multipliant la ressource spontanée utilisable par
~25 (kln : 182 clips → 4 973 segments). Le fine-tuning (74 % spontané / 16 % lu / 10 %
swa) a produit un gain massif **sur les locuteurs du dev** (kik spontané : 0,476 →
0,266) qui ne s'est **pas transféré** au test (+0,031) : sur-ajustement aux 4-10 voix
d'entraînement par langue. Le levier données reste réel ; il exige un dev multi-locuteurs
que le corpus ne permet pas de construire.

**Unigrams du décodeur.** Les langues agglutinantes comptent 109k-237k formes ; la
liste de 50k unigrams de pyctcdecode laissait 11-12 % des mots du dev hors liste
(kln, mas). L'élargissement à 200k, gagnant en mesure appariée (−0,007), a rendu
+0,001 au leaderboard — cas d'école du biais de sélection (meilleure combinaison,
langues choisies d'après le dev). Écarté, et versé à la doctrine de la section 2.

## 4. Le gain final : réduire le poids du LM sur un test à dominante spontanée

Trois éléments convergents ont motivé la configuration finale.

D'abord, **deux grilles indépendantes pointaient la même direction** : sur un dev à
dominante spontanée, quatre langues sur six avaient leur optimum exactement en
(0,5, 0,0) ; sur le dev stratifié pondéré, la politique globale (0,5, 0,0) projetait
−0,004 quand aucune adoption par case ne sortait du bruit.

Ensuite, **le mécanisme est identifiable**. Le KenLM v2 est nourri de transcriptions
lues (×3) et de texte encyclopédique — un registre écrit/lu. Un contrôle de
contamination du corpus a en outre montré qu'en scripted, plusieurs locuteurs lisent
les **mêmes prompts** : 5 à 17 références dev sur 150 sont verbatim dans le corpus
d'entraînement du LM (contre 0 en spontané). Le LM aide donc d'autant plus le scripted
qu'il en connaît littéralement une partie ; sur le spontané — jusqu'à 42 % du test —
son prior guide mal, et un poids élevé impose au beam des hypothèses du mauvais
registre.

Enfin, le **protocole anti-sur-ajustement** de la section 2 : changement global (les
six langues, les deux types — aucune sélection par case), corroborations multiples,
une seule soumission. Résultat : **(α, β) = (0,5, 0,0)** → **0,39477**, conforme à la
prédiction (annoncé −0,004, obtenu −0,0045). Interprétation : sur ce test, l'acoustique
de v9-2 est plus fiable que le prior du LM ; réduire son poids rend la décision au bon
juge.

## 5. Conformité, reproductibilité, hygiène

- **Edge** : modèle 606 M (fp32 ~2,4 Go) + KenLM mmappé (max 0,82 Go) ≪ 8 Go ; RTF
  par clip inchangé (le parallélisme du générateur accélère la production du CSV,
  pas le coût unitaire). Conformément au règlement, un rapport de validation matérielle
  daté est généré par `afrivoices_validation_materielle.ipynb` (seuil de repli ~60 s →
  RSS pic 1,80 Go et RTF 0,688 ; latence projetée 333 h sur les 467 h d'audio du test).
- **KenLM v2 documenté** : recette `lmplz -o 5 --discount_fallback --prune 0 0 1`,
  corpus = transcriptions train ×3 + Wikipedia sw/so/ki + MasakhaNEWS luo ;
  listes d'unigrams : top-50k par langue.
- **Générateur final** : `afrivoices_soumission_par_type.ipynb` (routage par type,
  troncature, décodage direct, reprise automatique, asserts 41 733) — config par
  défaut = soumission finale.
- **Intégrité de la soumission** : 7 clips du test aux bytes audio illisibles
  (tous lecteurs), transcrits « _ » par conception (impact ≤ 0,0002) :
  `SmEs1odilz_20Dec2024034114GMT_1734666074667.wav` (kik),
  `SHEggxiUvQ_29Dec2024142904GMT_1735482544464.wav`,
  `TR3zDQP2xB_4Jan2025081325GMT_1735978405014.wav` (kln),
  `SEfGgWbeoe_13Dec2024101448GMT_1734084888302.wav`,
  `TFHDGbA3YC_5Feb2025094551GMT_1738748751488.wav` (luo),
  `fdrEs5QOqD_20Dec2024211737GMT_1734729457557.wav`,
  `St7ulG8J9G_2Jan2025080441GMT_1735805081046.wav` (som).
- **Note d'environnement** : sur les images Colab récentes, installer pyctcdecode en
  `--no-deps` (+ pygtrie) pour ne pas rétrograder numpy (incompatibilité binaire sinon).

## 6. Limites et suites

Le plancher lexical est mesuré : même à 200k unigrams, 7-8 % des mots spontanés
kln/mas sont **absents du corpus** — aucun LM dérivé des transcriptions ne peut les
produire ; il faudrait du texte externe dans ces langues (piste : corpus bibliques
sous licence permissive). Le levier « données spontanées » (×25 de ressource via
l'alignement forcé) reste ouvert mais exige un dev multi-locuteurs pour être validé.
Enfin, le swahili dispose d'un vrai split dev (par domaine) non exploité ici — notre
case swa, découpée dans le train, partage potentiellement des locuteurs et reste la
mesure la moins fiable du banc.
