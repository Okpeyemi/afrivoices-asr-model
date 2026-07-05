# KenLM v2 (modèles de langue)

6 binaires : swa.bin (819 Mo), kik.bin (207), luo.bin (157), kln.bin (142),
mas.bin (163), som.bin (164) + les corpus .txt correspondants (pour les unigrams
pyctcdecode). Hébergés en Kaggle Dataset (lien : à compléter) — trop volumineux
pour un notebook.

Régénération complète : notebooks/afrivoices_kenlm_v2.ipynb (session CPU).
Réglage de décodage : alpha = 0.5, beta = 1.0 (grille sur éval interne).
