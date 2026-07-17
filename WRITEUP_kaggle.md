# A single edge-deployable ASR for 6 East African languages

*One `w2v-bert-2.0` + character-CTC model with a per-language KenLM, running offline on CPU (≤ 8 GB RAM) across all six languages.*

Hi everyone! 👋 First, a big thank you to Digital Umuganda and the KenCorpus Consortium for the data and for organizing this hackathon 🏁, and to everyone who took part. Working across Swahili, Kikuyu, Dholuo, Kalenjin, Maasai and Somali — three of them among the lowest-resourced languages in the world — was a genuinely great ride.

In this post I share my full approach: what worked, what did **not** work (the useful part 🙂), and links to the model and code. The honest short version: most of my late-stage gains came not from a bigger model, but from **building — then distrusting — my own measurements**.

## 🟢 Background

The brief had two hard parts: six languages under **one** model, and a real **edge** constraint (single model ≤ 1 B params, ≤ 8 GB RAM, CPU-only, offline, RTF ≤ 2×, permissive license). The macro metric (unweighted mean of per-language WER) means the hard low-resource languages — Kalenjin and Maasai — dominate the score, so that is where the leverage is.

## 🎯 The solution

**TLDR:** `w2v-bert-2.0` speech encoder + character-level CTC, decoded with a per-language KenLM, engineered to run on a CPU / edge device. Final model: `baobab-asr-v9-2`.

- **Pretrained model:** `facebook/w2v-bert-2.0` (~606 M params, Conformer, **MIT** → edge-legal by construction)
- **Learning objective:** character CTC with a **single 67-char vocabulary shared across all 6 languages** (lets the three Nilotic languages kln/luo/mas transfer to each other)
- **Two independent stages:** (1) acoustic CTC model, (2) per-language n-gram LM — tunable separately, which is what let me iterate fast
- **Training strategy — "tranche" training:** the full audio (~750 GB) far exceeds Colab/local disk, so I stream it in ~60–80 k-clip slices, one epoch each, chaining checkpoints. Add temperature sampling (T = 1.6) to balance languages, soft SpecAugment, and a > 20 s length filter
- **Targeted refinement:** a progressive chain v2 → v5 → v6 → **v7-soup** (weight averaging — a free gain here) → v8/v9 (retraining biased toward the weakest languages) → **v9-2**
- **Decoding:** per-language **5-gram KenLM** via pyctcdecode beam search; corpus = training transcriptions ×3 + external text (Wikipedia sw/so/ki, MasakhaNEWS luo), filtered to the exact 67-char alphabet; recipe `lmplz -o 5 --discount_fallback --prune 0 0 1`; per-language `(α, β)` tuned on dev
- **Hyper-parameters:** LR 5×10⁻⁵ (base) → 1×10⁻⁵ (fine-tuning), effective batch 16 (4 × grad-accum 4), bf16, 1 epoch per tranche
- **Framework / hardware:** HuggingFace ecosystem, Google Colab, **single NVIDIA A100 40 GB** (bf16), **~30 h+** of training cumulative across the chain

## 📈 What improved the score [public leaderboard, macro-WER]

| System | Leaderboard |
|---|---|
| v2, greedy CTC (no LM) | 0.566 |
| + KenLM v1 (transcriptions only) | 0.469 |
| + KenLM v2 (external text) + tranche training (v5-t3) | 0.42 |
| v9-2 + KenLM v2, (α, β) = (0.7, 0.5) | 0.40283 |
| **v9-2 + KenLM v2, (α, β) = (0.5, 0.0) — FINAL** | **0.39477** |

Decomposition of the final score: acoustic-only ≈ 0.45; the **language model is the single biggest lever after the acoustic model** (≈ −0.055); post-plateau decoding corrections ≈ −0.008.

That last −0.008 came from a measurement campaign with **zero retraining**: an anti-padding truncation fix, direct full-length decoding, and — the main one — **lowering the LM weight** to (0.5, 0.0). The test is up to 42 % spontaneous speech (Swahili is 100 %), and my KenLM is fed read/written text; on spontaneous audio its prior guides worse, so handing the decision back to the acoustic model helps.

## ⚡ The edge part (this year's emphasis) — measured, not claimed

CPU-only, offline, one language model resident at a time. Numbers are from dated per-submission hardware-validation reports in the repo (`validation/`).

| Requirement | Limit | Measured |
|---|---|---|
| Parameters | < 1 B | 0.606 B |
| Peak RAM (fp32) | ≤ 8 GB | floor 1.46 GB · **6.71 GB** direct decode ≤ 60 s · **1.80 GB** windowed fallback beyond |
| Latency (RTF) @ 4 cores (Raspberry-Pi-class) | ≤ 2.0 | 0.75 aggregate / 1.48 max (direct) · 0.69 (windowed) |
| Offline, CPU-only | required | yes |
| License | permissive | Apache-2.0 |

The conformer's attention transient grows as T², so direct decoding of long clips peaks at 6.71 GB near 60 s. The generator auto-switches to a **windowed fallback** (`SEUIL_DIRECT_S`) beyond that, which keeps RAM ≤ 1.80 GB **at any clip length**. int8 dynamic quantization (~4× smaller) is available but not needed for compliance.

## 📉 What did NOT lead to improvement

I spent real effort mapping the ceiling. An error audit showed **CER ≈ 0.094** (the model *hears* well) but ~32 % of the residual word errors on kln/mas are hard acoustic substitutions — an acoustic floor a decoder can't fix. Confirmed repeatedly:

- **Fixed-window slicing of long clips** → hypothesis that the model breaks past its 20 s training regime was *wrong*; windowing degraded 0.40283 → 0.42737 (it cuts mid-word). Direct full-length decoding actually **wins** — the conformer extrapolates in length.
- **Spontaneous-speech retraining (v10):** forced-alignment + silence-cutting recovered ×25 more spontaneous training data and gave a huge **dev** gain (kik spontaneous 0.476 → 0.266) — that reversed to **+0.031** on the leaderboard. Overfit to the 4–10 training speakers per language.
- **200 k-unigram decoder list** → −0.007 on paired dev, **+0.001** on the leaderboard (textbook selection bias).
- **Per-(language × type) "best-pick" α/β** → flat surfaces, optima unstable at ~80 clips/case, every individual adoption under the noise.
- **omniASR-CTC-1B** (eligible, Apache-2.0) → worse zero-shot (0.66 vs 0.49) and heavy fairseq2 fine-tuning. **MMS** → ineligible (CC-BY-NC propagates to derivatives).
- **Prediction normalization** → every variant *increased* WER.

**Lesson:** decide with the LM, never greedy decoding (repeatedly misleading); reason in **deltas** and only ship effects **corroborated several times and applied globally**. And know your instrument — my dev set (only 2–6 speakers per language × type) could rank *decoding* changes but **not** *model* changes, so those must be validated online. That discipline is exactly what turned a 0.40283 plateau into 0.39477 with no retraining.

## 🔎 If I had more time

- **VAD / silence-based segmentation** for long clips (cut *between* words, not at fixed offsets).
- **int8 dynamic quantization** to drop the direct-decode peak well below the fallback threshold.
- **External text for the kln/mas KenLMs** (permissive corpora) — the measured lexical floor: even at 200 k unigrams, 7–8 % of spontaneous kln/mas words are absent from any transcription-derived corpus.
- A **multi-speaker dev split** to make model-level changes measurable — the single biggest blocker this run.
- **Sub-word / syllable tokenization** instead of pure characters.

## 🛢️ Shared resources

- 🤗 **Model + KenLM + validation + cards:** https://huggingface.co/Okpeyemi/afrivoices-asr-baobab
- 💻 **Code (training / decoding / submission / edge-validation):** https://github.com/Okpeyemi/afrivoices-asr-model
- 📄 Full technical report (`RAPPORT_afrivoices.md`), reproduction guide, model card and dated per-submission hardware-validation reports are in both repos.

Thanks again to the organizers and everyone who took part. Happy to answer any questions in the comments! 🙌
