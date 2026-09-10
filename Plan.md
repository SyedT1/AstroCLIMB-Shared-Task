# AstroCLIMB Experimental Plan

## Main conclusion from the data audit

The metadata audit changes the strategy: metadata retrieval is an excellent source of supervision, but it is not a viable test-time feature.

- Kaggle training coverage: **99.7%** (9,970/10,000 pairs)
- Accuracy on resolved training pairs: **100%**
- Kaggle test pair coverage: **0%**
- Only four individual test objects received tentative matches; no test pair had enough metadata to infer a relationship
- Pixel hashing is unlikely to change this conclusion and would require another expensive image scan

Consequently, further experiments should not focus on metadata matching or metadata overrides. Metadata should instead be used to generate correctly labelled training pairs. The submitted classifier must ultimately operate only on figures and captions.

## Immediate next step

Create and run:

```text
notebooks/02_hf_pair_generation_and_validation.ipynb
```

This notebook should extend `generate_hf_pairs.py`. The current generator produces only image-caption pairs, which is insufficient because the competition has ten valid class-modality cells.

| Modality | `same_figure` | `same_paper` | `related_papers` | `unrelated_papers` |
| --- | ---: | ---: | ---: | ---: |
| text-text | — | 1,000 | 1,000 | 1,000 |
| text-image | 1,000 | 1,000 | 1,000 | 1,000 |
| image-image | — | 1,000 | 1,000 | 1,000 |

This structure appears exactly in both Kaggle train and test. Synthetic pair generation must reproduce it.

The new notebook should:

1. Split Hugging Face records by paper DOI before constructing pairs.
2. Generate text-text, text-image, and image-image examples in the exact ten-cell distribution above.
3. Keep training and validation DOIs completely disjoint.
4. Limit the number of pairs contributed by any one paper or citation edge.
5. Produce random, moderately hard, and very hard unrelated pairs.
6. Save compact pair manifests initially and resolve images only during embedding extraction.
7. Create at least three fixed split seeds while using one primary split for most experiments.

Recommended initial dataset sizes:

- Synthetic training: **160,000 pairs**, or 16,000 per valid class-modality cell
- Synthetic validation: **40,000 pairs**, or 4,000 per valid class-modality cell
- Kaggle training: retain all 10,000 pairs
- Use DINOv2 consistently at first; silently switching between DINOv2 and gated DINOv3 would confound comparisons

## Validation protocol

Two complementary validation evaluations are required.

### 1. Kaggle-distribution out-of-fold validation

Use the 10,000 Kaggle training pairs. Construct groups from connected object or paper components so that an object or DOI never occurs in both training and validation.

Metadata may be used to construct the groups, but it must not be supplied as a classifier feature or used to override validation predictions.

### 2. Synthetic unseen-paper validation

Split Hugging Face records into training and validation paper sets before generating pairs. The validation papers must have no DOI overlap with the training papers. Generate relationship pairs independently inside each partition.

### Required reporting

Every experiment should report:

- Overall macro-F1
- Per-class precision, recall, and F1
- Macro-F1 for each modality
- F1 for each of the ten valid class-modality cells
- Confusion matrix
- Prediction-class distribution
- Training time, inference time, and peak GPU memory where practical
- Mean and standard deviation across three seeds for finalist configurations

Models should be selected using their average rank across Kaggle grouped OOF and synthetic unseen-paper validation. A gain on synthetic validation accompanied by a Kaggle OOF decline may indicate synthetic-distribution overfitting.

Use paired bootstrap confidence intervals on identical validation predictions when comparing finalist systems. Small single-split improvements should not be treated as reliable.

## Existing notebook assessment

The **0.48279** result from `notebooks/metadata-first-hybrid-training.ipynb` is the correct performance anchor. Because test metadata coverage is zero, its Kaggle submission effectively represents the learned CatBoost fallback rather than a metadata hybrid.

The following notebooks should not be run unchanged:

- `notebooks/metadata-first-unified-multimodal.ipynb` supplies only 22 aggregate similarity features to a relatively large MLP. It discards the individual embedding dimensions before training, so the neural network receives little more information than CatBoost while being easier to overfit on 10,000 pairs.
- `notebooks/unrun/metadata-first-unified-lora.ipynb` fine-tunes encoders on all high-confidence Kaggle training labels before constructing OOF predictions. This exposes validation labels to the encoders and invalidates the resulting OOF estimate. LoRA must be trained independently inside every fold or only on the DOI-disjoint synthetic training partition.

The most promising representation improvement is to retain dimension-wise embedding interactions rather than reducing each encoder to cosine, L1, L2, and maximum-distance scalars:

$$
\mathbf{x}_{\mathrm{pair}} =
\left[
|\mathbf{e}_1-\mathbf{e}_2|;
\mathbf{e}_1\odot\mathbf{e}_2;
\cos(\mathbf{e}_1,\mathbf{e}_2)
\right].
$$

PCA or learned projection layers can control dimensionality. Any PCA transformation must be fitted only on the training portion of a fold.

## Common experimental rules

- Change one primary factor per experiment.
- Reuse identical folds, generated manifests, embeddings, and seeds within each comparison block.
- Cache embeddings by object hash so model experiments do not repeatedly encode the same objects.
- Keep metadata out of learned input features and prediction overrides.
- Maintain balance across the ten valid class-modality cells, not merely across four labels.
- Cap samples per DOI and citation edge to prevent a few prolific papers from dominating training.
- Record raw OOF probabilities for every run so calibration and ensembling can be performed later without retraining.
- Tune thresholds and ensemble weights only on OOF predictions, never on public leaderboard results.
- Use macro-F1 as the primary metric; accuracy is only diagnostic.

## Thirty-experiment matrix

The best configuration from each block becomes the anchor for the next block. Within a block, all non-target settings remain fixed.

### Controls and data experiments

| ID | Experiment | Purpose |
| ---: | --- | --- |
| E01 | Modality-aware random/prior baseline | Validate label constraints, metric calculation, and the complete pipeline |
| E02 | Reproduce CLIP plus histogram gradient boosting (`0.44505`) | Establish a reproducibility control |
| E03 | Global CatBoost with the current 22 features, Kaggle only, no metadata override | Establish a global-model baseline |
| E04 | Three CatBoost modality specialists, Kaggle only, no metadata override | Reproduce the `0.48279` anchor under the new validation protocol |
| E05 | E04 plus 40,000 synthetic pairs | Measure the value of small-scale metadata-generated augmentation |
| E06 | E04 plus 160,000 synthetic pairs | Main augmentation candidate |
| E07 | E04 plus 400,000 synthetic pairs | Test whether performance continues to scale or saturates |
| E08 | E06 with Kaggle examples weighted four times | Counter synthetic-to-Kaggle domain shift |
| E09 | E06 with random unrelated examples only | Easy-negative control |
| E10 | E06 with 50% hard unrelated examples | Test moderate hard-negative sampling |
| E11 | E06 with 75% hard unrelated examples | Test the current generator's hard-negative policy |
| E12 | E06 with unrelated examples balanced across semantic-similarity deciles and capped per DOI/edge | Test calibrated difficulty rather than a single hard-negative ratio |

For E05-E12, the generated data must remain balanced across all ten valid class-modality cells.

### Feature ablations and additions

Use the best data configuration from E05-E12 and identical CatBoost settings.

| ID | Experiment | Question answered |
| ---: | --- | --- |
| E13 | Handcrafted features only: TF-IDF, OCR overlap, pHash, modality, and lengths | How much can inexpensive features explain? |
| E14 | Encoder scalar similarities only | How much signal comes from frozen pretrained encoders? |
| E15 | Full feature set minus SPECTER2 | Does astronomy-oriented text representation help? |
| E16 | Full feature set minus SigLIP2 | Is shared cross-modal alignment essential? |
| E17 | Full feature set minus DINO | Does visual structure help beyond SigLIP2? |
| E18 | Full feature set minus word and character TF-IDF | Measure lexical-similarity value |
| E19 | Full feature set minus OCR | Measure OCR value relative to its computational cost |
| E20 | Full feature set minus pHash | Measure near-duplicate visual matching value |
| E21 | Add PCA-reduced, dimension-wise absolute-difference and Hadamard-product vectors | Test how much information was lost through scalar aggregation |
| E22 | Encode image OCR with the text encoder and compare it directly with caption embeddings | Add a stronger image-text textual alignment signal |

For E21, begin with 64 or 128 PCA dimensions per eligible encoder block. SigLIP2 can provide aligned interactions for all modalities. SPECTER2 interactions apply to text-text and OCR-derived text; DINO interactions apply to image-image.

### Architecture experiments

Use the best data and feature configuration from the earlier blocks.

| ID | Experiment | Design |
| ---: | --- | --- |
| E23 | One global CatBoost classifier | Shared model with modality indicators |
| E24 | Three CatBoost modality specialists | Independent text-text, text-image, and image-image classifiers |
| E25 | Hierarchical classifier | First classify same-figure versus not-same-figure, then classify same-paper, related, or unrelated |
| E26 | Gated shared-trunk MLP | Shared representation with modality-specific output heads |
| E27 | Supervised contrastive pair network | Contrastive auxiliary objective combined with the four-class prediction head |

E25 is particularly important because `same_figure` is legal only for mixed pairs and is qualitatively different from the three paper-level relationships.

### Encoder adaptation and final inference

| ID | Experiment | Design |
| ---: | --- | --- |
| E28 | Fold-safe SPECTER2 LoRA | Fine-tune only on synthetic training or independently inside each validation fold |
| E29 | Fold-safe SigLIP2 LoRA | Adapt both towers with pair classification and/or supervised contrastive loss |
| E30 | OOF-calibrated ensemble with modality-aware quota evaluation | Blend the strongest diverse systems and compare unconstrained versus constrained decisions |

For E28 and E29, compare the adapted encoder with its frozen version using the same pair representation and downstream classifier. This isolates the effect of fine-tuning.

## Post-processing and class constraints

At minimum, enforce the structural constraint:

```text
same_figure is impossible for text-text and image-image pairs
```

The observed construction also suggests the following test quotas:

- Text-image: 1,000 examples of each of the four classes
- Text-text: 1,000 examples each of `same_paper`, `related_papers`, and `unrelated_papers`
- Image-image: 1,000 examples each of `same_paper`, `related_papers`, and `unrelated_papers`

This should be tested rather than assumed. For E30, compare:

1. Raw row-wise argmax
2. OOF-learned class biases without exact quotas
3. Exact modality-specific quota assignment

Exact quota assignment can be implemented as a maximum-total-log-probability constrained assignment separately within each modality. It should be used for the final submission only if it improves every grouped OOF split consistently.

## Recommended execution order

Run E01-E06 first. If Hugging Face augmentation fails to improve unseen-paper validation, stop and audit the generated pairs before proceeding.

The next highest-priority experiments are:

1. E08: Kaggle upweighting
2. E10-E12: negative-sampling policy
3. E21: dimension-wise embedding interactions
4. E22: OCR encoded as text
5. E24: modality specialists
6. E25: hierarchical classification
7. E30: calibrated ensemble and quota evaluation

LoRA should be attempted only after the data, representation, and validation pipelines are stable. The most plausible sources of improvement, in order, are:

1. Correct multimodality-balanced Hugging Face augmentation
2. Better hard-negative sampling
3. Dimension-wise embedding interactions
4. Hierarchical or modality-specialist classification
5. OOF ensembling and prior calibration

## Kaggle submission policy

If the limit of 30 refers to Kaggle submissions rather than local experiments, do not submit every configuration. Use local validation to select submissions. Suggested leaderboard checkpoints are:

1. Reproduced E04 anchor
2. Best data configuration from E05-E12
3. Best feature configuration from E13-E22
4. Best architecture or adapted encoder from E23-E29
5. Final E30 ensemble

The public leaderboard should be treated as a confirmation signal, not as the hyperparameter-selection objective.
