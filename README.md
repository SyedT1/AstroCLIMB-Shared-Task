# AstroCLIMB Shared Task

AstroCLIMB is a multimodal classification shared task hosted at WASP 2026 and scored through Kaggle. It is organized by the NASA Science Explorer in partnership with [AstroExplorer](https://astroexplorer.org/).

The benchmark evaluates whether a model can partially reconstruct the citation graph of astronomy papers using only scientific figures and their captions.

## Task

Given a pair of objects, predict exactly one relationship between them. An object can be either an astronomy figure or an English-language figure caption, so a pair may be:

- figure–figure
- figure–caption
- caption–caption

The four target classes are:

| Class | Meaning |
| --- | --- |
| `same_figure` | The objects are the figure and caption belonging to the same scientific figure. This class applies only to figure–caption pairs. |
| `same_paper` | The objects originated from the same paper and therefore share a DOI. |
| `related_papers` | The objects came from different papers, and one paper cites the other. |
| `unrelated_papers` | None of the preceding relationships applies. |

Each pair has one label, and all relationships are treated as symmetric.

## Data

The dataset is available in two forms.

### Kaggle dataset

The Kaggle version contains approximately 10,000 pairs in both the training and test sets.

Training columns:

```text
id,same_figure,same_paper,related_papers,unrelated_papers,obj_1,obj_2
```

Test columns:

```text
id,obj_1,obj_2
```

`obj_1` and `obj_2` contain either plain caption text or a PNG image encoded as a Base64 string. Encoded images must be decoded before use by an image model.

The complete Kaggle download contains four CSV files and is approximately 20 GB. In particular, `test.csv` is about 10 GB, so implementations should avoid loading all decoded images into memory at once. Chunked or streaming processing is recommended.

### Full Hugging Face dataset

The full dataset contains 94,233 figure-caption records from recent open-access astronomy papers. Its features are:

```text
image
UUID
Image ID
Paper DOI
Paper Title
Image Caption
Image Authors
References DOIs
Citing DOIs
```

Images are stored as PIL PNG objects. The DOI and citation fields describe the paper citation graph as adjacency lists. Unlike the Kaggle data, this version does not enumerate object pairs; additional training pairs can be generated from its metadata.

## Evaluation

Predictions are evaluated using **macro-averaged F1** across the four target classes. Each class consequently contributes equally to the final score, regardless of its frequency.

Submissions must contain one-hot predictions in the following format:

```csv
id,same_figure,same_paper,related_papers,unrelated_papers
0,0,1,0,0
1,1,0,0,0
2,0,1,0,0
```

Every row must assign exactly one class.

## Notebook results

The experiment notebooks are kept in the [`notebooks/`](notebooks/) directory.

| Notebook | Approach | Kaggle score |
| --- | --- | ---: |
| [`astroclimb_kaggle_starter.ipynb`](notebooks/astroclimb_kaggle_starter.ipynb) | CLIP pair features + balanced logistic regression | **0.39283** |
| [`astroclimb_kaggle_improved.ipynb`](notebooks/astroclimb_kaggle_improved.ipynb) | Cached CLIP, pHash, TF-IDF, grouped OOF validation, threshold tuning, and nonlinear model comparison | **0.44505** |
| [`metadata-first-hybrid-training.ipynb`](notebooks/metadata-first-hybrid-training.ipynb) | Metadata-first DOI/citation-graph resolution with candidate consensus and modality-specific CatBoost fallbacks using SPECTER2, SigLIP2, DINO, TF-IDF, OCR, and pHash features | **0.48279** |

The reported values are the scores rendered by Kaggle for the submissions produced by the corresponding notebooks. The competition evaluates submissions with macro-averaged F1, as described above. The improved notebook raises the score from 0.39283 to 0.44505, an absolute gain of 0.05222.

### Notebook working procedure

For each pair, the notebook first identifies each object as either caption text or a Base64-encoded PNG. A pretrained CLIP model maps both modalities into the same embedding space. If the raw CLIP representation of object \(o_i\) is \(g(o_i)\), its normalized embedding is

$$
\mathbf{e}_i = \frac{g(o_i)}{\max(\lVert g(o_i) \rVert_2,\,10^{-8})}.
$$

Given normalized pair embeddings \(\mathbf{e}_1\) and \(\mathbf{e}_2\), cosine similarity simplifies to their dot product:

$$
s_{\mathrm{cos}} = \mathbf{e}_1^\top \mathbf{e}_2.
$$

The symmetric pair representation concatenates the absolute embedding difference, elementwise product, cosine similarity, a three-value modality indicator, two text-length comparison features, and an exact-text-match flag:

$$
\mathbf{x} = \left[
|\mathbf{e}_1-\mathbf{e}_2|\;;\;
\mathbf{e}_1 \odot \mathbf{e}_2\;;\;
s_{\mathrm{cos}}\;;\;
\mathbf{m}\;;\;
|\ell_1-\ell_2|\;;\;
\min(\ell_1,\ell_2)\;;\;
q_{\mathrm{exact}}
\right],
$$

where \(\mathbf{m}\) is one-hot over text–text, mixed, and image–image pairs, and \(\ell_i=\min(|o_i|,5000)/5000\) for text (zero for images).

A class-balanced logistic-regression classifier then calculates one score per class,

$$
p(y=k\mid\mathbf{x}) =
\frac{\exp(\mathbf{w}_k^\top\mathbf{x}+b_k)}
{\sum_{j=1}^{4}\exp(\mathbf{w}_j^\top\mathbf{x}+b_j)},
\qquad
\hat{y}=\arg\max_k p(y=k\mid\mathbf{x}).
$$

The predicted class is converted to the required one-hot submission row. Processing is performed in small CSV chunks and saved as compressed feature shards, which keeps the pipeline usable with the approximately 10 GB input files and allows interrupted extraction to resume.

### Improved notebook procedure

The [`astroclimb_kaggle_improved.ipynb`](notebooks/astroclimb_kaggle_improved.ipynb) pipeline retains the normalized CLIP representation above but avoids recomputing embeddings for repeated objects. Each object is assigned a fixed-size SHA-256 key that includes its modality,

$$
h(o)=\mathrm{SHA256}\left(\mathrm{modality}(o) \Vert o\right),
$$

and its embedding is stored persistently as \(h(o)\mapsto\mathbf{e}(o)\) in SQLite. The full caption or Base64 image is therefore never used as a dictionary key.

For image–image pairs, the notebook computes a 64-bit perceptual hash from the low-frequency coefficients of a two-dimensional discrete cosine transform. Its normalized similarity is

$$
s_{\mathrm{pHash}}(o_1,o_2)
=1-\frac{d_H\left(p(o_1),p(o_2)\right)}{64},
$$

where \(p(o)\) is the perceptual hash and \(d_H\) is Hamming distance. This feature measures visual resemblance despite small encoding or pixel-level changes.

For caption–caption pairs, separate word and character TF-IDF representations are fitted. For term \(t\) in caption \(d\), the weight is

$$
\mathrm{TFIDF}(t,d)
=\mathrm{TF}(t,d)
\left[\log\left(\frac{N+1}{\mathrm{DF}(t)+1}\right)+1\right],
$$

and the similarity of two L2-normalized TF-IDF vectors is

$$
s_{\mathrm{TFIDF}}(d_1,d_2)
=\mathbf{v}_{d_1}^{\top}\mathbf{v}_{d_2}.
$$

Both word and character similarities are included. Modality indicators ensure that TF-IDF is used only for caption–caption pairs and perceptual hashing only for image–image pairs. The improved feature vector is

$$
\mathbf{x}_{\mathrm{improved}}
=\left[
|\mathbf{e}_1-\mathbf{e}_2|\;;\;
\mathbf{e}_1\odot\mathbf{e}_2\;;\;
s_{\mathrm{cos}}\;;\;
\mathbf{m}\;;\;
|\ell_1-\ell_2|\;;\;
\min(\ell_1,\ell_2)\;;\;
q_{\mathrm{exact}}\;;\;
s_{\mathrm{pHash}}\;;\;
s_{\mathrm{word}}\;;\;
s_{\mathrm{char}}
\right].
$$

To reduce leakage, object hashes form a graph: two hashes are joined when they occur in the same training pair. Connected components define group labels,

$$
g_i=\mathrm{component}\left(h(o_{i1}),h(o_{i2})\right),
$$

and `StratifiedGroupKFold` keeps every component wholly inside either the training or validation side of a fold. For fold \(f\), out-of-fold probabilities are produced only by a model that did not train on that fold:

$$
\widehat{\mathbf{p}}_i^{\mathrm{OOF}}
=M^{(-f)}(\mathbf{x}_i),\qquad i\in f.
$$

The notebook compares balanced multinomial logistic regression with histogram gradient boosting. The nonlinear model represents each class score as an additive ensemble of decision trees,

$$
F_k(\mathbf{x})=F_{k,0}+\eta\sum_{r=1}^{R}f_{k,r}(\mathbf{x}),
$$

allowing interactions such as high perceptual similarity being important only for image–image pairs. The candidate with the stronger grouped OOF macro-F1 is refitted on all training rows.

Finally, class-specific thresholds are tuned on OOF probabilities. Prediction uses scaled class competition,

$$
\hat y_i
=\arg\max_{k\in\{1,\ldots,4\}}
\frac{\widehat p_{ik}}{\tau_k},
$$

where the threshold vector \(\boldsymbol{\tau}\) is selected by coordinate search to maximize macro-F1. The resulting `submission_improved.csv` received a Kaggle score of **0.44505**.

## Suggested approach

A practical solution can combine modality-specific representations and pairwise similarity features:

- detect whether each object is text or a Base64-encoded image;
- use text embeddings or TF-IDF features for captions;
- use visual embeddings and perceptual hashes for figures;
- use a shared image-text model for figure–caption pairs;
- add exact-match, lexical-overlap, OCR, and embedding-similarity features;
- train a four-class classifier and select it using validation macro-F1;
- exploit the full dataset's DOI and citation metadata to construct additional labeled pairs.

Validation splits should prevent identical or closely related objects from appearing in both training and validation sets. A naive random row split may overestimate performance when objects are repeated across pairs.

## Competition notes

- The Kaggle competition is used only to automate scoring and offers no monetary reward.
- The benchmark is intended primarily to evaluate multimodal models rather than to require training a large model from scratch.
- Participants wishing to submit to TRACS must also register for AACL-IJCNLP 2026.
- The competition brief states that the full benchmark test set is planned for release in November 2026.

See [`details.txt`](details.txt) for the source competition description.

## Metadata matching

`metadata_matching.py` first indexes compact caption metadata, then collects the
image hashes present in the requested competition CSV. By default it streams the
Hugging Face image column in batches of 32 and retains only hashes relevant to
that CSV:

```bash
python /kaggle/working/metadata_matching.py \
  --competition-file train.csv \
  --image-batch-size 32
```

Use `--metadata-only` to skip image matching, or `--pixel-hashes` to additionally
match decoded pixels when PNG byte encodings differ. The latter is slower because
every streamed metadata image must be decoded. The compact cache is reused on
later runs, including a subsequent `test.csv` run.

## Balanced Hugging Face pair generation

`generate_hf_pairs.py` converts the Hugging Face metadata graph into balanced
image-caption training manifests without reading or copying the 72 GB image
column. It splits paper DOIs before sampling, so no paper is shared between its
synthetic train and validation sets:

```bash
python /kaggle/working/generate_hf_pairs.py \
  --metadata /kaggle/input/astroclimb/AstroCLIMB.parquet \
  --output-dir /kaggle/working/hf_pairs \
  --train-pairs-per-class 50000 \
  --validation-pairs-per-class 10000
```

The output directory contains:

- `hf_train_pairs.csv` and `hf_validation_pairs.csv`: balanced compact pair
  manifests with source row indices, UUIDs, DOIs, labels, and sampling method;
- `hf_doi_split.csv`: the reproducible paper-level split assignment;
- `hf_pair_summary.json`: class, split, and hard-negative sampling counts.

The four classes are constructed from exact row identity, shared DOI, citation
edges, and DOI pairs with no citation edge. By default, 75% of unrelated pairs
are sought from captions sharing a relatively uncommon token; remaining
unrelated pairs are sampled randomly. Use `--hard-negative-fraction` to change
that mixture. The downstream embedding job should resolve `image_row` and
`caption_row` (or their stable UUID fields) against the original dataset and
cache each source object's representation once.
