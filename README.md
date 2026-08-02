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

## Starter notebook results

The starter notebook is kept in the [`notebooks/`](notebooks/) directory.

| Notebook | Approach | Kaggle score |
| --- | --- | ---: |
| [`astroclimb_kaggle_starter.ipynb`](notebooks/astroclimb_kaggle_starter.ipynb) | CLIP pair features + balanced logistic regression | **0.39283** |
| [`astroclimb_kaggle_improved.ipynb`](notebooks/astroclimb_kaggle_improved.ipynb) | Cached CLIP, pHash, TF-IDF, grouped OOF validation, threshold tuning, and nonlinear model comparison | Not evaluated yet |

The reported value is the score rendered by Kaggle for the submission produced by this notebook. The competition evaluates submissions with macro-averaged F1, as described above.

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
