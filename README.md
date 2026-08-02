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
| [`astroclimb_kaggle_graph_resolver.ipynb`](notebooks/astroclimb_kaggle_graph_resolver.ipynb) | Hugging Face entity resolution, DOI citation-graph reconstruction, and confidence-gated fallback overrides | Not evaluated yet |

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

### Graph resolver notebook procedure

The [`astroclimb_kaggle_graph_resolver.ipynb`](notebooks/astroclimb_kaggle_graph_resolver.ipynb) treats classification primarily as entity resolution followed by deterministic citation-graph inference. It streams only the required metadata columns from `adsabs/AstroCLIMB` and stores compact fingerprints, record metadata, and matches in SQLite.

Caption text is normalized with Unicode NFKC normalization, case folding, whitespace collapsing, and trimming. Its fingerprint is

$$
f_T(t)=\mathrm{SHA256}\left(\mathrm{normalize}(t)\right).
$$

Image matching does not hash the PNG file bytes because identical pixels can have different compression. Each image is decoded into canonical RGB pixels, and its fingerprint includes its dimensions:

$$
f_I(I)=\mathrm{SHA256}\left(
w(I) \Vert h(I) \Vert \mathrm{RGB}(I)
\right).
$$

For a Kaggle object \(o\), exact fingerprint lookup returns a possibly empty candidate set of Hugging Face records,

$$
\mathcal{C}(o)=
\bigl[r \mid f_{m(o)}(r)=f_{m(o)}(o)\bigr],
$$

where \(m(o)\in(T,I)\) denotes text or image modality. The square brackets denote the collection of all matching records. Candidate collections are retained rather than forcing an arbitrary match when duplicate captions or images occur.

DOIs are canonicalized by lowercasing, trimming punctuation, and removing prefixes such as `https://doi.org/` and `doi:`. For a resolved record \(r\), let \(u(r)\) be its figure UUID, \(d(r)\) its normalized source DOI, \(R(r)\) its reference-DOI set, and \(C(r)\) its citing-DOI set. The pair label is reconstructed hierarchically:

$$
L(r_1,r_2)=
\begin{cases}
\mathrm{same\_figure},
& u(r_1)=u(r_2) \text{ and } m(o_1)\ne m(o_2),\\
\mathrm{same\_paper},
& d(r_1)=d(r_2),\\
\mathrm{related\_papers},
& d(r_2)\in R(r_1)\cup C(r_1)\\
& \quad \text{or } d(r_1)\in R(r_2)\cup C(r_2),\\
\mathrm{unrelated\_papers},
& \text{otherwise}.
\end{cases}
$$

For ambiguous matches, the notebook evaluates every candidate combination:

$$
\mathcal{L}(o_1,o_2)=
\bigl[L(r_1,r_2) \mid
r_1\in\mathcal{C}(o_1),\;
r_2\in\mathcal{C}(o_2)
\bigr].
$$

A graph prediction is accepted only when both candidate sets are nonempty and all candidate combinations agree:

$$
\widehat y_{\mathrm{graph}}=
\begin{cases}
y, & |\mathcal{L}(o_1,o_2)|=1
\text{ and }y\in\mathcal{L}(o_1,o_2),\\
\varnothing, & \text{otherwise}.
\end{cases}
$$

This unanimity rule prevents an uncertain metadata match from silently replacing a learned prediction. If \(c(o)\) is the strongest entity-resolution confidence for object \(o\), pair confidence is conservative:

$$
c_{\mathrm{pair}}=\min\left(c(o_1),c(o_2)\right).
$$

The final hybrid prediction uses the graph result only when it is unambiguous and sufficiently confident; otherwise it retains the improved model's fallback prediction:

$$
\widehat y=
\begin{cases}
\widehat y_{\mathrm{graph}},
& \widehat y_{\mathrm{graph}}\ne\varnothing
\text{ and }c_{\mathrm{pair}}\ge\tau,\\
\widehat y_{\mathrm{fallback}},
& \text{otherwise}.
\end{cases}
$$

The current notebook indexes only exact caption and pixel matches, assigns them confidence 1, and therefore effectively uses \(\tau=1\). The confidence form also supports adding calibrated approximate matches later without changing the hybrid decision rule.

Evaluation reports graph accuracy together with resolved-pair coverage. If \(S\) is the set of pairs receiving an unambiguous graph prediction, coverage is

$$
\mathrm{coverage}=\frac{|S|}{N},
$$

and resolved accuracy is

$$
\mathrm{accuracy}_{S}
=\frac{1}{|S|}\sum_{i\in S}
\mathbb{1}\left[\widehat y_i=y_i\right].
$$

Coverage must accompany the resolved-subset score: high accuracy on a small resolved subset does not imply high performance over all Kaggle pairs. The optional Hugging Face image pass is disabled by default because it may scan most of the 72.4 GB image data; the metadata pass requests Parquet column projection during loading to exclude the image column.

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
