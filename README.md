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

The reported values are the scores rendered by Kaggle for the submissions produced by the corresponding notebooks. The competition evaluates submissions with macro-averaged F1, as described above. The improved notebook raises the starter score from 0.39283 to 0.44505, and the metadata-first hybrid raises it to 0.48279.

### Starter notebook working procedure

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

### Metadata-first hybrid notebook working procedure

The [`metadata-first-hybrid-training.ipynb`](notebooks/metadata-first-hybrid-training.ipynb) notebook uses a two-stage decision system. It first attempts to recover the source figure and paper of each object from the full Hugging Face metadata. A deterministic DOI/citation-graph rule is used whenever that evidence gives one unambiguous class. Only unresolved pairs are sent to a learned, modality-specific fallback model. The metadata rule therefore has priority over the statistical prediction:

$$
\widehat y_i =
\begin{cases}
r_i, & r_i \in \mathcal{Y},\\
\displaystyle\arg\max_{k\in\mathcal{Y}} p_{m_i,k}(\mathbf{x}_i),
& r_i=\texttt{unmatched},
\end{cases}
$$

where \(\mathcal{Y}=\{\texttt{same\_figure},\texttt{same\_paper},\texttt{related\_papers},\texttt{unrelated\_papers}\}\), \(r_i\) is the metadata-derived relationship, and \(p_{m_i,k}\) is the fallback probability from the specialist for modality \(m_i\).

#### 1. Prepare and locate the inputs

Run the notebook on a Kaggle GPU and attach the full `train.csv`, `test.csv`, and `sample_submission.csv`. The included `_1000.csv` slices contain only `same_figure` examples and are suitable for smoke testing metadata matching, but not for training the four-class fallback. Run the embedded `%%writefile metadata_matching.py` cell before input discovery; it creates the matcher used by the remaining cells. When Internet access is disabled, also attach local copies of the required model folders.

The setup cell defines the random seed, chunk and model batch sizes, number of folds, OCR switch, model identifiers, and output directory. DINOv3 is gated: accept its Hugging Face license and provide `HF_TOKEN` to use it. If it cannot be loaded, the code automatically falls back to DINOv2. Inputs are searched under `/kaggle/input`, `data`, the current directory, and `/kaggle/working`.

#### 2. Build or reuse the exact metadata index

Let \(n_c(o)\) denote NFKC Unicode normalization followed by whitespace collapse, trimming, and case folding. The matcher indexes a caption by

$$
h_c(o)=\mathrm{SHA256}\!\left(n_c(o)\right).
$$

Images are matched by a SHA-256 hash of the file bytes obtained after Base64 decoding; optional decoded-pixel hashes can recover visually identical PNGs with different file encodings. The metadata index stores compact identifiers, figure IDs, normalized paper DOIs, reference DOIs, and citing DOIs rather than the large objects themselves. It is saved as `astroclimb_metadata_index.pkl` and reused for train, test, and later notebook runs.

For uniquely matched metadata records \(a\) and \(b\), the rule is evaluated in priority order:

$$
\rho(a,b)=
\begin{cases}
\texttt{same\_figure}, & \mathrm{row}(a)=\mathrm{row}(b),\\
\texttt{same\_paper}, & d_a=d_b\ne\varnothing,\\
\texttt{related\_papers}, & d_a\leftrightarrow d_b,\\
\texttt{unrelated\_papers}, & \text{otherwise},
\end{cases}
$$

where \(d_a\leftrightarrow d_b\) means that either DOI occurs in the other paper's reference or citation set. If an object is not found or does not have a unique metadata row, the initial result is `unmatched`. During labeled training matching, a known `same_figure` caption can also bootstrap the exact image-to-record hash mapping; the cached mapping is then available to the test pass.

#### 3. Recover safe ambiguous matches by candidate consensus

An exact caption or image may correspond to several metadata rows. Let \(C_1\) and \(C_2\) be the candidate record sets for the two objects. The notebook evaluates every cross-product pair and removes `unmatched` results:

$$
R_i=\lbrace\rho(a,b):a\in C_1,\ b\in C_2\rbrace
\setminus\lbrace\texttt{unmatched}\rbrace.
$$

The pair is recovered only when all viable candidate combinations agree, that is, when \(|R_i|=1\). Otherwise it remains unresolved and is reserved for the learned fallback. The audit column `resolution` distinguishes `unique`, `candidate_consensus`, and `unresolved` rows.

#### 4. Cache modality-specific representations

Each raw object is assigned the stable key \(h(o)=\mathrm{SHA256}(o)\). Normalized neural vectors and OCR text are persisted in `representations.sqlite`, so repeated objects and interrupted runs do not require another forward pass. For any encoder \(g\), the stored vector is

$$
\mathbf{e}(o)=
\frac{g(o)}{\max\!\left(\lVert g(o)\rVert_2,10^{-8}\right)}.
$$

The encoders have complementary roles:

- SPECTER2 embeds captions in a scientific-text space.
- SigLIP2 embeds both captions and figures in a shared text-image space, making it the main signal for mixed pairs.
- DINOv3, or DINOv2 as fallback, embeds figures in a visual space.
- EasyOCR extracts figure text for figure-caption and figure-figure comparison.

#### 5. Construct the 22-dimensional pair feature vector

For two normalized embeddings \(\mathbf{u}\) and \(\mathbf{v}\), each neural encoder contributes four summary statistics:

$$
q(\mathbf{u},\mathbf{v})=
\left[
\mathbf{u}^{\top}\mathbf{v},\
\frac{1}{d}\sum_{j=1}^{d}|u_j-v_j|,\
\lVert\mathbf{u}-\mathbf{v}\rVert_2,\
\lVert\mathbf{u}-\mathbf{v}\rVert_\infty
\right].
$$

Caption-caption rows also receive word and character TF-IDF cosine similarities. With L2-normalized sparse vectors, each is simply

$$
s_{\mathrm{TFIDF}}(o_1,o_2)=\mathbf{v}_1^{\top}\mathbf{v}_2.
$$

Figure-figure rows receive 64-bit perceptual-hash similarity

$$
s_{\mathrm{pHash}}(o_1,o_2)
=1-\frac{d_H\!\left(p(o_1),p(o_2)\right)}{64}.
$$

OCR comparison contributes token Jaccard overlap

$$
J(A,B)=\frac{|A\cap B|}{\max(|A\cup B|,1)}
$$

and a character-sequence similarity. Three one-hot modality flags and two clipped raw-object lengths complete the feature vector:

$$
\mathbf{x}_i=\left[
q_{\mathrm{SPECTER}},
q_{\mathrm{SigLIP}},
q_{\mathrm{DINO}},
s_{\mathrm{word}},
s_{\mathrm{char}},
s_{\mathrm{pHash}},
J_{\mathrm{OCR}},
s_{\mathrm{OCR,char}},
\mathbf{m},
\ell_1,\ell_2
\right]\in\mathbb{R}^{22},
$$

where \(\mathbf{m}\) is one-hot over caption-caption, caption-figure, and figure-figure, and \(\ell_j=\min(|o_j|,5000)/5000\). Features that do not apply to a modality are set to zero. Train and test matrices are cached as compressed `train_features.npz` and `test_features.npz` files.

#### 6. Train three balanced fallback specialists

The notebook trains a separate CatBoost classifier for caption-caption, caption-figure, and figure-figure pairs. This prevents the model from forcing very different similarity regimes into one decision boundary. For class \(k\), balanced training uses

$$
w_k=\frac{N}{K N_k},
$$

where \(N\) is the number of rows available to that specialist, \(K\) is the number of classes present, and \(N_k\) is the class count. CatBoost produces class scores \(F_{m,k}(\mathbf{x})\), converted to probabilities by

$$
p_{m,k}(\mathbf{x})=
\frac{\exp(F_{m,k}(\mathbf{x}))}
{\sum_{j\in\mathcal{Y}}\exp(F_{m,j}(\mathbf{x}))}.
$$

Validation uses `StratifiedGroupKFold`. The group key is the ordered DOI pair `obj_1_doi|obj_2_doi`; unmatched rows receive unique row groups. Thus the same known paper pair cannot be split between a fold's training and validation partitions. Out-of-fold predictions evaluate both the fallback by itself and the complete metadata-first override.

The selection metric is macro-F1:

$$
F_1^{\mathrm{macro}}=\frac{1}{4}\sum_{k=1}^{4}
\frac{2P_kR_k}{P_k+R_k}.
$$

After validation, one 800-iteration specialist per modality is refitted on all corresponding training rows and saved to `specialist_models.joblib`.

#### 7. Run hybrid inference and verify the submission

For every test row, the relevant specialist supplies a fallback distribution. If exact matching or candidate consensus produced a valid metadata relationship, that rule replaces the fallback label; otherwise the highest-probability fallback class is used. Predictions are converted to one-hot columns, merged onto `sample_submission.csv` by `id`, and checked for missing values and exactly one active class per row.

The main outputs under `/kaggle/working/astroclimb_hybrid` (or local `results/astroclimb_hybrid`) are:

| Artifact | Purpose |
| --- | --- |
| `submission_hybrid.csv` | Kaggle-ready one-hot predictions |
| `prediction_audit.csv` | Metadata rule, resolution type, fallback label, and final label per test row |
| `train_metadata_graph.csv`, `test_metadata_graph.csv` | Exact and candidate-consensus metadata results |
| `representations.sqlite` | Reusable neural-vector and OCR cache |
| `tfidf.joblib` | Fitted word and character TF-IDF models |
| `train_features.npz`, `test_features.npz` | Cached 22-column feature matrices |
| `specialist_models.joblib` | Final modality-specific CatBoost models |

Run the notebook from top to bottom on the first execution. Later runs reuse existing artifacts. If the source CSVs, feature definition, encoder, OCR setting, or model configuration changes, remove only the corresponding stale cache files before rerunning; the notebook does not fingerprint configuration changes automatically. The resulting `submission_hybrid.csv` received a Kaggle score of **0.48279**.

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
