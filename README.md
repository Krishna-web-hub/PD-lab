# PD-Lab: Food Spoilage Detection

A Vision Transformer that classifies food photographs as **fresh** or **spoiled**,
plus the data pipeline that prepares the dataset for it.

**Status:** the spoilage classifier is built, trained and evaluated. The other two
tasks described in the roadmap below are *not built*. See [Roadmap](#roadmap).

---

## Results

ViT-B/16 (ImageNet pretrained), backbone frozen, classification head fine-tuned
for 5 epochs. Evaluated on the **held-out test split**, which is used for neither
training nor model selection.

| class | recall | precision | F1 | support |
|---|---:|---:|---:|---:|
| fresh | 1.000 | 0.990 | 0.995 | 99 |
| spoiled | 0.967 | 1.000 | 0.983 | 30 |

**Accuracy 99.22%, macro-F1 0.989.** Confusion matrix `[[99, 0], [1, 29]]`: one
spoiled item was classified as fresh.

Read accuracy with care: the split is roughly 3:1 fresh:spoiled, so a model that
answers "fresh" unconditionally scores **76.74%** while catching zero spoiled
items. Recall on `spoiled` is the number that matters, because a missed spoiled
item is the expensive error in this domain.

Validation accuracy during training reached 98.43% at epoch 5
(`[[96, 1], [1, 29]]` on 127 images).

**Is it overfitting?** No. Validation loss falls monotonically across all five
epochs (0.1530 → 0.0760 → 0.0598 → 0.0532 → 0.0511) rather than turning upward,
test accuracy exceeds validation accuracy, and only **1,538 parameters** are
trainable. The backbone is frozen, so this is a linear probe on fixed ImageNet
features and has nothing like the capacity to memorise 598 images. Training
accuracy reaching 100% means those features are linearly separable for this
task, not that the model memorised the data.

Read the headline numbers with two caveats instead. First, the sample is small:
`spoiled` recall of 29/30 carries a 95% confidence interval of
**[0.833, 0.994]**, so these 30 images cannot distinguish a 97% model from an
85% one. Second, and more seriously, see issues 1 and 2 below: the class being
predicted is not quite the one the project names, and the test split is not
independent of the training data.

---

## 1. Dataset

Images were collected into three folders:

```
Images/
   AI_gen _food/   340 images   AI-generated food
   good_food/      652 images   fresh food
   bad_food/       203 images   spoiled food
```

> **Note:** the 340 AI-generated images were collected but never made it into the
> task dataset. See [Known issues](#known-issues).

## 2. Preprocessing

`cleaning.py` opens every image, reports any that fail to decode, and (when
given `--out`) writes a **224×224** copy to that folder.

Uniform dimensions are what let the images batch into a single tensor. Note that
`Images/` on disk is *already* 224×224 and `processed_images/` is a parallel copy
of it, so the resize has effectively already been applied at source; the training
transforms resize again at load time regardless.

## 3. Organisation by task

The project was structured around more than one classification task, so the
images were reorganised by task rather than by their original folder:

```
dataset/
   ai_detection/
      real_food/       652
      ai_generated/      0   <-- never populated
   spoilage_detection/
      fresh/           652
      spoiled/         203
```

| Original folder | Task label | Populated |
|---|---|---|
| `good_food` | `spoilage_detection/fresh` | yes |
| `bad_food` | `spoilage_detection/spoiled` | yes |
| `good_food` | `ai_detection/real_food` | yes |
| `AI_gen _food` | `ai_detection/ai_generated` | **no** |

Because `ai_generated` is empty, the AI-detection task has only one class and
cannot be trained. Only **spoilage detection** is a usable dataset.

## 4. Train / validation / test split

`split_dataset.py` splits each task 70 / 15 / 15.

| split | fresh | spoiled | total |
|---|---:|---:|---:|
| train | 456 | 142 | 598 |
| val | 97 | 30 | 127 |
| test | 99 | 30\* | 129 |

\* The test directory holds 31 spoiled files, but one is `.avif` and is silently
skipped. See [Known issues](#known-issues).

## 5. Tensor conversion and loading

`ImageFolder` reads images and infers labels from the directory names;
`DataLoader` batches them. Images become `3 × 224 × 224` float tensors with
pixel values scaled to `[0, 1]`, then normalised with ImageNet statistics,
which the pretrained ViT requires, since that is the distribution its weights
were learned on.

A batch is `torch.Size([32, 3, 224, 224])`.

## 6. Verification

`visualise_batch.py` converts a batch back into viewable images and plots them
with their labels, to confirm the labels line up with the pictures and the
pipeline has not silently scrambled anything.

## 7. Training

`pD_final.ipynb` is the notebook that produced the results above. It lives
outside this repository, alongside it, and resolves its paths relative to the
repo root.

The ViT backbone is **frozen** and only the classification head is trained:
1,538 parameters out of 86.6 M. With 598 training images, fine-tuning the whole
backbone would mostly memorise them.

```
Raw images → clean & resize → organise by task → split → tensors → DataLoader → train → evaluate
```

---

## Known issues

These are recorded rather than quietly fixed, because each one is a real defect
worth understanding.

1. **The `spoiled` class is mostly contamination, not spoilage.**
   A visual sample of `dataset/spoilage_detection/spoiled/` shows that much of it
   is *foreign-object contamination*, hairs and insects in canteen food, several
   images carrying a red circle drawn on by whoever originally published them,
   rather than mould, rot or degradation. Those are different phenomena with
   different visual signatures and different handling procedures.

   This matters more than any metric on this page: the class the model is scored
   against does not match the name it is given. "Fresh vs spoiled" is not the task
   being learned. Before any further modelling, the 203 images should be sorted by
   what they actually show (contamination, genuine spoilage, or neither), which
   would yield a much smaller but coherent spoilage set, and a contamination set
   that does not currently exist as such (see the Roadmap).

   Not measured, observed. Sorting the class by hand is the way to size it.

2. **The test split is not independent of the training data.**
   85% of the files in both classes are named `aug_*`. The corpus is not ~855
   photographs but augmented derivatives of roughly 125 originals, and the
   augmentation was applied *before* the split. Held-out therefore does not mean
   what it normally means.

   Measured with the model's own 768-dimensional features: two randomly chosen
   training images have a cosine similarity of **0.172** (99th percentile 0.712),
   but the median test image's nearest training neighbour sits at **0.844**, and
   14% exceed 0.9. The typical test image has a training image that closely
   resembles it.

   So 99.22% accuracy is a fair measure of performance *on photographs very like
   the ones seen in training*, and not evidence of generalisation to new food,
   new kitchens or new cameras. Issue 8 is the same finding from the other
   direction. Fixing it needs augmentation moved after the split (issue 6) and
   grouping by source photograph, not by file (issue 7).

3. **340 AI-generated images are silently dropped.**
   `organise_dataset.py` tests `if "AI_gen_food" in root`, but the folder is
   named `AI_gen _food`, with a space before `_food`. The condition never
   matches, every AI-generated image is skipped, and the script still prints
   `Dataset organization completed.` This is why `ai_detection/ai_generated` is
   empty and the AI-detection task cannot be trained. A one-character mismatch
   with no error message.

   `organise_dataset.py` now matches on a normalised folder name (lowercased,
   non-alphanumerics stripped), reports any source folder that matched no route,
   and prints a per-destination total so a destination receiving zero images is
   visible. A dry run confirms all 340 images now route to
   `ai_detection/ai_generated`. The dataset has **not** been regenerated, because
   that would require re-splitting and would invalidate the trained model and
   every number above.

4. **One test image is never evaluated.**
   `dataset/test/spoilage_detection/spoiled/` contains 31 files, but one is
   `.avif`, which is not in `torchvision`'s `IMG_EXTENSIONS`. `ImageFolder`
   skips it without warning, so the test split is 129 images, not 130. The image
   is not corrupt; it simply never enters the loader. `split_dataset.py` now
   prints a warning for any file whose extension `ImageFolder` cannot load, so
   the next occurrence is visible rather than silent; converting the file to
   `.jpg` would recover it.

5. **The preprocessing scripts had hardcoded Windows paths.** *(fixed)*
   Every script contained a literal `D:\PD_LAB\...` path, so none of them ran
   unmodified on another machine. All six now resolve their defaults relative to
   the repo root and accept a path argument, so they run anywhere. Each was
   executed against this repo to confirm it works.

   `cleaning.py` also moved from OpenCV to Pillow. Pillow arrives with
   torchvision, so the script now runs in the same environment as the rest of the
   project instead of needing a dependency that was not installed. Its resize
   step, which the README described but the script never actually performed, is
   now implemented behind `--out` and refuses to overwrite an existing folder.

6. **The heavy-augmentation path in the notebook leaks between splits.**
   The optional `RUN_HEAVY_AUG` path generates 50 variations per training image
   and then performs a `random_split` over the *augmented* set. Near-duplicates
   of the same source photograph end up in both train and validation, which
   inflates validation accuracy without improving the model. The path is
   disabled by default (`RUN_HEAVY_AUG = False`) and the reported results above
   do not use it. Augmentation should be applied *after* splitting, not before.

7. **Duplicate images leak across the train/val/test splits.**
   The image collection contains byte-identical copies: the `name (1).jpg` /
   `name (2).jpg` pattern that downloads and file managers produce. The original
   `split_dataset.py` shuffled files individually, so those copies were scattered
   across different splits: **9 groups of exact duplicates span two splits**, and
   3 test images are byte-identical to a training image.

   Measured impact on the reported results: **negligible.** Recomputing the test
   metrics with the 3 leaked images removed gives 99.21% accuracy versus 99.22%,
   with `spoiled` recall unchanged at 29/30. The validation split is dirtier
   (6 of 127) and is mildly flattered. `split_dataset.py` now groups files by
   SHA-256 and assigns each group to exactly one split, so this cannot recur.
   The existing splits have deliberately **not** been regenerated, because doing
   so would invalidate the trained model and every number in this README.

8. **The model is confident on data it has never seen.**
   Run over the 340 AI-generated images in `Images/AI_gen _food`, a different
   image source absent from every split, the model predicts `fresh` for 327 of
   340 with a **mean confidence of 0.953**, and 86.8% of its predictions exceed
   0.90 confidence.

   The predictions are defensible; AI-generated food tends to look glossy and
   fresh. The confidence is not. The model is near-certain on a distribution it
   has never encountered, which means low confidence cannot be used as a signal
   that an input is unfamiliar. Any deployment needs an explicit abstention
   threshold rather than trust in the probability, and the probabilities
   themselves should be calibrated (e.g. temperature scaling) before they are
   read as likelihoods.

9. **Small, single-source, imbalanced dataset.**
   598 training images at roughly 3:1. Expect the model to degrade on food
   types, lighting and cameras unlike the training set, and do not expect its
   confidence to drop when it does.

---

## Roadmap

Not built. Listed as intent, not capability.

1. **AI-generated food detection** – blocked on issue 3 above; the data exists
   but has never reached the dataset.
2. **Contamination detection** (hair, insects, plastic) – much of this data has
   in fact already been collected; it is sitting inside the `spoiled` class,
   mislabelled (issue 1). Relabelling that class by what the images actually
   show would produce this dataset and clean up spoilage detection at once.
3. **A combined multi-stage pipeline** – depends on both of the above.

---

## Files

```
pre_processing/
   cleaning.py            decode-check every image; --out writes 224×224 copies
   organise_dataset.py    copy Images/ into per-task folders (--dry-run to preview)
   split_dataset.py       70/15/15 split per task, deduplicating by content hash
                          (--dry-run to preview, --seed for reproducibility)
   tensor_pipeline.py     transforms + DataLoader over one batch (--save for headless)
   visualise_batch.py     plot a batch with labels (--save for headless)
   testing.py             ImageFolder load check; reports files it could not load
```

## Tools

| Tool | Purpose |
|---|---|
| Python | Programming |
| Pillow | Image loading, decode checks, resizing |
| PyTorch + torchvision | Model, transforms, dataset loading |
| Matplotlib | Visualisation |
