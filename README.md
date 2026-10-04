# Sparse Scout probe planning

Fine-tunes an ImageNet-pretrained ResNet on sparse scout survey images to
predict, for each image, the ordered four-probe sweep plan (two horizontal and
two vertical line sweeps out of a 32-probe catalogue) and the parking
coordinate.

A feature-pyramid map is pooled into per-row and per-column profiles read by a
shared 1-D head; probe logits are sampled at the catalogue coordinates.
Heads cover probe selection, ordered positions, probe pairs and a 64x64
parking heat-map, with eight-fold dihedral augmentation and TTA.

    python3 solution.py <public_dir> <submission_out>

Key files: `solution.py` (end-to-end), `core.py` (training, decoding,
scoring), `train.py` / `train2.py`, `model2.py`, `fit_predict.py`,
`profiles.py`, `shrink.py`, `analyze*.py`. `logs/` holds training runs.

The dataset and out-of-fold predictions are not included.

Licensed under the MIT License.
