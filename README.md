# Size-Invariant Metrics

This is the official code for the computation of **Size-Invariant Metrics** in paper "Size-invariance Matters: Rethinking Metrics and Losses for Imbalanced Multi-object Salient Object Detection" accepted by International Conference on Machine Learning (ICML2024). The paper is available [here](https://arxiv.org/abs/2405.09782), and the complete repository is [here](https://github.com/Ferry-Li/SI-SOD).

With this repository, SI Metrics can be directly computed given the path to the prediction and  ground truth directory.

## Datasets

All that you need to prepare is a directory containing the ground mask "xxx.png", and a directory containing the prediction mask "xxx.png".

Here some notifications:

1. The range of prediction map can be either in [0, 255] or in [0, 1]. If the range is [0, 255], set the `normalize` to `True` in `config.yaml`.
2. The `epsilon` in `config.yaml` is designed to remove small noise points in the ground truth map. If there is no connected component found after the denoising, just set `epsilon` to a larger value.

## Evaluation

The evaluation configs are stored at `config.yaml`, where you can modify the settings of data, model, visualization, and metrics.

To begin evaluation, you can run the following command:

```python
python main.py
```

## Citation

If you find this work or repository useful, please cite the following:

```bib
@inproceedings{li2024sizeinvariance,
title={Size-invariance Matters: Rethinking Metrics and Losses for Imbalanced Multi-object Salient Object Detection}, 
author={Feiran Li and Qianqian Xu and Shilong Bao and Zhiyong Yang and Runmin Cong and Xiaochun Cao and Qingming Huang},booktitle={The Forty-first International Conference on Machine Learning},
year={2024}
}
```

## Contact us

If you have any detailed questions or suggestions, feel free to email us: lifeiran@iie.ac.cn! Thanks for your interest in our work!