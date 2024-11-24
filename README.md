# catvton-flux

An advanced virtual try-on solution that combines the power of [CATVTON](https://arxiv.org/abs/2407.15886) (Contrastive Appearance and Topology Virtual Try-On) with Flux fill inpainting model for realistic and accurate clothing transfer.

## Showcase
| Original | Result |
|----------|--------|
| ![Original](example/person/1.jpg) | ![Result](example/result/1.png) |
| ![Original](example/person/00008_00.jpg) | ![Result](example/result/2.png) |
| ![Original](example/person/00008_00.jpg) | ![Result](example/result/3.png) |

## Model Weights
The model weights are trained on the [VITON-HD](https://github.com/shadow2496/VITON-HD) dataset.
🤗 [catvton-flux-alpha](https://huggingface.co/xiaozaa/catvton-flux-alpha)

## Prerequisites
```bash
bash
conda create -n flux python=3.10
conda activate flux
pip install -r requirements.txt
```

## Usage

```bash
python tryon_inference.py \
--image ./example/person/00008_00.jpg \
--mask ./example/person/00008_00_mask.png \
--garment ./example/garment/00034_00.jpg \
--seed 42
```

## TODO:
- [ ] Release the FID score
- [ ] Add gradio demo
- [ ] Release updated weights with better performance

## Citation

```bibtex
@misc{jiang2024catvton,
title={CATVTON: A Contrastive Approach for Virtual Try-On Network},
author={Chao Jiang and Xujie Zhang}
}
```

## License
- The code is licensed under the MIT License.
- The model weights have the same license as Flux.1 Fill and VITON-HD.