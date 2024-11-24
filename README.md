# catvton-flux

An advanced virtual try-on solution that combines the power of [CATVTON](https://arxiv.org/abs/2407.15886) (Contrastive Appearance and Topology Virtual Try-On) with Flux fill inpainting model for realistic and accurate clothing transfer.
Also inspired by [In-Context LoRA](https://arxiv.org/abs/2410.23775) for prompt engineering.

## Showcase
| Original | Garment | Result |
|----------|---------|---------|
| ![Original](example/person/1.jpg) | ![Garment](example/garment/1.jpg) | ![Result](example/result/1.png) |
| ![Original](example/person/1.jpg) | ![Garment](example/garment/04564_00.jpg) | ![Result](example/result/2.png) |
| ![Original](example/person/00008_00.jpg) | ![Garment](example/garment/00034_00.jpg) | ![Result](example/result/3.png) |

## Model Weights
Hugging Face: 🤗 [catvton-flux-alpha](https://huggingface.co/xiaozaa/catvton-flux-alpha)

The model weights are trained on the [VITON-HD](https://github.com/shadow2496/VITON-HD) dataset.

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
@misc{chong2024catvtonconcatenationneedvirtual,
 title={CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models}, 
 author={Zheng Chong and Xiao Dong and Haoxiang Li and Shiyue Zhang and Wenqing Zhang and Xujie Zhang and Hanqing Zhao and Xiaodan Liang},
 year={2024},
 eprint={2407.15886},
 archivePrefix={arXiv},
 primaryClass={cs.CV},
 url={https://arxiv.org/abs/2407.15886}, 
}
@article{lhhuang2024iclora,
  title={In-Context LoRA for Diffusion Transformers},
  author={Huang, Lianghua and Wang, Wei and Wu, Zhi-Fan and Shi, Yupeng and Dou, Huanzhang and Liang, Chen and Feng, Yutong and Liu, Yu and Zhou, Jingren},
  journal={arXiv preprint arxiv:2410.23775},
  year={2024}
}
```

## License
- The code is licensed under the MIT License.
- The model weights have the same license as Flux.1 Fill and VITON-HD.