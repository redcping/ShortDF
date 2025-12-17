
# (CVPR 2025, Highlight) ShortDF: Optimizing for the Shortest Path in Denoising Diffusion Model

![license](https://img.shields.io/badge/License-MIT-brightgreen)  ![python](https://img.shields.io/badge/Python-3.9-blue)  ![pytorch](https://img.shields.io/badge/PyTorch-2.1-orange)

## 🧭 Description

This repository is the official implementation of **ShortDF** (CVPR 2025). 

ShortDF acts as an **"intelligent navigation system"** for diffusion models. Instead of blindly following a fixed trajectory, it solves for the optimal path via **Implicit Graph Modeling** and **Shortest-Path Relaxation**. This allows **a single step to achieve the efficacy of multiple steps**.

### Core Mechanism
- **Path Optimization**: We treat diffusion steps as nodes in a graph. If a multi-step path (e.g., $10 \to 2 \to 0$) yields better quality than a direct step ($10 \to 0$), the model optimizes the direct step to match that higher quality.
- **Error Propagation**: Through iterative training, long paths (e.g., $100 \to 0$) absorb the refined information from intermediate steps, achieving **fewer-step convergence** comparable to the original multi-step process.

### Highlights
- **5× Speedup**: Achieves quality comparable to 10-step DDIM on CIFAR-10 in just **2 steps**.
- **Higher Fidelity**: Improves FID by **18.5%** on CIFAR-10.
- **Robustness**: Demonstrates superior performance on CelebA and LSUN-Church datasets across various sampling steps.

<div align="center">
  <img width="100%" alt="Visual comparison of generated images on CIFAR-10, showing high quality achieved with only 2 steps by ShortDF compared to 10-step DDIM." src="https://github.com/user-attachments/assets/a0dfa05a-bed9-4ec8-95e2-bcb992d71eee" />
  <p style="font-size: 0.9em; margin-top: 5px;"><strong>Figure 1: Extreme Speed Test on CIFAR-10.</strong> ShortDF achieves comparable quality at 2 steps, demonstrating a 5× speedup.</p>
  
  <img width="100%" alt="ShortDF Architecture and comparison of sampling quality across CelebA and LSUN-Church, illustrating the clear advantage of ShortDF at different step counts (step i indicates the i-th image in the sampling sequence)." src="https://github.com/user-attachments/assets/4e3b77c4-8cea-44f0-a9c2-8c6c64426030" />
  <p style="font-size: 0.9em; margin-top: 5px;"><strong>Figure 2: Multi-Dataset Performance and Sampling Trajectory (CelebA, Church).</strong> Note ShortDF's clear quality advantage across the sampling sequence (Step i).</p>
 
</div>

For more details, please refer to our [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Optimizing_for_the_Shortest_Path_in_Denoising_Diffusion_Model_CVPR_2025_paper.pdf).

---

## 🚀 Running the Experiments

### Training

Training follows the standard DDPM protocol. 

```bash
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --ni
````

#### Loss Design & Strategy

The ShortDF-specific loss is implemented in `./functions/losses.py` as `shortdf_relax_loss`. We recommend the following training strategies:

1.  **Two-stage training (Recommended)**:

      - **Phase 1**: Train using standard noise loss (or load a pretrained DDPM checkpoint) to stabilize the model.
      - **Phase 2**: Fine-tune with `shortdf_relax_loss` to optimize for shortest-path residuals.
      - *Benefit*: Reduces training complexity and ensures stable convergence.

2.  **One-stage training (Optional)**:

      - Train with both standard noise loss and `shortdf_relax_loss` from scratch.
      - *Configuration*: Adjust `noise_weight` and `relax_weight` in the config file to balance the contributions based on your dataset and model size.


## Sampling

### 1\. Download Pretrained Models

We provide pretrained models for the **CIFAR-10**, **CelebA**, and **LSUN-Church** datasets.

  * **Download Link**: [Google Drive](https://drive.google.com/drive/folders/1YWOY0UKxjE3P1hvca0_-L_kRM17SdybR?usp=sharing)
  * **Setup**: After downloading, please place the model file in the following directory structure: `logs/{DATASET}/ckpt.pth`

### 2\. General Sampling (FID Evaluation)

To generate samples and evaluate the Fréchet Inception Distance (FID):

```bash
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --fid --timesteps {STEPS} --eta {ETA} --ni
```

  * `--eta`: Controls the variance scale ($\eta=0$ for DDIM, $\eta=1$ for DDPM).
  * `--timesteps`: Specifies the number of diffusion steps ($T$).
  * `--doc`: Identifies the folder name containing the checkpoint.

**Example (CIFAR-10):**

```bash
python main.py --config cifar10.yml --exp ./ --doc cifar10 --sample --fid --timesteps 2 --eta 0 --ni --skip_type quad
python main.py --config cifar10.yml --exp ./ --doc cifar10 --sample --fid --timesteps 10 --eta 1 --ni --skip_type quad
```

**Example (LSUN-Church):**

```bash
python main.py --config church.yml --exp ./ --doc church --sample --fid --timesteps 20 --eta 1 --ni --skip_type uniform
```

-----

> **Note:**
>
> FID scores are computed using the provided reference statistics in the `stats/` directory, and are intended for relative comparison under a unified evaluation setting.
> 
> When the number of steps increases, it poses a common risk of over-denoising, which is similar to other distillation schemes. In such cases, it is recommended to **decrease the $\eta$ parameter** to achieve better results.


---

## ⚙️ Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.6
- **Dependencies**: `torchvision`, `numpy`, `tqdm`

---

## 📖 References and Acknowledgements

```

@inproceedings{chen2025optimizing,

title={Optimizing for the Shortest Path in Denoising Diffusion Model},

author={Chen, Ping and Zhang, Xingpeng and Liu, Zhaoxiang and Hu, Huan and Liu, Xiang and Wang, Kai and Wang, Min and Qian, Yanlin and Lian, Shiguo},

booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},

pages={18021--18030},

year={2025}
}

```

This implementation is based on / inspired by:
- [DDIM PyTorch repo](https://github.com/ermongroup/ddim) (code structure).
- [PyTorch-DDPM repo](https://github.com/w86763777/pytorch-ddpm) (accelerated FID evaluation).



---

## 🔮 Future Directions

We are extending ShortDF to **text-to-image** and multi-modal tasks. We encourage the community to explore more efficient training strategies based on this shortest-path paradigm.
