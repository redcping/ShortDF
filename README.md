# ShortDF: Optimizing for the Shortest Path in Denoising Diffusion Model (CVPR 2025, Highlight)  
![license](https://img.shields.io/badge/License-MIT-brightgreen)  ![python](https://img.shields.io/badge/Python-3.9-blue)  ![pytorch](https://img.shields.io/badge/PyTorch-2.1-orange)  


## 🫖 Description
This repository contains the official code for our paper "Optimizing for the Shortest Path in Denoising Diffusion Model" (CVPR 2025).  
ShortDF can be seen as giving AI an “intelligent navigation system” for generation: instead of following all diffusion steps, it dynamically finds the optimal path, allowing **one step to achieve the effect of multiple steps**.

- **Implicit Graph Modeling**: Model parameters form a “path graph,” where each node represents a timestep in the diffusion process.  
- **Shortest-Path Relaxation optimization**: During training, the model compares direct paths with multi-step paths. If the direct path has higher error, it is optimized using the multi-step path, letting one step absorb the benefits of multiple steps.
- **Intuitive Example**:
  - Compare path 10→0 vs 10→2→0: if the direct path has larger error, it is optimized using the two-step path, making one step as effective as two.  
  - Compare path 100→0 vs 100→10→0: as 10→0 has already been optimized, 100→10→0 carries the improved information, which can further optimize 100→0.  
  - Repeating this process, long paths gradually absorb intermediate optimizations, achieving “one-step convergence” comparable to many original steps.

- **Main Results in CIFAR-10**:
  Original DDIM requires 10 steps to generate an image, whereas ShortDF achieves similar quality in just 2 steps — a **5× speedup**. The image fidelity measured by **FID** improves by **18.5%**. Comparison example:
 

  <div align="center">
    <img width="800" height="160" alt="image" src="https://github.com/user-attachments/assets/a0dfa05a-bed9-4ec8-95e2-bcb992d71eee" />
  </div>
For more details and experimental results, see our CVPR 2025 [paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Optimizing_for_the_Shortest_Path_in_Denoising_Diffusion_Model_CVPR_2025_paper.pdf).


---

## ⚙️ Requirements
- Python ≥ 3.9
- PyTorch ≥ 1.6
- torchvision, numpy, tqdm (standard PyTorch dependencies)

---

## 🚀 Running the Experiments

### Training
Training is identical to DDPM, e.g.:
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --ni
```

#### Loss Design Reference
- ShortDF-specific loss is implemented in `./functions/losses.py` as `shortdf_relax_loss`.
- Recommended training strategies:
  1. **Two-stage training (recommended)**:
     - First, train using the standard noise loss (or load a pretrained DDPM model) to stabilize training.
     - Then, fine-tune with `shortdf_relax_loss` to optimize shortest-path residuals.
     - This approach reduces training complexity and improves convergence stability.
  2. **One-stage training (optional)**:
     - Train both the standard noise loss and `shortdf_relax_loss` together from scratch.
     - Adjust the relative contributions using the config file to balance training:
     - You can modify `noise_weight` and `relax_weight` to suit your dataset, model size, or desired training behavior.



### Sampling

#### 1. General sampling for FID evaluation
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --fid --timesteps {STEPS} --eta {ETA} --ni
```
- ETA controls variance scale (0: DDIM, 1: DDPM).
- STEPS specifies the number of diffusion steps.
- MODEL_NAME identifies the pretrained checkpoint path.


#### 2. Sampling the sequence of images leading to a sample
- Use the `--sequence` option.

Note: Some hard-coded lines are included for specific image generation cases; you may need to modify them for your own applications.

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

We are also extending ShortDF to **text-to-image generation models**, exploring shortest-path optimization in generative multi-modal tasks. Stay tuned for the corresponding work!
> **Note**: Currently, this is one feasible way to train ShortDF. We encourage the community to explore more efficient and faster training strategies to further reduce the number of diffusion steps while maintaining high-quality samples. We hope this idea inspires additional research and practical applications in diffusion-based generation.


