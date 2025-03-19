---
permalink: /projects/knee-oa-gen.html
layout: page
title: Generating Severity-conditioned Knee Osteoarthritis X-rays Using Diffusion Neural Networks
description: A deep learning project focused on generating synthetic knee X-ray images with controlled osteoarthritis severity using diffusion models
poster: /assets/images/knee-oa-gen.png
importance: 1
category: research
github: https://github.com/khizar-anjum/KneeOAGen
paper: https://github.com/khizar-anjum/KneeOAGen
---

<style>
	.row {
	  display: flex;
	}

	.column {
	  display: grid;
	  flex: 50%;
	  padding: 5px;
	  align-content: center;
	}
</style>

## Resources

- [GitHub Repository](https://github.com/khizar-anjum/KneeOAGen)
- [Technical Report](https://github.com/khizar-anjum/KneeOAGen/blob/main/technical_report.pdf?raw=true)
- [Pre-trained Model Weights](https://drive.google.com/file/d/1I0nHw2Vi-gVkUmYXtl5AYUdf9zD9ABxK/view?usp=sharing)


## Overview

This project presents an implementation and evaluation of diffusion neural networks for generating synthetic knee X-ray images. We explore both unconditional and conditional variants of Denoising Diffusion Probabilistic Models (DDPMs), with a focus on generating images across different osteoarthritis severity levels. Our conditional DDPM architecture enables controlled image generation based on Kellgren-Lawrence (KL) grades by incorporating severity information into the diffusion process.

## Key Features

- Implementation of both unconditional and conditional DDPM models for knee X-ray generation
- Controlled generation based on Kellgren-Lawrence (KL) grades (0-4)
- High-fidelity synthetic images capturing grade-specific characteristics
- Quantitative evaluation using FID and Inception Score metrics
- Comprehensive model architecture with U-Net backbone
- Pre-trained model weights available for verification

## Technical Details

### Model Architecture
The model uses a U-Net architecture with:
- ResNet blocks with group normalization and SiLU activation
- Self-attention blocks for spatial attention
- Symmetric encoder-decoder paths with skip connections
- Learnable embedding layer for class conditioning
- Progressive spatial dimension reduction and channel depth increase

### Results
- Unconditional Model:
  - FID Score: 85.41
  - Inception Score: 1.03
  
- Conditional Model (by KL Grade):
  - Grade 0: FID 177.82, IS 1.56
  - Grade 1: FID 173.35, IS 1.50
  - Grade 2: FID 181.05, IS 1.52
  - Grade 3: FID 182.03, IS 1.57
  - Grade 4: FID 220.93, IS 1.57

## Impact

This work addresses two critical needs in medical imaging:
1. Augmenting training data for classification models by generating synthetic but realistic knee X-rays
2. Creating a conditioned dataset to help train medical students and physicians by providing diverse examples across different severity grades

## Citation

If you use this work in your research, please cite:

```bibtex
@techreport{anjum2025generating,
  title={Generating Severity-conditioned Knee Osteoarthritis X-rays Using Diffusion Neural Networks},
  author={Anjum, Khizar},
  institution={Rutgers University},
  year={2025},
  url={https://github.com/khizar-anjum/KneeOAGen/blob/main/technical_report.pdf}
}
```
