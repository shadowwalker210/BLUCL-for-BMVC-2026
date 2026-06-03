# BLUCL: Boundary-guided Lightweight Unified Contrastive Learning Framework for Polyp Segmentation

## Official Implementation of the BLUCL Framework Currently Under Review with BMVC 2026

## Complete code will be released upon receiving a publication decision

## Abstract:
Timely diagnosis of pre-malignant polyps is crucial for early detection of colorectal cancer. However, precise semantic segmentation of polyps in colonoscopy frames remains intricate owing to indistinct boundaries, irregular morphology and indiscernible contexts. Existing polyp segmentation paradigms predominantly employ computationally intensive supervised architectures heavily reliant on pre-trained backbones, with limited emphasis on boundary-aware delineation. Despite recent advances, the integration of boundary-guided feature refinement within a lightweight hybrid supervised-contrastive framework Contrastive Learning (CL) framework for discriminative region-aware representation learning is relatively under-explored. To address these limitations, we propose BLUCL, a Lightweight Unified Boundary-guided Contrastive Learning framework for morphologically consistent polyp segmentation. BLUCL introduces a shared encoder coupled with a prototype CL branch and a supervised segmentation branch, jointly optimizing a unified learning objective. BLUCL integrates a novel Boundary-guided Contextual Attention (BCA) module that leverages morphological gradient-based boundary priors to enhance boundary representations through channel-spatial contextual refinement. The end-to-end framework constitutes residual depthwise separable convolutional blocks that substantially optimizes parameters. Additionally, Scale-aware Context Aggregation Block (SCAB) embedded within encoder-decoder skip connections, captures multi-scale polyp features. BLUCL (3.05M parameters; 7.89 GFLOPs) achieves competitive segmentation results, registering a mean Dice of 0.9144 and 0.9620 on Kvasir-SEG and CVC-ClinicDB, respectively. Despite its lightweight design, BLUCL consistently outperforms several competing methods, while utilizing approximately 50-97\% fewer parameters, demonstrating reasonable generalizability with boundary-consistent predictions evidenced through interpretability.

## Overview of Proposed BLUCL Architecture:

<p align="center">
  <img src="BLUCL Diagram.png" width="1000">
</p>


## Code Availability

This repository contains the core implementation of the BLUCL architecture in the BLUCL.py script.
The complete training pipeline, optimization code, and associated experimental components will be released upon receiving a publication decision. Furthermore, pretrained model weights will be provided to facilitate inference and evaluation.
