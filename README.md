# BLUCL: Boundary-guided Lightweight Unified Contrastive Learning Framework for Polyp Segmentation

## Official Implementation of the BLUCL Framework Currently Under Review

## Abstract:
Timely diagnosis of pre-malignant polyps is crucial for early detection of colorectal cancer. However, precise semantic segmentation of polyps in colonoscopy frames remains intricate owing to indistinct boundaries, irregular morphology and indiscernible contexts. Existing polyp segmentation paradigms predominantly employ computationally intensive supervised architectures heavily reliant on pre-trained backbones, while boundary-aware approaches primarily focus on local feature refinement and may not explicitly encourage semantically consistent representations under appearance variations. Conversely, contrastive representation-learning approaches can improve semantic discrimination but do not inherently enforce precise boundary delineation. To address these limitations, we propose BLUCL, a Boundary-guided Lightweight Unified Contrastive Learning framework for morphologically consistent polyp segmentation. BLUCL introduces a shared encoder coupled with a prototype CL branch and a supervised segmentation branch, jointly optimizing a unified learning objective. BLUCL integrates a novel Boundary-guided Contextual Attention (BCA) module that leverages morphological gradient-based boundary priors to enhance boundary representations through channel-spatial contextual refinement. The end-to-end framework constitutes residual depthwise separable convolutional blocks that substantially optimizes parameters. Additionally, Scale-aware Context Aggregation Block (SCAB) embedded within encoder-decoder skip connections, captures multi-scale polyp features. BLUCL (3.05M parameters; 7.89 GFLOPs) achieves competitive segmentation results, registering a mean Dice of 0.9144 and 0.9620 on Kvasir-SEG and CVC-ClinicDB, respectively. Despite its lightweight design, BLUCL consistently outperforms several competing methods, while utilizing approximately 50-97\% fewer parameters, demonstrating reasonable generalizability with boundary-consistent predictions evidenced through interpretability. The code is available at https://github.com/shadowwalker210/BLUCL-for-BMVC-2026.



## Code Availability

### Code will be released upon receiving a publication decision

