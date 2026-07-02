---
title: "Generative Control as Optimization: Time Unconditional Flow Matching for Adaptive and Robust Robotic Control"
authors:
  - Zunzhe Zhang
  - Runhan Huang
  - Yicheng Liu
  - Shaoting Zhu
  - Linzhan Mou
  - Hang Zhao
date: "2026-04-26T00:00:00Z"
publishDate: "2026-03-18T00:00:00Z"
publication_types: ["paper-conference"]
publication:
  name: "ICLR Workshops (LLA and ReALM-GEN)"
  short_name: "ICLR Workshops"
peer_reviewed: true
open_access: true
abstract: "Generative Control as Optimization (GeCO) reframes flow-matching action generation for robotic control as adaptive iterative optimization. Instead of following a fixed integration schedule, GeCO learns a stationary velocity field in action-sequence space so inference can stop early for simple states and refine longer for difficult states. The same stationary geometry also provides a training-free signal for detecting out-of-distribution states, supporting safer deployment of Vision-Language-Action policies."
summary: "GeCO turns flow-matching robotic control into adaptive optimization, improving efficiency, robustness, and out-of-distribution awareness for VLA policies."
tags:
  - Vision-Language-Action Models
  - Robotic Control
  - Flow Matching
  - Imitation Learning
  - Out-of-Distribution Detection
featured: true
hugoblox:
  ids:
    arxiv: 2603.17834
    doi: 10.48550/arXiv.2603.17834
links:
  - type: custom
    label: arXiv
    url: https://arxiv.org/abs/2603.17834
  - type: pdf
    label: PDF
    url: https://arxiv.org/pdf/2603.17834
  - type: custom
    label: Project
    url: https://hrh6666.github.io/GeCO/
projects: []
slides: ""
image:
  caption: ''
  focal_point: ''
  preview_only: false
---

GeCO addresses a structural inefficiency in diffusion and flow-matching policies: a fixed inference schedule spends the same computation on easy and difficult states. The method treats action synthesis as optimization over a learned stationary velocity field, allowing adaptive computation at test time.

In addition to improving the success-latency trade-off, the field norm after optimization serves as an intrinsic OOD indicator. This makes the approach useful not only as a replacement for flow-matching action heads in VLA policies, but also as a mechanism for safer robotic deployment.

Zunzhe Zhang and Runhan Huang contributed equally to this work.
