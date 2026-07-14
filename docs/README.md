# NitroGen — step-by-step guide

A ten-chapter walkthrough of the whole vision-action foundation model. Each
chapter is short and links directly to the code it explains. Read in order:

1. [Overview: what NitroGen is and why it matters](01_overview.md)
2. [The standardized gamepad action space](02_action_space.md)
3. [Building a video-action dataset from controller overlays](03_data_pipeline.md)
4. [The vision encoder (SigLIP-style ViT)](04_vision_encoder.md)
5. [Flow matching — the generative engine](05_flow_matching.md)
6. [The action head — a Diffusion Transformer](06_action_head.md)
7. [Training — large-scale behavior cloning](07_training.md)
8. [Inference — action chunking and closed-loop play](08_inference.md)
9. [The multi-game benchmark](09_benchmark.md)
10. [Cross-game transfer — the generalist payoff](10_transfer.md)

New here? Start with the [repository README](../README.md) for the big picture and
quickstart, then come back and work through these in sequence.
