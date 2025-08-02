# DGMHA-ProtoNet
Official Code for "Towards Interpretable and Accurate Breast Cancer Classification via Dual-Granularity Attention and Optimized Prototype Learning"

**Abstract:**
Breast cancer remains the most prevalent malignancy among women globally, underscoring the critical need for early detection and accurate classification. However, existing deep learning-based computer-aided diagnosis (CAD) systems often struggle to simultaneously achieve high accuracy and provide meaningful interpretability. Addressing this critical trade-off, we propose a novel framework that synergistically integrates an attention-enhancement mechanism with prototype-based interpretability.
At the core of our approach is a Dual-Granularity Multi-Head Attention (DGMHA) module, which is designed to capture both coarse- and fine-grained long-range dependencies, thereby enriching the expressiveness of visual features. To enhance interpretability, we introduce a Dual Latent Space Prototype Projection mechanism. This component strategically leverages both the shallow embedding space from the backbone and the deep embedding space from our attention-enhanced features, facilitating a more precise mapping between learned prototypes and representative training image patches.
Extensive experiments on the BreaKHis breast cancer dataset demonstrate that our method not only achieves superior classification accuracy but also provides more faithful and intuitive visual explanations.

[框架概图.pdf](https://github.com/user-attachments/files/21555953/default.pdf)
