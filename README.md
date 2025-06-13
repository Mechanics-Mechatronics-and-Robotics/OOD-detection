# Project: OOD Detection Enhanced with World Knowledge

![Status](https://img.shields.io/badge/status-in%20progress-yellow)

> **Summer internship project** exploring enhanced OOD detection through the integration of world knowledge from LLMs.  
> Final goal: NeurIPS 2025 Workshop submission.


This repository was developed using methodologies suggested by the ChatGPT (OpenAI, 2025) language model.

## Problem Statement

Out-of-distribution (OOD) detection is critical for improving the reliability and robustness of AI models. Recent advancements using foundation models such as CLIP (Radford et al., 2021) have already shown significant improvements in OOD detection. Building upon this, our project aims to further enhance OOD detection by integrating the vast world knowledge encoded in modern large language models (LLMs). By leveraging multi-modal world models and extending their use to several learning tasks—including multi-label classification, object detection, image segmentation, and even online learning—we hope to create more dependable systems. 

As a **Plan B**, we also propose exploring **OOD Detection & Open-Set Noisy Labels**. Here, instead of completely suppressing open-set noisy samples, the idea is to use them beneficially for outlier exposure, potentially improving the OOD detection process in semi-supervised settings.

## Keywords

- Out-of-Distribution (OOD) Detection  
- World Knowledge Integration  
- Multi-modal Models  
- Foundation Models (e.g., CLIP)  
- Large Language Models (LLMs)  
- Open-Set Noisy Labels  
- Unsolvable Problem Detection (UPD)

## Possible Solution Paths

1. **OOD Detection Enhanced with World Knowledge**  
   - **Data & Models**: Use benchmark OOD datasets such as ImageNet-O or CIFAR-10/100 with known OOD splits.  
   - **LLM Integration**: Leverage LLMs (e.g., GPT-4, DeepSeek) to inject world knowledge for contextualizing model predictions. For instance, the LLM can provide semantic embeddings or textual cues that aid in discriminating between in-distribution and OOD samples.  
   - **Multi-Modal Fusion**: Combine outputs from multi-modal models (e.g., CLIP) with insights from LLMs to refine OOD detection thresholds.
   - **Evaluation**: Quantitatively compare the performance against baseline OOD detectors, looking at metrics like AUROC, false positive rate, etc.

2. **Plan B – OOD Detection & Open-Set Noisy Labels**  
   - **Data Preparation**: Collect datasets that include open-set noisy labels.  
   - **Learning from Noise**: Instead of simply filtering out noisy samples, utilize them for outlier exposure by integrating techniques from robust learning and semi-supervised learning.
   - **Hybrid Strategy**: Design a dual-stage model where one branch learns to classify in-distribution samples while the other adjusts dynamically to noisy/outlier signals, possibly using an LLM to provide context or suggest re-labeling strategies.
   - **Analysis**: Evaluate if incorporating open-set noisy data improves both classification accuracy and OOD detection, particularly when compared to traditional noise suppression techniques.

3. **OOD Detection for Broader Learning Tasks**  
   - **Task Extension**: Investigate the application of the above OOD methods to multi-label classification, object detection, and segmentation tasks.  
   - **Zero-Shot Evaluation**: Examine how well pre-trained zero-shot models (e.g., CLIP) perform on OOD tasks when enhanced with additional world knowledge.
   - **Adaptive Scenarios**: Develop and test methods for continuously adaptive or online learning environments where the OOD detector must update in real time.
   - **Upscaling**: Explore the potential of using OOD detection in safety-critical applications such as robotic manipulation or medical imaging, and analyze how world knowledge can mitigate hallucination or ambiguity issues.

## End Goal

The final deliverable is a concise workshop paper (approximately 4 pages) aiming for a NeurIPS 2025 Workshop submission. The paper should:
- Document the proposed methodologies.
- Compare the enhanced OOD detection performance against baselines.
- Include discussions on the broader applicability and limitations.
- Explore the potential of merging world knowledge and open-set noisy labels for robust AI.

## Milestones

1. **Week 1–2:**  
   - Literature review (including the Springer review [link](https://link.springer.com/article/10.1007/s11263-024-02117-4) and relevant foundational papers).  
   - Dataset selection and baseline setup (e.g., ImageNet-O, CIFAR-10/100 OOD splits).
2. **Week 3–4:**  
   - Implementation of baseline OOD detection models (e.g., using CLIP as a starting point).  
   - Develop an initial integration strategy for LLM-based world knowledge.
3. **Week 5–6:**  
   - Integrate multi-modal fusion and test additional solution paths (Plan B: OOD & open-set noisy labels).  
   - Run experiments and analyze results.
4. **Week 7–8:**  
   - Refine models and consolidate findings.  
   - Prepare visualizations and draft the workshop paper.

## Datasets

- [ImageNet-O]([https://paperswithcode.com/dataset/imagenet-o) (OOD splits for ImageNet)  
- CIFAR-10/100 with OOD partitions 
- Custom open-set noisy label datasets (for Plan B)

## Tools & Libraries

- PyTorch, TensorFlow (optional)  
- HuggingFace Transformers (for LLM integration)  
- Multi-modal frameworks (e.g., CLIP implementations)  
- Experiment tracking tools (Weights & Biases, ClearML)  
- Data processing libraries (NumPy, Pandas)

## References
- [NIPS-2025](https://neurips.cc/Conferences/2025/CallForPapers)
- Yang, et al. (2024). *Generalized Out-of-Distribution Detection: A Survey*. https://arxiv.org/abs/2110.11334
- Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). https://arxiv.org/abs/2103.00020
