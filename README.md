 <h1>Pose Extraction & Rendering Code for SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations</h1>
  <div align="center">
  <a href='https://arxiv.org/abs/2512.05905'><img src='https://img.shields.io/badge/📖 arXiv-2512.05905-red'></a>
  <a href='https://teal024.github.io/SCAIL/'><img src='https://img.shields.io/badge/🌐 Project Page-green'></a>
  <a href="https://github.com/zai-org/SCAIL">
    <img src="https://img.shields.io/badge/%20Main GitHub Repo-181717?logo=github">
  </a>
</div>


This repository works as a submodule of the 3D pose extraction & rendering code for **SCAIL (Studio-Grade Character Animation via
In-Context Learning)**, a framework that enables high-fidelity character animation under diverse and challenging
conditions, including large motion variations, stylized characters, and multi-character interactions. The main repository is at [zai-org/SCAIL](https://github.com/zai-org/SCAIL) and please follow instructions in that repo to extract and render the pose.
<p align="center">
  <img src="resources/pose_teaser.png" alt="teaser" width="90%">
</p>


## 📋 Methods Outline
We connect estimated 3D human keypoints according to skeletal topology and represent bones as spatial cylinders. The resulting 3D skeleton is rasterized to obtain 2D motion guidance signals.

When processing multi-character data, we segment each character, extract their poses, and then render them together to achieve multi-character pose extraction.
<p align="center">
  <img src="resources/data.png" alt="data" width="90%">
</p>

Our multi-stage pose extraction pipeline provides robust estimations under multi-character interactions, benefiting from [NLFPose](https://github.com/isarandi/nlf)’s reliable depth estimation:
<p align="center">
  <img src='resources/pose_result.png' alt='Teaser' width='95%'>
</p>

Utilizing such representation, our framework further resolves the challenge that pose representations cannot simultaneously prevent identity leakage and preserve rich motion information.
<p align="center">
  <img src="resources/pose_comp.png" alt="comp" width="90%">
</p>

## 📄 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{yan2025scail,
  title={SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations},
  author={Yan, Wenhao and Ye, Sheng and Yang, Zhuoyi and Teng, Jiayan and Dong, ZhenHui and Wen, Kairui and Gu, Xiaotao and Liu, Yong-Jin and Tang, Jie},
  journal={arXiv preprint arXiv:2512.05905},
  year={2025}
}
```
