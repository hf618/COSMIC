<div align="left">
<h2>COSMIC (CVPR 2025)</h2>

<div>    
    <a href='https://hf618.github.io/' target='_blank'>Fanding Huang</a><sup>1 *</sup>&nbsp;
    <a href='https://www.jiangjingyan.com/' target='_blank'>Jingyan Jiang</a><sup>2 *</sup>&nbsp;
    <a href='https://dblp.org/pid/285/0493.html' target='_blank'>Qinting Jiang</a><sup>1</sup>&nbsp;
    <a href='https://scholar.google.cz/citations?user=rgeEZmsAAAAJ&hl=zh-CN' target='_blank'>Hebei Li</a><sup>3</sup>&nbsp;
    <a href='https://www.sigs.tsinghua.edu.cn/Faisal_en/main.psp' target='_blank'>Faisal Nadeem Khan</a><sup>1 †</sup>&nbsp;
    <a href='http://pages.mmlab.top/' target='_blank'>Zhi Wang</a><sup>1 †</sup>
</div>

<div>
    <sup>*</sup>These authors contributed equally to this work;</span> <sup>†</sup>Co-corresponding authors.</span>
</div>
<div>
    <sup>1</sup>Shenzhen International Graduate School, Tsinghua University, China
</div>
<div>
    <sup>2</sup>Shenzhen Technology University, China
</div>
<div>
    <sup>3</sup>University of Science and Technology of China, China
</div>

<div>
    <h4 align="left">
        <a href="https://hf618.github.io/COSMIC_Project/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2503.23388" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2412.11365-b31b1b.svg">
        </a>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/hf618/COSMIC">
    </h4>
</div>

---

<div align="center">
    <h4>
        This repository is the official PyTorch implementation of "COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation".
    </h4>
</div>

## Environment Setting
To run this project, you need to set up your environment as follows:
```bash
conda create -n COSMIC python=3.10 -y
conda activate COSMIC
pip install -r requirements.txt
```
## Dataset
For both Out-of-Distribution and Cross-Domain benchmarks, please refer to [this page](https://github.com/azshue/TPT).

## Run COSMIC
To execute the COSMIC, navigate to the `scripts` directory.

#### OOD Benchmark
Run COSMIC on the OOD Benchmark:
```
bash ./scripts/cosmic_ood.sh
```

#### Cross-Domain Benchmark
Run COSMIC on the Cross-Domain Benchmark:
```
bash ./scripts/cosmic_cd.sh
```


