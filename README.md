# Exploring Contrast Consistency of Open-Domain Question Answering Systems on Minimally Edited Questions

This repository contains the data and code for the paper [*Exploring Contrast Consistency of Open-Domain Question Answering Systems on Minimally Edited Questions*](https://arxiv.org/pdf/2305.14441.pdf) in TACL 2023. In this study, we explored the problem of contrast consistency in open-domain question answering by collecting **M**inimally **E**dited **Q**uestions (MEQs) as challenging contrast sets to the popular Natural Questions (NQ) benchmark, in addition to its standard test set. Through our experiments, we find that the widely used dense passage retrieval (DPR) model performs poorly on distinguishing training questions and their minimally-edited contrast set questions. Moving a step forward, we improved the contrast consistency of DPR model via data augmentation and a query-side contrastive learning objective.


### Datasets
To be updated


### Environment

The Python environment mainly follows the one used by the [original DPR repo](https://github.com/facebookresearch/DPR).

1. Install PyTorch:
```bash
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
```

2. Install the other dependencies:
```bash
pip install -r requirements.txt
```

### Checkpoints

To be updated.

### Citation

If you use our data or code, please kindly cite our paper:
```
@article{zhang2023exploring,
  author={Zhihan Zhang and Wenhao Yu and Zheng Ning and Mingxuan Ju and Meng Jiang},
  title={Exploring Contrast Consistency of Open-Domain Question Answering Systems on Minimally Edited Questions},
  journal={Transactions of the Association for Computational Linguistics},
  volume={11},
  year={2023},
  publisher={MIT Press}
}
```
