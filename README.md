![](https://github.com/WangRongsheng/Aurora-Mixtral-8x7B-Chat/blob/main/assets/aurora.png)

<div align="center">
<h2>
  Aurora: Activating chinese chat capability for Mistral-8x7B sparse Mixture-of-Experts through Instruction-Tuning
</h2>
</div>

<h5 align="center">
  
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-blue.svg)](https://huggingface.co/wangrongsheng/Aurora)
[![arXiv](https://img.shields.io/badge/Arxiv-2312.12999-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2312.12999.pdf) <br>

</h5>

## Overview

## Evaluation

It is known that LLM evaluation remains a significant challenge. We use three public benchmarks in our study.
|Model|[CMMLU](https://opencompass.org.cn/dataset-detail/CMMLU)|[MMLU](https://opencompass.org.cn/dataset-detail/MMLU)|[C-EVAL](https://opencompass.org.cn/dataset-detail/C-Eval)|
|:-|:-|:-|:-|
|Aurora(checkpoints-3000)|**49.69**|**67.74**||
|LLaMA-2-70B-Chat|43.3|63.8|44.3|
|LLaMA-65B|40.4|63.7|40.6|

<!--CMMLU：**Average: 49.69**</br>STEM: 44.69</br>Social Sciences: 52.03</br>Humanities: 49.14</br>Other: 51.58-->
<!--MMLU：**Average: 67.74**</br>STEM: 57.53</br>Social Sciences: 77.42</br>Humanities: 63.34</br>Other: 74.41-->

Next are some references we gave you about GPU memory usage during the training and inference stage. **Please note that we did all inference and training on a single GPU.**

|Stage|GPU Memory Usage|
|:-|:-|
|Training|~43 GiB|
|Inference|~25 GiB|

## Easy-to-Use

#### 1. Clone and Set up

```git
https://github.com/WangRongsheng/Aurora.git
cd Aurora
pip install -r requirements.txt
```

#### 1. Download Model

*Base Model*:
|Model|Download|
|:-|:-|
|Mixtral-8x7B-Instruct-v0.1|[HuggingFace](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/tree/125c431e2ff41a156b9f9076f744d2f35dd6e67a), [HuggingFace-mirror](https://hf-mirror.com/mistralai/Mixtral-8x7B-Instruct-v0.1/tree/125c431e2ff41a156b9f9076f744d2f35dd6e67a)|

*LoRA Model*:
|Model|Download|
|:-|:-|
|Aurora|[HuggingFace](https://huggingface.co/wangrongsheng/Aurora)|

## Acknowledgments

This work is mainly done by the [Faculty of Applied Sciences](https://www.mpu.edu.mo/esca/zh/index.php) of the Macao Polytechnic University. The computational resources used in this work were obtained from AWS servers. The fine-tuning framework we used is [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), which brings a lot of convenience to our work. We also thank the public datasets from the open source community, such as [shareAI](https://huggingface.co/shareAI), [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM). Most importantly we are very grateful to [Mistral AI](https://mistral.ai/), who are leading a new technology boom that will dramatically change the future of technology development.

## Citation
If you find our work helpful, feel free to give us a cite.
```bib
coming soon
```

## License
Please follow the [Apache 2.0 License](https://github.com/WangRongsheng/Aurora/blob/main/LICENSE).
