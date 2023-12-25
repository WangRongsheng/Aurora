![](https://github.com/WangRongsheng/Aurora-Mixtral-8x7B-Chat/blob/main/assets/aurora.png)

<div align="center">
<h2>
  Aurora: Activating chinese chat capability for Mistral-8x7B sparse Mixture-of-Experts through Instruction-Tuning
</h2>
</div>

***Rongsheng Wang**, Haoming Chen, Ruizhe Zhou, Yaofei Duan, Kunyan Cai, Han Ma, Jiaxi Cui, Jian Li, Patrick Cheong-Iao Pang, Yapeng Wang, Tao Tan☨*

☨Corresponding author

<h5 align="center">

<a href='https://arxiv.org/abs/2312.14557'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>  <a href='https://huggingface.co/wangrongsheng/Aurora'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>

</h5>



## Overview

Existing research has demonstrated that refining large language models (LLMs) through the utilization of machine-generated instruction-following data empowers these models to exhibit impressive zero-shot capabilities for novel tasks, without requiring human-authored instructions. In this paper, we systematically investigate, preprocess, and integrate three Chinese instruction-following datasets with the aim of enhancing the Chinese conversational capabilities of Mixtral-8x7B sparse Mixture-of-Experts model. Through instruction fine-tuning on this carefully processed dataset, we successfully construct the Mixtral-8x7B sparse Mixture-of-Experts model named "Aurora." To assess the performance of Aurora, we utilize three widely recognized benchmark tests: C-Eval, MMLU, and CMMLU. Empirical studies validate the effectiveness of instruction fine-tuning applied to Mixtral-8x7B sparse Mixture-of-Experts model. This work is pioneering in the execution of instruction fine-tuning on a sparse expert-mixed model, marking a significant breakthrough in enhancing the capabilities of this model architecture.

## Evaluation

It is known that LLM evaluation remains a significant challenge. We use three public benchmarks in our study.

![](./assets/eval.png)

<!--
|Model|[CMMLU](https://opencompass.org.cn/dataset-detail/CMMLU)|[MMLU](https://opencompass.org.cn/dataset-detail/MMLU)|[C-EVAL](https://opencompass.org.cn/dataset-detail/C-Eval)|
|:-|:-|:-|:-|
|Aurora(checkpoints-3000)|**49.69**|**67.74**|**51.9**|
|LLaMA-2-70B-Chat|43.3|63.8|44.3|
|LLaMA-65B|40.4|63.7|40.6|
-->

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

#### 2. Download Model

*Base Model*:
|Model|Download|
|:-|:-|
|Mixtral-8x7B-Instruct-v0.1|[HuggingFace](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/tree/125c431e2ff41a156b9f9076f744d2f35dd6e67a), [HuggingFace-mirror](https://hf-mirror.com/mistralai/Mixtral-8x7B-Instruct-v0.1/tree/125c431e2ff41a156b9f9076f744d2f35dd6e67a)|

*LoRA Model*:
|Model|Download|
|:-|:-|
|Aurora|[HuggingFace](https://huggingface.co/wangrongsheng/Aurora)|

> The huge model parameters are not convenient for you to manage your task, so we provide LoRA weights, which will be merged with the base model before inference. You don't have to worry about it.

#### 3. Inference

*Web*:
```python
CUDA_VISIBLE_DEVICES=0 python src/web_demo.py \
    --model_name_or_path ./Mixtral-8x7B-Instruct-v0.1 \
    --checkpoint_dir Aurora \
    --finetuning_type lora \
    --quantization_bit 4 \
    --template mistral
```
Then you can visit: http://127.0.0.1:7860/

*CLI*:
```python
CUDA_VISIBLE_DEVICES=0 python src/cli_demo.py \
    --model_name_or_path ./Mixtral-8x7B-Instruct-v0.1 \
    --checkpoint_dir Aurora \
    --finetuning_type lora \
    --quantization_bit 4 \
    --template mistral
```

*API*:
```python
CUDA_VISIBLE_DEVICES=0 python src/api_demo.py \
    --model_name_or_path ./Mixtral-8x7B-Instruct-v0.1 \
    --checkpoint_dir Aurora \
    --finetuning_type lora \
    --quantization_bit 4 \
    --template mistral
```

If you need to load weights for specific checkpoints, you can set them up like this: `--checkpoint_dir Aurora/checkpoint-5000`.

## Train

![](./assets/instruction_finetuning.png)

If you have a single GPU and its GPU memory size is larger than 48GB, you can train your own models.

<details>
<summary>Train your MoE model</summary>
  
```python
CUDA_VISIBLE_DEVICES=5 python   src/train_bash.py \
    --stage sft \
    --model_name_or_path ./Mixtral-8x7B-Instruct-v0.1 \
    --do_train \
    --dataset alpaca_zh,alpaca_gpt4_zh,sharegpt \
    --finetuning_type lora \
    --quantization_bit 4 \
    --overwrite_cache \
    --output_dir output/ \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16 \
    --template mistral \
    --lora_target q_proj,v_proj
```

`--quantization_bit 4` means you will use `QLoRA`, If you have a larger GPU memory size you can remove it and use `LoRA`.

</details>

<details>
<summary>Evaluation your MoE model</summary>
  
```python
CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
    --model_name_or_path ./Mixtral-8x7B-Instruct-v0.1 \
    --checkpoint_dir Aurora/checkpoint-5000 \
    --finetuning_type lora \
    --quantization_bit 4 \
    --template mistral \
    --task cmmlu \ # cmmlu, mmlu, ceval
    --split test \
    --lang en \ # zh, en
    --n_shot 5 \
    --batch_size 8
```

</details>

## Results

![](./assets/results1.png)
![](./assets/results2.png)
![](./assets/results3.png)
![](./assets/results4.png)
![](./assets/results5.png)
![](./assets/results6.png)
![](./assets/results7.png)
![](./assets/results8.png)
![](./assets/results9.png)

## Acknowledgments

This work is mainly done by the [Faculty of Applied Sciences](https://www.mpu.edu.mo/esca/zh/index.php) of the Macao Polytechnic University. The computational resources used in this work were obtained from AWS servers. The fine-tuning framework we used is [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), which brings a lot of convenience to our work. We also thank the public datasets from the open source community, such as [shareAI](https://huggingface.co/shareAI), [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM). Most importantly we are very grateful to [Mistral AI](https://mistral.ai/), who are leading a new technology boom that will dramatically change the future of technology development.

## Citation
If you find our work helpful, feel free to give us a cite.
```bib
@misc{wang2023auroraactivating,
      title={Aurora:Activating Chinese chat capability for Mistral-8x7B sparse Mixture-of-Experts through Instruction-Tuning}, 
      author={Rongsheng Wang and Haoming Chen and Ruizhe Zhou and Yaofei Duan and Kunyan Cai and Han Ma and Jiaxi Cui and Jian Li and Patrick Cheong-Iao Pang and Yapeng Wang and Tao Tan},
      year={2023},
      eprint={2312.14557},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
Please follow the [Apache 2.0 License](https://github.com/WangRongsheng/Aurora/blob/main/LICENSE).
