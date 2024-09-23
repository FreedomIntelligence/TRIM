# Less is More: A Simple yet Effective Token Reduction Method for Efficient Multi-modal LLMs

[Dingjie Song](https://dingjie-song.netlify.app/), Wenjun Wang, Shunian Chen, Xidong Wang, Michael Guan, Benyou Wang*


[[Paper](https://arxiv.org/abs/2409.10994)] [[Project Page](https://github.com/bbsngg/AdaptiveLLaVA)]

<div align="center">
  <img src="https://github.com/bbsngg/AdaptiveLLaVA/blob/main/images/TRIM.png" alt="Our approach" width="100%">
</div>



## How to run

### Step.0: Set the environment the same as LLaVA-1.5

Note that the core of our proposed module is [here](https://github.com/bbsngg/AdaptiveLLaVA/blob/main/llava/model/multimodal_encoder/clip_encoder.py) in the CLIP image encoder.  

### Step.1 (for inference): Download Checkpoints

Download the checkpoints from [huggingface](https://huggingface.co/liuhaotian/llava-v1.5-7b) to liuhaotian/llava-v1.5-7b.

### Step.2 (for inference): Change the methods (TextSim or TextSim+).

Change the call function of token reduction from [here](https://github.com/bbsngg/AdaptiveLLaVA/blob/main/llava/model/multimodal_encoder/clip_encoder.py) in the CLIP image encoder. 

### Step.3 (for inference): Run the script.

For example, the evaluation for TextVQA is:

```shell
CUDA_VISIBLE_DEVICES=7 XDG_CACHE_HOME='/data/shangyuzhang/' bash scripts/v1_5/eval/testvqa.sh
```

For other inference scripts, refer to [LLaVA Evaluation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).
