# Less is More: A Simple yet Effective Token Reduction Method for Efficient Multi-modal LLMs

[Dingjie Song](https://bbsngg.github.io/), Wenjun Wang, Shunian Chen, Xidong Wang, Michael Guan, Benyou Wang*

[**🤗 Paper**](https://arxiv.org/abs/2409.10994) | [**📖 arXiv**](https://arxiv.org/abs/2409.10994) | [**GitHub**](https://github.com/bbsngg/AdaptiveLLaVA)]

<div align="center">
  <img src="https://github.com/bbsngg/AdaptiveLLaVA/blob/main/images/TRIM.png" alt="Our approach" width="100%">
</div>

## 🌈 Update

- **[2024.9.23]** 🎉🎉🎉 TRIM is public!🎉🎉🎉



## How to run

### Step.0: Set the environment the same as LLaVA-1.5

Note that the core of our proposed module is [here](https://github.com/bbsngg/AdaptiveLLaVA/blob/main/llava/model/multimodal_encoder/clip_encoder.py) in the CLIP image encoder.  

### Step.1 (for inference): Download Checkpoints

Download the checkpoints from [huggingface](https://huggingface.co/liuhaotian/llava-v1.5-7b) to liuhaotian/llava-v1.5-7b.

### Step.2 (for inference): Change the methods (TextSim or TextSim+).

Change the call function of token reduction from [here](https://github.com/bbsngg/AdaptiveLLaVA/blob/main/llava/model/multimodal_encoder/clip_encoder.py) in the CLIP image encoder. 

### Step.3 (for inference): Run the script.

For all benchmark，the evaluation is：

```shell
bash eval_all_benchmarks.sh
```

For example, the evaluation for TextVQA is:

```shell
bash scripts/v1_5/eval/testvqa.sh
```

For other inference scripts, refer to [LLaVA Evaluation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).



## License

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](https://github.com/MileBench/MileBench/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-orange.svg)](https://github.com/MileBench/MileBench/blob/main/DATA_LICENSE)

All software is licensed under the Apache License, Version 2.0 (Apache 2.0).
All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY).



## Contact

- Dingjie Song: bbsngg@outlook.com
- Benyou Wang: wangbenyou@cuhk.edu.cn

## Citation

If you find this repository helpful, please consider citing it:


```
@misc{song2024moresimpleeffectivetoken,
      title={Less is More: A Simple yet Effective Token Reduction Method for Efficient Multi-modal LLMs}, 
      author={Dingjie Song and Wenjun Wang and Shunian Chen and Xidong Wang and Michael Guan and Benyou Wang},
      year={2024},
      eprint={2409.10994},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.10994}, 
}
```

