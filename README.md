# Don't `Well, Actually' Me Unless You Know What You're Talking About: Weak Presupposition Verification Degrades General QA Performance
### Shenran Wang, Vered Shwartz, Hila Gonen

🤗 **[Dataset](https://huggingface.co/datasets/shenranw/CancerMyth-TPQ)** | 📄 **[Paper Link](https://arxiv.org/abs/2608.06539)**

Quick Starter
-------------
Setup the environment with [env_setup.sh](./job_scripts/preprocess/env_setup.sh).  
Preprocess datasets with [prepare_datasets.sh](./job_scripts/preprocess/prepare_datasets.sh).  

Reproducing Methods
-------------
**Prompt-based methods**: [prompting](./job_scripts/prompting/pipeline/).  
**GEPA**: [FPQ-only](./job_scripts/GEPA/) and [FPQ-TPQ mixed](./job_scripts/GEPA_balanced/).  
**FAITH**: [FAITH](./job_scripts/FAITH/).  
**Fine-tuning**: [FalseQA](./job_scripts/FalseQA/).  

Fact-checking Ablation Studies
-------------
See [check_gold](./job_scripts/check_gold/)

Citation
-------------
```tex
@misc{wang2026dontwellactuallyunless,
      title={Don't `Well, Actually' Me Unless You Know What You're Talking About: Weak Presupposition Verification Degrades General QA Performance}, 
      author={Shenran Wang and Vered Shwartz and Hila Gonen},
      year={2026},
      eprint={2608.06539},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.06539}, 
}
```
