# Guided Decoding in Goal-Conditioned Reward Models

This folder contains guided decoding experiments for the goal-conditioned rewards model paper. This folder is adapted from [Decomposition Enhances Reasoning via Self-Evaluation Guided Decoding](https://arxiv.org/abs/2305.00633) and their [Github Repo](https://github.com/YuxiXie/SelfEval-Guided-Decoding/tree/main/src). For our language model, we use Llama and do not use OpenAI to generate our langauge model generations. 

## Requirements

#### Environment

```
matplotlib                         3.3.4
numpy                              1.20.1
ipdb                               0.13.9
tqdm                               4.64.1
```

## Running

We show examples of how to run our method on different datasets in [`scripts`](scripts). Specifically, scripts with names starting with `run_generation_` are for running our methods with either PAL or CoT as basic prompting methods.

---
<sub><sup>This folder is adapted from the code of the works [PaL: Program-Aided Language Model](https://github.com/reasoning-machines/pal) and [Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks](https://github.com/wenhuchen/Program-of-Thoughts). </sup></sub>

As well as our work, please cite the original authors of this repo:

```
@misc{xie2023decomposition,
      title={Decomposition Enhances Reasoning via Self-Evaluation Guided Decoding}, 
      author={Yuxi Xie and Kenji Kawaguchi and Yiran Zhao and Xu Zhao and Min-Yen Kan and Junxian He and Qizhe Xie},
      year={2023},
      eprint={2305.00633},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



