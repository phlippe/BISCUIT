# 🍪 BISCUIT: Causal Representation Learning from Binary Interactions

[![ProjectPage](https://img.shields.io/static/v1.svg?logo=html&label=Website&message=Project%20Page&color=red)](https://phlippe.github.io/BISCUIT/)
[![Demo](https://colab.research.google.com/assets/colab-badge.svg?label=Demo)](https://colab.research.google.com/github/phlippe/BISCUIT/blob/main/demo.ipynb)
[![Paper](https://img.shields.io/static/v1.svg?logo=arxiv&label=Paper&message=Open%20Paper&color=green)](https://arxiv.org/abs/2306.09643)
[![Datasets](https://img.shields.io/static/v1.svg?logo=zenodo&label=Zenodo&message=Download%20Datasets&color=blue)](https://zenodo.org/record/8027138)   

This is the official code repository for the papers **"BISCUIT: Causal Representation Learning from Binary Interactions"** (UAI 2023), by Phillip Lippe, Sara Magliacane, Sindy Löwe, Yuki M. Asano, Taco Cohen, and Efstratios Gavves. For a paper summary, check out our [project page](https://phlippe.github.io/BISCUIT/)!

> **Note**
> We are currently in the process of releasing the code and the demo. Once the code has been approved by our funding partners, it will be uploaded here. We expect the process to take max. 4 weeks (expected release latest mid-July, before UAI). 


## Citation

If you use this code or find it otherwise helpful, please consider citing our work:
```bibtex
@inproceedings{lippe2023biscuit,
   title        = {{BISCUIT: Causal Representation Learning from Binary Interactions}},
   author       = {Lippe, Phillip and Magliacane, Sara and L{\"o}we, Sindy and Asano, Yuki M and Cohen, Taco and Gavves, Efstratios},
   year         = 2023,
   booktitle    = {The 39th Conference on Uncertainty in Artificial Intelligence},
   url          = {https://openreview.net/forum?id=VS7Dn31xuB},
   abstract     = {Identifying the causal variables of an environment and how to intervene on them is of core value in applications such as robotics and embodied AI. While an agent can commonly interact with the environment and may implicitly perturb the behavior of some of these causal variables, often the targets it affects remain unknown. In this paper, we show that causal variables can still be identified for many common setups, e.g., additive Gaussian noise models, if the agent's interactions with a causal variable can be described by an unknown binary variable. This happens when each causal variable has two different mechanisms, e.g., an observational and an interventional one. Using this identifiability result, we propose BISCUIT, a method for simultaneously learning causal variables and their corresponding binary interaction variables. On three robotic-inspired datasets, BISCUIT accurately identifies causal variables and can even be scaled to complex, realistic environments for embodied AI.}
}
```

### Contact

If you have questions or found a bug, feel free to open a github issue or send a mail to p.lippe@uva.nl. 
