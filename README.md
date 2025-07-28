# Improved Uncertainty Quantification in Physics-Informed Neural Networks Using Error Bounds and Solution Bundles

This repository contains the code for the paper "Improved Uncertainty Quantification in Physics-Informed Neural Networks Using Error Bounds and Solution Bundles".

# Citation

```
@InProceedings{pmlr-v286-flores25a,
  title = 	 {Improved Uncertainty Quantification in Physics-Informed Neural Networks Using Error Bounds and Solution Bundles},
  author =       {Flores, Pablo and Graf, Olga and Protopapas, Pavlos and Pichara, Karim},
  booktitle = 	 {Proceedings of the Forty-first Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {1289--1336},
  year = 	 {2025},
  editor = 	 {Chiappa, Silvia and Magliacane, Sara},
  volume = 	 {286},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--25 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v286/main/assets/flores25a/flores25a.pdf},
  url = 	 {https://proceedings.mlr.press/v286/flores25a.html},
  abstract = 	 {Physics-Informed Neural Networks (PINNs) have been widely used to obtain solutions to various physical phenomena modeled as Differential Equations. As PINNs are not naturally equipped with mechanisms for Uncertainty Quantification, some work has been done to quantify the different uncertainties that arise when dealing with PINNs. In this paper, we use a two-step procedure to train Bayesian Neural Networks that provide uncertainties over the solutions to differential equation systems provided by PINNs. We use available error bounds over PINNs to formulate a heteroscedastic variance that improves the uncertainty estimation. Furthermore, we solve forward problems and utilize the obtained uncertainties when doing parameter estimation in inverse problems in cosmology.}
}
```
