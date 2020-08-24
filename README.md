# Covid 19 Simulation and Presecription

This repository is the reference for the paper 

*Data-driven Simulation and Optimization for Covid-19 Exit Strategies*
**DOI: 10.1145/3394486.3412863 @ KDD '20: The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining**

(a pre-print version is also available at : https://arxiv.org/abs/2006.07087)

## Abstract

The rapid spread of the Coronavirus SARS-2 is a major challenge that led almost all governments worldwide to take drastic measures to respond to the tragedy. Chief among those measures is the massive lockdown of entire countries and cities, which beyond its global economic impact has created some deep social and psychological tensions within populations. While the adopted mitigation measures (including the lockdown) have generally proven useful, policymakers are now facing a critical question: how and when to lift the mitigation measures? A carefully-planned exit strategy is indeed necessary to recover from the pandemic without risking a new outbreak. Classically, exit strategies rely on mathematical modeling to predict the effect of public health interventions. Such models are unfortunately known to be sensitive to some key parameters, which are usually set based on rules-of-thumb.

In this paper, we propose to augment epidemiological forecasting with actual data-driven models that will learn to fine-tune predictions for different contexts (e.g., per country). We have therefore built a pandemic simulation and forecasting toolkit that combines a deep learning estimation of the epidemiological parameters of the disease in order to predict the cases and deaths, and a genetic algorithm component searching for optimal trade-offs/policies between constraints and objectives set by decision-makers.

Replaying pandemic evolution in various countries, we experimentally show that our approach yields predictions with much lower error rates than pure epidemiological models in 75% of the cases and achieves a 95% R² score when the learning is transferred and tested on unseen countries. When used for forecasting, this approach provides actionable insights into the impact of individual measures and strategies.


## Structure

* models/: pre-trained MLP model (feature processing, scaler) for predicting the Reproduction Number
* data/: the initial raw features used for training
* ga/: the definition of the multi-objective genetic algorithm problem
* utils/: the definition of the SEI-HCDR epidemiological model and the hill decay fitting. 

### Replication Scripts
* model_run.py: the SEI-HCDR epidemiological simulation (RQ1)
* fit_experiments: the SEI-HCDR epidemiological fitting to build the Rt ground truth for the Machine Learning model (RQ1)
* model_taining: the Machine Learning training and prediction (RQ2)
* interpretability: global interpretation using SHAP (RQ2)
* exit_strategies.py: manual prediction of different scenarios (RQ2)
* ga_experiments.py: the Genetic Algorithm Search for optimal scenarios (RQ3)
* ga_analysis: Pareto extraction of optimal scenarios among the last population of scenarios (RQ3)


