# Covid 19 Simulation and Presecription

This repository is the reference for the paper 

*Data-driven Simulation and Optimization for Covid-19 Exit Strategies*
**DOI: 10.1145/3394486.3412863 @ KDD '20: The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining**

and for the online simulator located on https://serval-snt.github.io/covid19**

The Master branch is the one used for the live simulator. 

Use the *replication* branch to get the full experiments for our KDD paper.

Documentation and Guides are WiP


## Abstract

The rapid spread of the Coronavirus SARS-2 is a major challenge that led almost all governments worldwide to take drastic measures to respond to the tragedy. Chief among those measures is the massive lockdown of entire countries and cities, which beyond its global economic impact has created some deep social and psychological tensions within populations. While the adopted mitigation measures (including the lockdown) have generally proven useful, policymakers are now facing a critical question: how and when to lift the mitigation measures? A carefully-planned exit strategy is indeed necessary to recover from the pandemic without risking a new outbreak. Classically, exit strategies rely on mathematical modeling to predict the effect of public health interventions. Such models are unfortunately known to be sensitive to some key parameters, which are usually set based on rules-of-thumb.

In this paper, we propose to augment epidemiological forecasting with actual data-driven models that will learn to fine-tune predictions for different contexts (e.g., per country). We have therefore built a pandemic simulation and forecasting toolkit that combines a deep learning estimation of the epidemiological parameters of the disease in order to predict the cases and deaths, and a genetic algorithm component searching for optimal trade-offs/policies between constraints and objectives set by decision-makers.

Replaying pandemic evolution in various countries, we experimentally show that our approach yields predictions with much lower error rates than pure epidemiological models in 75% of the cases and achieves a 95% R² score when the learning is transferred and tested on unseen countries. When used for forecasting, this approach provides actionable insights into the impact of individual measures and strategies.

## Installation:
Use the ./requirements.txt file to install the dependencies

### Branch Master (Simulator)
The simulator is designed to work with Google App Engine Flex and uses the endpoint *main.py* and its configuration file is *app.yaml*.

It is also available as a plain **FLASK Webserver** with the endpoint *backML.py*. Running 
```python
python backML.py
```
will spawn a web server listening at the address *http://0.0.0.0:8080/* 

The main route is **POST /predict** with a json body as a calendar of measures we want to predict the outcome

You can find an example of a web client (using gatsby) on *https://github.com/yamizi/serval-snt.github.io*


### Branch Replication (KDD Experiments)
The branch is split in different script files for each research question.  
Refer to the README.md of the branch **replication** for more details.