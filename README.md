# Investigating Subjective Factors of Argument Strength: Storytelling, Emotions, and Hedging

## Overview
This repository contains the source code and data for the ArgMining 2025 paper titled "Investigating Subjective Factors of Argument Strength: Storytelling, Emotions, and Hedging" [[paper](https://webis.de/publications.html#quensel_2025)].

> **Abstract**
> 
> In assessing argument strength, the notions of what makes a good argument are manifold. With the broader trend towards treating subjectivity as an asset and not a problem in NLP, new dimensions of argument quality are studied. Although studies on individual subjective features like personal stories exist, there is a lack of large-scale analyses of the relation between these features and argument strength. To address this gap, we conduct regression analysis to quantify the impact of subjective factors $-$ emotions, storytelling, and hedging $-$ on two standard datasets annotated for objective argument quality and subjective persuasion. As such, our contribution is twofold: at the level of contributed resources, as there are no datasets annotated with all studied dimensions, this work compares and evaluates automated annotation methods for each subjective feature. At the level of novel insights, our regression analysis uncovers different patterns of impact of subjective features on the two facets of argument strength encoded in the datasets. Our results show that storytelling and hedging have contrasting effects on objective and subjective argument quality, while the influence of emotions depends on their rhetoric utilization rather than the domain.

## Experiments
To re-run the experiments and obtain the same results as reported in the paper, you should run the two Jupyter notebooks under the following specifications:

```
python==3.12.6
forestplot==0.4.1
matplotlib==3.9.2
numpy==2.2.0
pandas==2.2.3
scipy==1.14.1
seaborn==0.13.2
statsmodels==0.14.4
```

Running the code will create and populate a directory `img` with SVG files.

The original datasets used as the basis for the analysis and for training the classifiers are provided under `data/argument`, `data/emotion`, and `data/storytelling`, though these versions already include some preprocessing steps in accordance with the original papers (argumentation) or to sample data for each feature (emotion). The final versions of the datasets as used in the analyses can be obtained by running the Python scripts in `data/` and consolidating the results of all three features. For any questions, please contact the first author: c.quensel@ai.uni-hannover.de.

## Citation

If you use this code in your research, please cite:

```bib
@InProceedings{quensel:2025,
  author =                      {Carlotta Quensel and Neele Falk and Gabriella Lapesa},
  booktitle =                   {12th Workshop on Argument Mining (ArgMining 2025) at ACL},
  codeurl =                     {https://github.com/CarlottaQuensel/subjective-argument-strength},
  doi =                         {},
  editor =                      {Elena Chistova and Philipp Cimiano and Shohreh Haddadan and Gabriella Lapesa and Ramon Ruiz-Dolz},
  keywords =                    {args, argument, argument mining, natural language processing, nlp},
  month =                       jul,
  pages =                       {},
  publisher =                   {Association for Computational Linguistics},
  site =                        {Wien, Austria},
  title =                       {{Investigating Subjective Factors of Argument Strength: Storytelling, Emotions, and Hedging}},
  year =                        2025
  }

```
