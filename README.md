# What makes a good argument? Investigating subjective factors of argument strength

This repository contains all code relevant to the thesis *What makes a good argument? Investigating subjective factors of argument strength* submitted on November 2, 2023 
to obtain the Master of Science in Computational Linguistics from the University of Stuttgart. The thesis is included as a PDF and all relevant data and code resources are
linked and referenced in the corresponding code.

## Data
The thesis uses two argument datasets as basis for a regression analysis, as well as an emotion and storytelling dataset as training data to annotate the argument corpora.
Further, methods and lexicons for hedge detection are adopted and modified from three sources which are linked in the thesis.
### Results
The aggregated results of annotating the two argument datasets with the three features of storytelling, hedging, and emotions are saved under results. Rerunning the code for
each feature results in additional files, e.g., individual evaluation results for each training split (10 splits for 11 emotions and storytelling), the non-aggregated annotations
on the argument corpora for each feature separately, etc.
## Regression analysis
The two notebooks <IBM_regression.ipynb> and <CMV_regression.ipynb> contain the main analysis regressing each of the three features separately and together (up to 2-way interaction)
on the argument strength annotation of the corresponding corpus, i.e., the argument quality score of IBM ArgQ-5.3k and the persuasiveness label of Cornell-CMV respectively.
