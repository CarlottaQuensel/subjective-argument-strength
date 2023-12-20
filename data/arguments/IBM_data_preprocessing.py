# The IBM Arg-Q 5.3k data used as basis for argument quality regression is obtained from https://research.ibm.com/haifa/dept/vst/debating_data.shtml#Argument%20Quality
# and is part of 
# Assaf Toledo, Shai Gretz, Edo Cohen-Karlik, Roni Friedman, Elad Venezian, Dan Lahav, Michal Jacovi, Ranit Aharonov, and Noam Slonim. 2019. 
# Automatic argument quality assessment - new datasets and methods. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language 
# Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5625-5635, Hong Kong, China. Association 
# for Computational Linguistics.

from os import listdir
import pandas as pd

# The relevant arguments of the ArgQ-5.3k corpus are split into files by their stance, i.e. for 11
# topics in the dataset, the 5.3k arguments are split over 22 files.
files = listdir("IBM-ArgQ-5.3kArgs")
stances = [" ".join(file[:len(file)-10].split("-")) for file in files]

data = {"text_id": [], "text": [], "rank": []}

for i in range(len(files)):
    # Read each stance file separately
    tmp = pd.read_csv("IBM-ArgQ-5.3kArgs/"+files[i], sep="\t", header=0)
    # Add the new instances to the full dataset
    data["text_id"].extend(list(tmp["id"]))
    # Include the stance as the first sentence of each argument, as they often constitute dependant clauses (X) of the stance,
    # e.g., [missing stance] because X; or [missing stance] 1. X, 2. X, or [missing stance] Yes, we should, because X
    texts = [stances[i]+". "+arg for arg in list(tmp["argument"])]
    data["text"].extend(texts)    
    data["rank"].extend(list(tmp["rank"]))

# After all instances are consolidated into one dataframe, save this to serve as basis for all future annotations/regression analyses.
ibm_data = pd.DataFrame(data)
ibm_data.set_index("text_id", inplace=True)
ibm_data.sort_index(inplace=True)
ibm_data.to_csv("ibm-argq_aggregated.csv", sep="\t")