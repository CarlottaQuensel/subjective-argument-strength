import pandas as pd
import os
# To train the emotion classifier used to annotate the argument data, the crowd-enVENT corpus 
# serves as training data:
# Enrica Troiano, Laura Oberländer, and Roman Klinger. 2023. Dimensional Modeling of Emotions in 
# Text with Appraisal Theories: Corpus Creation, Annotation Reliability, and Prediction. 
# Computational Linguistics, 49(1):1-72.
# Obtained from https://www.romanklinger.de/data-sets/crowd-enVent2023.zip

# generated_text  --  unedited crowd generated event descriptions
# hidden_emo_text --  generated text with masked emotion words (words with same lemma as emotion labels)
text_used = ("generated_text", "hidden_emo_text")
if not os.path.exists(text_used):
    os.mkdir(text_used)

# Loading the emotion data (crowd-enVENT) and extracting the relevant layers: instance texts and emotion annotation
corpus = pd.read_table("crowd-enVent_generation.tsv", index_col="text_id")
cols = list(corpus.columns)
cols.remove('emotion')
cols.remove(text_used)
corpus = corpus.drop(columns=cols)
# Separating out the emotion annotation into binary annotations for each emotion
binary_emos = pd.get_dummies(corpus['emotion'], dtype=int)
data = pd.concat((corpus, binary_emos), axis=1)
data = data.drop(columns=['emotion'])
# Merging emotions guilt and shame as according to the dataset's paper (Troiano et al., 2023)
# they belong together (and have only 275 instead of 550 instances each)
data["guilt_shame"] = data["guilt"]+data['shame']
data = data.drop(columns=['guilt', 'shame'])

emotions = ['anger', 'boredom', 'disgust', 'fear', 'guilt_shame', 'joy',
            'no-emotion', 'pride', 'relief', 'sadness', 'surprise', 'trust']

for emo in emotions:
    if emo == "no-emotion":
        continue
    # For each emotion split the 550 positive instances from the rest of the dataset
    emo_instances = data.loc[data[emo] == 1]
    inverse_instances = data.loc[data[emo] == 0]
    # Each emotion has 550 instances, which results in only 8% of positive training examples
    # using the whole dataset, or a dataset size of 1100 for a completely balanced set.
    # As a compromise, the rest of the dataset is downsampled to only 3x the number of positives, 
    # i.e., 1650 negative instances and 2200 instances overall
    inverse_instances = inverse_instances.sample(n=1650, random_state=42)

    emo_dataset = pd.concat([emo_instances, inverse_instances], axis=0)
    cols = list(emo_dataset.columns)
    # Remove all columns except for the instance text and current emotion label
    cols.remove(emo)
    cols.remove(text_used)
    emo_dataset.drop(columns=cols, inplace=True)
    emo_dataset.rename(columns={emo:"label"}, inplace=True)
    # Save both text variants:
    # a) original generated texts
    if not os.path.exists("unmasked"):
        os.mkdir("unmasked")
    unmasked = emo_dataset[[text_used[0], "label"]]
    unmasked = unmasked.rename(columns={text_used[0]: 'text'}, inplace=True)
    unmasked.to_csv(f"unmasked/crowd-enVent_{emo}.csv", sep="\t")
    # b) generated texts with masked emotion words
    if not os.path.exists("masked"):
        os.mkdir("masked")
    masked = emo_dataset[[text_used[1], "label"]]
    masked = masked.rename(columns={text_used[1]: 'text'}, inplace=True)
    masked.to_csv(f"masked/crowd-enVent_{emo}.csv", sep="\t")

