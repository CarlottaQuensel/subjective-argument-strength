import os, pandas
import stanza
stanza.download("en")

pipeline = stanza.Pipeline(
    lang="en", processors="tokenize,mwt,pos,lemma,depparse")

hedge_verbs = ["suggest", "believe", "appear", "indicate", "assume", "seem", "consider", "doubt", "estimate", "expect", "feel",
               "guess", "find", "read", "imagine", "speculate", "suppose", "think", "understand", "imply", "presume", "suspect",
               "postulate", "reckon", "infer", "hope", "tend", "hear"]
hedge_mod = ["may", "might", "maybe", "shall",
             "should", "can", "could", "would", "ought"]
hedge_adv = ["rather", "slightly", "barely", "strictly", "presumably", "fairly", "theoretically", "basically", "relatively",
             "possibly", "preferably", "often", "slenderly", "scantily", "decidedly", "arguably", "seemingly", "occasionally",
             "partially", "partly", "practically", "roughly", "virtually", "allegedly", "pretty"]
hedge_adj = ["presumable", "possible", "probably", "likely", "apparent", "probable", "improbable", "around", "general", "seldom", "about",
             "partial", "unlikely", "rarely", "improbably", "unclearly", "unsure", "sure", "chance", "unclear", "fair", "mainly", "rough"]
hedge_app = ["usually", "approximately", "generally", "frequently",
             "normally", "largely", "apparently", "necessarily", "like"]
hedge_misc = ["much", "basic", "most", "mostly", "several", "perhaps",
              "etc", "impression", "sorta", "kinda", "imo", "imho", "afaik"]

ngram_hedges = {"to": [("according", "to")], "mean": [("I", "mean")], "say": [("I", "would", "say"), ("I'd", "say")], "mind": [("in", "my", "mind")],
                "opinion": [("in", "my", "opinion"), ("in", "my", "humble", "opinion"), ("in", "my", "honest", "opinion")],
                "understanding": [("my", "understanding")], "view": [("my", "view")],
                "like": [("look", "like"), ("looks", "like"), ("sound", "like"), ("sounds", "like")],
                "thinking": [("my", "thinking")], "bit": [("a", "bit")], "bunch": [("a", "bunch"), ("a", "whole", "bunch")],
                "couple": [("a", "couple")], "few": [("a", "few")], "little": [("a", "little")], "other": [("something", "or", "other")],
                "others": [("among", "others")], "that": [("and", "all", "that")], "forth": [("and", "so", "forth")],
                "on": [("and", "so", "on")], "suchlike": [("and", "suchlike")], "least": [("at", "least")], "cetera": [("et", "cetera")],
                "part": [("for", "the", "most", "part"), ("in", "part")], "way": [("in", "a", "way"), ("in", "some", "way")],
                "ways": [(" in ", "some", "ways")], "of": [("kind", "of"), ("sort", "of")], "less": [("more", "or", "less")],
                "much": [("pretty", "much")], "extent": [("to", "a", "certain", "extent"), ("to", "some", "extent"), ("to", "an", "extent")],
                "know": [("as", "far", "as", "I", "know")]
                }

hedges = set(hedge_verbs)
hedges.update(hedge_mod)
hedges.update(hedge_adv)
hedges.update(hedge_adj)
hedges.update(hedge_app)

hedge_rules = {"feel", "suggest", "believe", "consider", "doubt", "hope", "guess", "imagine", "think", "appear", "find", "assume", "suppose", "tend",
               "likely", "should", "rather", "about", "around", "unlikely", "fairly", "general", "roughly", "pretty", "like", "partial", "fair", "impression"}


booster = {"clearly", "obviously", "certainly", "show", "actually", "absolutely", "always", "apparently", "assuredly", "categorically", "compelling", "completely",
           "comprehensively", "conclusively", "confirmed", "confirmation", "considerable", "considerably", "consistently", "conspicuously", "constantly", "convincingly",
           "corroborate", "corroboration", "credible", "credibly", "crucially", "decisively", "definitely", "definitively", "demonstrate", "deservedly", "distinctively",
           "doubtlessly", "enhanced", "entirely", "especially", "essentially", "establish", "evidently", "exceptionally", "exhaustively", "extensively", "extraordinary",
           "extremely", "firmly", "forcefully", "fully", "strikingly", "successfully", "fundamentally", "genuinely", "highly", "impossible", "impressive", "impressively",
           "incontrovertible", "indispensable", "indispensably", "inevitable", "inevitably", "know", "manifestly", "markedly", "meaningfully", "necessarily", "never",
           "notable", "notably", "noteworthy", "noticeable", "noticeably", "outstanding", "particularly", "perfectly", "persuasively", "plainly", "powerful", "precise",
           "precisely", "profoundly", "prominently", "proof", "prove", "quite", "radically", "really", "reliably", "remarkable", "remarkably", "rigorously", "safe",
           "safely", "secure", "securely", "self-evident", "sizably", "superior", "sure", "surely", "thoroughly", "totally", "truly", "unambiguously", "unarguably",
           "unavoidable", "unavoidably", "undeniable", "undeniably", "undoubtedly", "unequivocally", "uniquely", "unlimited", "unmistakable", "unmistakably", "unprecedented",
           "unquestionably", "uphold", "vastly", "vitally", "prove", "honestly", "mostly", "largely", "mainly"}


def hedge_detection(document: str):
    # Use Stanford CoreNLP pipeline to tokenize arguments, split into sentences,
    # tag with POS and then parse for dependencies
    doc = pipeline(document)
    # Employ multiple hedging counters for different hedging feature variants
    avg_hedge = 0
    global_abs_hedge = 0
    for i, sent in enumerate(doc.sentences):
        num_hedge = 0
        for word in sent.words:
            if word.lemma in hedges and true_hedge(word):
                num_hedge += 1

            elif word.text in ngram_hedges and true_hedge(word):
                num_hedge += 1

            elif word.lemma in booster:
                if word.id > 1 and sent.words[word.id-2].lemma in {"not", "no"}:
                    num_hedge += 1
                else:
                    for dep in sent.words:
                        if dep.head == word.id and dep.lemma == "not":
                            num_hedge += 1
                            break

        # Save the number of hedges in the argument text's first sentence separately
        if i == 0:
            first_abs_hedge = num_hedge
            first_hedge = first_abs_hedge/len(sent.words)
        # Add up the number of hedges over all sentences in the argument text
        global_abs_hedge += num_hedge
        avg_hedge += (num_hedge/len(sent.words))

    # The number of hedges in the last sentence is the variable when the loop finished
    final_abs_hedge = num_hedge
    final_hedge = final_abs_hedge/len(sent.words)
    # global_abs_hedge /= len(doc.sentences)
    avg_hedge /= len(doc.sentences)

    return {"first_hedge": first_hedge, "first_abs_hedge": first_abs_hedge, "final_hedge": final_hedge, "final_abs_hedge": final_abs_hedge, "global_abs_hedge": global_abs_hedge, "avg_hedge": avg_hedge}


def true_hedge(word):
    sent = word.sent.words
    if word.lemma in hedges and word.lemma not in hedge_rules:
        return True

    # Phrasal hedges are matched as ngrams of unlemmatized tokens
    if word.text in ngram_hedges and word.id >= len(max(ngram_hedges[word.text])):
        for phrase in ngram_hedges[word.text]:
            if word.id >= len(phrase):
                ngram = tuple(
                    [sent[word.id+i].text for i in range(-len(phrase), 0, 1)])
                if ngram == phrase:
                    return True

    # Hedge Term: feel, suggest, believe, consider, doubt, guess, hope, imagine
    # Rule: If token t is (i) a root word, (ii) has the part-of-speech VB* and (iii) has an nsubj (nominal subject) dependency
    # with the dependent token being a first person pronoun (i, we), t is a hedge, otherwise, it is a non-hedge.
    # Hedge: I don’t think it’s been a failure, but I hope that I’m on the right track.
    # Non-hedge: I’m still living with it, but without hope that I would find anyone.
    # and word.head == 0:
    if word.lemma in {"feel", "suggest", "believe", "consider", "doubt", "guess", "hope", "imagine"} and word.pos == "VERB":
        for dep in word.sent.words:
            if dep.head == word.id and dep.deprel == "nsubj" and "Person=1" in str(dep.feats):
                return True

    # Hedge Term: think
    # Rule: If token t is followed by a token with part-of-speech IN, t is a non-hedge, otherwise, hedge.
    # Hedge: I think it’s difficult to make generalizations about this kind of relationships.
    # Non-hedge: Even if it’s difficult, I always say, think about your children.
    if word.lemma == "think":
        if word.id == len(sent):
            return True
        elif sent[word.id].xpos != "IN":
            return True

    # Hedge Term: appear, assume, find
    # Rule: If token t has a ccomp (clausal complement) dependent, t is a hedge, otherwise, non-hedge.
    # Hedge: I assume they were responsible for this. It appears that there were people who wanted to attack the school.
    # Non-hedge: They have assumed the role of parents and are doing their best to fulfill it. I had to do all I could to appear like an old lady.
    if word.lemma in {"assume", "find", "appear"}:
        for dep in sent:
            if dep.head == word.id and dep.deprel in {"ccomp", "xcomp"}:
                return True

    # Hedge Term: suppose
    # Rule: If token t has an xcomp (open clausal complement) dependent d and d has a mark dependent to, t is a non-hedge, otherwise, it is a hedge.
    # Hedge: I suppose he was present during the discussion.
    # Non-hedge: I could see that they were skewing the real truth, the one they are supposed to tell me.
    if word.lemma == "suppose":
        for dep in sent:
            if dep.head == word.id and dep.deprel == "xcomp":
                for dep2 in sent:
                    if dep2.head == dep.id and dep.deprel == "mark":
                        return False
        return True

    # Hedge Term: tend
    # Rule: If token t has an xcomp (open clausal complement) dependent, t is a hedge, otherwise, it is a non-hedge.
    # Hedge: We tend to never forget.
    # Non-hedge: All political institutions tended toward despotism.
    if word.lemma == "tend":
        for dep in sent:
            if dep.head == word.id and dep.deprel == "xcomp":
                return True

    # Hedge Term: likely
    # Rule: If token t has relation amod with its head h and h has part of speech N*, t is a non-hedge, otherwise, it is a hedge.
    # Hedge: They will likely visit us in the future.
    # Non-hedge: He is a fine, likely young man.
    if word.lemma == "likely":
        if word.deprel == "amod" and sent[word.head-1].pos == "NOUN":
            return False
        return True

    # Hedge Term: should
    # Rule: If token t has relation aux with its head h and h has dependent have, t is a non-hedge, otherwise, it is a hedge.
    # Hedge: That’s precisely the message that should be sent to people who label others, isn’t it?
    # Non-hedge: They should have been more careful.
    if word.lemma == "should":
        if word.deprel == "aux":
            for dep in sent:
                if dep.head == word.head and dep.lemma == "head":
                    return False
        return True

    # Hedge Term: rather
    # Rule: If token t is followed by token than, t is a non-hedge, otherwise, it is a hedge.
    # Hedge: I never had the opportunity to go, but i know people who have gone and who came back rather depressed.
    # Non-hedge: He would have protected his flock rather than shoot at them.
    if word.lemma == "rather":
        if sent[word.id].lemma == "than":
            return False
        return True

    # Hedge Term: about, around
    # Rule: If token t has part-of-speech IN, t is non-hedge. Otherwise, hedge.
    # Hedge: There are about 10 million packages in transit right now.
    # Non-hedge: We need to talk about Mark.
    if word.lemma in {"about", "around"}:
        if word.xpos != "IN":
            return False
        return True

    # Hedge Term: unlikely
    # Rule: If the token has ADJ POS and is the root word or a complement , it is a hedge.
    # Hedge: It is unlikely that we win the game. I find him unlikely to be lying.
    # Non-hedge: He's an unlikely friend.
    if word.lemma == "unlikely" and (word.head == 0 or word.deprel == "xcomp"):
        return True

    # Hedge Term: fairly
    # Rule: ADV with ADJ head not VV head
    # Hedge:
    # Non-hedge:
    if word.lemma == "fairly" and word.pos == "ADV" and sent[word.head-1].pos == "ADJ":
        return True

    # Hedge Term: general
    # Rule: not NN
    # Hedge:
    # Non-hedge:
    if word.lemma == "general" and word.pos != "NOUN":
        return True

    # Hedge Term: roughly
    # Rule: Token has non-verb head or head is "speak", "say"
    # Hedge:
    # Non-hedge:
    if word.lemma == "roughly":
        if sent[word.head-1].pos != "VERB":
            return True
        elif sent[word.head-1].lemma in {"speak", "say"}:
            return True

    # Hedge Term: pretty
    # Rule: Token has POS ADV
    # Hedge: I am pretty certain that I'm right. He is a pretty knowledgeable person.
    # Non-hedge: This is a pretty cat.
    if word.lemma == "pretty" and word.pos == "ADV":
        return True

    # Hedge Term: like
    # Rule: POS SCONJ or INTJ
    # Hedge: There are like 2.5 mio miles inbetween.
    # Non-hedge: He looks like someone else.
    if word.lemma == "like" and word.pos in {"SCONJ", "INTJ"}:
        return True

    # Hedge Term: partial
    # Rule: If token is not head of sentence (0)
    # Hedge: There has been a partial withdrawal from enemy territory.
    # Non-hedge: I'm not partial to fish.
    if word.lemma == "partial" and word.head != 0:
        return True

    # Hedge Term: fair
    # Rule: not NN, amod deprel or head of lemma "say"
    # Hedge:
    # Non-hedge:
    if word.lemma == "fair" and word.pos != "NOUN":
        if word.head == 0:
            return True
        if sent[word.head-1].pos != "NOUN":
            return True

    # Hedge Term: impression
    # Rule: If token has a dependency nsubj or possessive pronoun in first person, it is a hedge.
    # Hedge: my impression / I get the impression?
    # Non-hedge:
    if word.lemma == "impression":
        for dep in word.sent.words:
            if dep.head == word.id and dep.deprel in {"nsubj", "nmod:poss"} and "Person=1" in str(dep.feats):
                return True
        if word.deprel == "obj":
            for dep in word.sent.words:
                if dep.head == word.head and dep.deprel == "nsubj" and "Person=1" in str(dep.feats):
                    return True

    return False


if __name__ == '__main__':
    dest_folder = "results/hedging"
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Load IBM argument data
    ibm = pandas.read_csv("data/arguments/ibm-argq_aggregated.csv", sep="\t")#C:/Users/HP Envy/Documents/Uni/Master/22_winter/what-makes-persuasiveness/
    ibm.set_index(("text_id"), inplace=True)
    print("Loaded IBM data...")
    # Detect hedges for all instances and add as new dataframe columns
    hedges = [hedge_detection(t) for t in ibm["text"]]
    print("Calculated hedge ratios for IBM...")
    ibm_hedges = {"first_hedge": [], "first_abs_hedge": [], "final_hedge": [], "final_abs_hedge": [], "global_abs_hedge": [], "avg_hedge": []}
    for instance in hedges:
        for h in ibm_hedges:
            ibm_hedges[h].append(instance[h])
    for h in ibm_hedges:
        ibm[h] = ibm_hedges[h]
    # Save the resulting table as a csv
    ibm.to_csv(dest_folder+"/ibm_with_hedging.csv", sep="\t")
    print("Saved IBM with hedges...")

    # Load CMV argument data
    cmv = pandas.read_csv("data/arguments/CMV_Cornell_2016.csv", sep="\t")#C:/Users/HP Envy/Documents/Uni/Master/22_winter/what-makes-persuasiveness/
    cmv.set_index(("text_id"), inplace=True)
    print("Loaded CMV data...")
    # Detect hedges for all instances
    hedges = [hedge_detection(t) for t in cmv["text"]]
    print("Calculated hedge ratios for CMV...")
    cmv_hedges = {"first_hedge": [], "first_abs_hedge": [], "final_hedge": [], "final_abs_hedge": [], "global_abs_hedge": [], "avg_hedge": []}
    for instance in hedges:
        for h in cmv_hedges:
            cmv_hedges[h].append(instance[h])
    # and add the results as new dataframe columns
    for h in cmv_hedges:
        cmv[h] = cmv_hedges[h]
    # Save the resulting table as a csv
    cmv.to_csv(dest_folder+"/cmv_with_hedging.csv", sep="\t")
    print("Saved CMV with hedges...")