"""
Auxiliary functions for stepwise regression, 
including the stepwise selection process and significance testing between two models.
"""
# Regression model
from statsmodels.formula.api import ols, logit
import pandas as pd

# Significance testing for logistic regression
def calculate_nested_f_statistic(small_model, big_model):
    """Given two fitted GLMs, the larger of which contains the parameter space of the smaller, 
    return the F Stat and P value corresponding to the larger model adding explanatory power"""
    addtl_params = big_model.df_model - small_model.df_model
    print("params_big:", big_model.df_model, "params_small:", small_model.df_model, "scale_big:", big_model.scale)
    f_stat = (sum(small_model.resid_dev**2) - sum(big_model.resid_dev**2)) / (addtl_params * big_model.scale) # resid_dev statt deviance?
    df_numerator = addtl_params
    # use fitted values to obtain n_obs from model object:
    df_denom = (big_model.fittedvalues.shape[0] - big_model.df_model)
    p_value = stats.f.sf(f_stat, df_numerator, df_denom)
    return (f_stat, p_value)

############################################
# Stepwise regression without interaction
############################################
def forward_selection(data: pd.DataFrame, dv: str, model_type: str):
    """Given a dataset and DV, incrementally select IVs to polulate a model of model_type
    :param data: DataFrame containing the data to analyse with columns for all IVs and DV
    :param dv: dependent variable (DV) of the regression
    :param model_type: string "logit" or "ols" to indicate the type of regression
    """
    # All possible IVs
    remaining = set(data.columns)
    # Remove dependent variable
    remaining.remove(dv)
    # Stepwise selected IVs and the corresponding r^2 and significance values
    selected = []
    r_vals = []
    sign = []

    current_score, best_new_score = float('0'), float('0')
    # Stop selection process if all variables are incorporated or do not improve the model anymore
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        # Compare the r^2 value of regression with all selected IVs and current candidates
        for candidate in remaining:
            formula = "{} ~ {}".format(dv,
                                        ' + '.join(selected + [candidate]))
            # Build regression model (logistic or linear)
            if model_type=="logit":
                score = logit(formula, data).fit().prsquared
            else:
                score = ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort(reverse=True)
        # Get candidate with highest r^2 score
        best_new_score, best_candidate = scores_with_candidates.pop(0)
        if current_score < best_new_score:
            if len(selected):
                old_model = current_model
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            # Calculate r^2 value of the new model including the best IV candidate
            if model_type=="logit":
                current_model = logit(f"{dv} ~ {' + '.join(selected)}", data).fit()
                r_vals.append(current_model.prsquared)
            else:
                current_model = ols(f"{response} ~ {' + '.join(selected)}", data).fit()
                r_vals.append(current_model.rsquared_adj)
            # Calculate the significance of the improvement between the models without and with the IV candidate
            if len(sign):
                if model_type="logit":
                    table = calculate_nested_f_statistic(old_model, current_model)
                    sign.append(table[1])
                else:
                    table = anova_lm(old_model, current_model)
                    sign.append(table["Pr(>F)"][1])
            else:
                sign.append("x")
            current_score = best_new_score
    # Build the full stepwise model with all selected candidates
    formula = "{} ~ {}".format(dv, ' + '.join(selected))
    if model_type=="logit":
        model = logit(formula, data).fit()
    else:
        model = ols(formula, data).fit()
    # Return the model and stepwise calculation results for all IVs
    return {"model": model, "remaining": remaining, "r2": r_vals, "significance":  sign}


############################################
# Stepwise regression with 2-way interaction
############################################
def poly_forward_selection(data: pd.DataFrame, dv: str, model_type: str):
    """Given a dataset and DV, incrementally select IVs to polulate a model of model_type
    :param data: DataFrame containing the data to analyse with columns for all IVs and DV
    :param dv: dependent variable (DV) of the regression
    :param model_type: string "logit" or "ols" to indicate the type of regression
    """
    # All possible individual IVs
    remaining = set(data.columns)
    remaining.remove(dv)
    # Find all possible 2-way interactions of IVs
    interactions = set()
    for i in remaining:
        for j in remaining:
            if i != j and j+":"+i not in remaining:
                # Save interactions according to formula convention (":")
                interactions.add(i+":"+j)
    remaining.update(interactions)
    # Stepwise selected IVs and the corresponding r^2 and significance values
    selected = []
    r_vals = []
    sign = []

    current_score, best_new_score = float('inf'), float('inf')
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        # Calculate the AIC of models including the selected and all candidate IVs
        for candidate in remaining:
            print(candidate)
            formula = "{} ~ {}".format(dv,
                                        ' + '.join(selected + [candidate]))
            if model_type=="logit":            
                score = logit(formula, data).fit().aic
            else:
                score = ols(formula, data).fit().aic
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        # Select candidate IV with best AIC score
        best_new_score, best_candidate = scores_with_candidates.pop(0)
        if current_score > best_new_score:
            # Calculate the r^2 and (for all but the first IV) significance of the new model over the smaller model
            if len(selected):
                old_model = current_model
                if model_type=="logit":
                    current_model = logit(f"{dv} ~ {' + '.join(selected + [best_candidate])}", data).fit()
                    table = calculate_nested_f_statistic(old_model, current_model)
                    sign.append(table[1])
                    r_vals.append(current_model.prsquared)
                else:
                    current_model = ols(f"{response} ~ {' + '.join(selected+[best_candidate])}", data).fit()
                    table = anova_lm(old_model, current_model)
                    sign.append(table["Pr(>F)"][1])
                    r_vals.append(current_model.rsquared_adj)
                # If the model improvement is significant, add the best IV candidate to selected IVs
                if sign[-1] < 0.05:
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
            else:
                current_model = logit(f"{dv} ~ {best_candidate}", data).fit()
                sign.append("x")
                r_vals.append(current_model.prsquared)
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
    # Populate the full model and return it with all stepwise calculation results
    formula = "{} ~ {}".format(dv, ' + '.join(selected))
    if model_type=="logit":
        model = logit(formula, data).fit()
    else:
        model = ols(formula, data).fit()
    return {"model": model, "remaining": remaining, "r2": r_vals, "significance":  sign}

############################################
# Plot individual regression effects
############################################
def plot_iv(iv: str, dv: str, corpus: str, num: (int, int)):
    if num[1] < 6:
        color = sns.color_palette("mako").as_hex()[num[0]+1]
    elif num[1] > 6:
        color = sns.color_palette("mako", n_colors=num[1]).as_hex()[num[0]]
    else:
        color = sns.color_palette("mako").as_hex()[num[0]]
    output = sns.regplot(data=cmv, x=iv, y=dv, scatter=False, color=color, logistic=True)
    output.set(xlabel=re_feats(cmv[iv].name))
    plt.savefig("img/"+corpus+"_"+iv+".svg", format="svg")
    plt.show()

############################################
# Convert column names into feature names
############################################
def feature(x):
    #"emotion", "train", "anno", "split"
    feat = ""
    if x[2] == "prob":
        feat += "prob_"
    feat += x[0] + "_" + x[1] 
    if x[3] == "agg":
        feat += "_" + x[3]
    return feat

def re_feats(x):
    out = ""
    if ":" in x:
        x1, x2 = x.split(":")
        out = re_feats(x1) + " : " + re_feats(x2)
        return out
    x = x.split("_")
    if "guilt" in x:
        return "guilt/shame"
    elif "storytelling" in x:
        return "storytelling"
    elif "hedge" in x:
        if "avg" in x:
            return "avg(all hedge)"
        elif "abs" not in x:
            return f"avg({x[0]} hedge)"
        elif "global" in x:
            return "all hedge"
        else:
            return x[0] + " hedge"
    else:
        if x[0] == "prob":
            return x[1]
        else: return x[0]