import json
import numpy as np
import pandas as pd
import pymc as pm

# Data preparation
data = json.load(open("data/scores.json"))

list_scores = []

for i, j in data.items():
    for z in j:
        stuff = {"rater": i.strip(),
               "question:": z["id"],
               "codequality": int(z["score_codequality"]),
               "faithfulness": int(z["score_faithfulness"])}
        list_scores.append(stuff)

data = pd.DataFrame(list_scores)

data.to_csv("answers_flat.csv", index=False)
data["rater"] = pd.Categorical(data["rater"]).codes
data["question_idx"] = pd.Categorical(data["question:"]).codes 
data.to_csv("answers_flat.csv", index=False)

R = data["rater"].nunique() 
Q = data["question_idx"].nunique() 

rater_index = data["rater"].values
question_index = data["question_idx"].values

# Function to fit the ordered logit model
def fit_orderlogit(y, rater_index, question_index, R, Q, draws = 2000, tune = 2000, seed = 2405):
    with pm.Model():
        #hyperprios
        sig_r = pm.Exponential("sigma_r", 1)
        sig_q = pm.Exponential("sigma_q", 1)


        #Random intercepts 
        a_rater = pm.Normal("a_rater", 0, sigma = sig_r, shape = R)
        a_question = pm.Normal("a_question", 0, sigma= sig_q, shape = Q)

        #cutpoints
        cut = pm.Normal("cutpoints", mu=0, sigma=1.5, shape=4,
                        transform=pm.distributions.transforms.ordered, # type: ignore
                        initval=np.array([-2, -1, 1, 2])) 

        eta = a_rater[rater_index] + a_question[question_index]
        
        pm.OrderedLogistic("y", eta =eta, cutpoints = cut, observed = y.astype(int) -1)

        data_return = pm.sample(
        draws=draws, tune=tune, chains=4, cores=4,
        target_accept=0.98, random_seed=seed) 

        return data_return
        
data_codecorrectness = fit_orderlogit(data["codequality"].values, rater_index, question_index,
                                      R, Q, seed=1) 
data_faithfulness = fit_orderlogit(data["faithfulness"].values, rater_index, question_index,
                                     R,Q, seed=1)

# Function to compute the scores
def scores(data, R, Q, K=5):
    a_rater = data.posterior["a_rater"].stack(sample=("chain", "draw")).values
    a_question = data.posterior["a_question"].stack(sample=("chain", "draw")).values
    cut = data.posterior["cutpoints"].stack(sample=("chain", "draw")).values

    cut = np.vstack([-np.inf * np.ones((1, cut.shape[1])), cut, np.inf * np.ones((1, cut.shape[1]))])

    smoothed_scores = []
    for i in range(Q):
        eta = a_rater + a_question[i]
        probabilities = []

        for j in range(K):
            top = 1 / (1 + np.exp(-(cut[j+1] - eta)))
            bot = 1 / (1 + np.exp(-(cut[j] - eta)))
            prob = top - bot
            probabilities.append(prob)

        probs = np.array(probabilities)
        score = (np.arange(1, K+1)[:, None, None] * probs).sum(axis=0)
        smoothed_scores.append(score.mean())
    return np.array(smoothed_scores)

post_cq = scores(data_codecorrectness, R, Q, K=5)
post_fa = scores(data_faithfulness, R, Q, K=5)

map = (data[["question:","question_idx"]].drop_duplicates().sort_values("question_idx"))
map2 = map["question:"].tolist()
mean = (data.groupby("question:")[["codequality","faithfulness"]].mean().reindex(map2))

output = pd.DataFrame({ "question:": map2, "smoothed_quality" :post_cq, "smoothed_faithfulness": post_fa}).merge(mean, on = "question:")
output.to_csv("artefacts/bayesian_posterior_means.csv")
