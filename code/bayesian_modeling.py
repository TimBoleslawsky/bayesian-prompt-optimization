# bayesian_ordered_logit.py

import json
import numpy as np
import pandas as pd
import pymc as pm

data = json.load(open("answers.json"))

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



#instead of "kesa", "tim" etc I want to have each rater as a categorical number
data["rater"] = pd.Categorical(data["rater"]).codes
#Q.1 = 0 which is how pyMC wants it later

data["question_idx"] = pd.Categorical(data["question:"]).codes #from like 001 to 1 instead

data.to_csv("answers_flat.csv", index=False)


R = data["rater"].nunique() # we know its 7 raters but for possible changes 
Q = data["question_idx"].nunique() #and 20 questions 

rater_index = data["rater"].values
question_index = data["question_idx"].values

#print(rater_index) #they are now 0 to 6 
#print(question_index) # they are 0 to 19
#print(Q)
#print(R)

"""""""""

The theory how I understand it (pieces we need):

y_i ~ orderlogit(eta_i, c) which is ordinal 1,2,3,4,5 and c is the cutting points (bc its discree)
(If we run it in R i think we dont need cutpoints as stan automatically enforces this?)

eta_i = a_rating[rater_i] + a_question[question_i] <- Linnear predictior

Where a_rating ~ Normal(0, sigma_rating) and a_question ~Normal(0, sigma_question)

And where sigma follows = Sigma_rating ~ Exponential(1) and sigma_question ~Exp(1)

cutoff(c) = sorted(c_unordered), where c_k are the 4 cutpoints of the 5 categories.
Like if we have 1,2,3,4,5 you see that it is really just 4 cutoffs.

We want to model the posterior using monte carlo markow chains as:
prior(params) * liklehood(data|params) -> p(params|data)


"""""""""
def fit_orderlogit(y, rater_index, question_index, R, Q, draws = 2000, tune = 2000, seed = 2405): #when we did it in R they use half for warmup so its draws + tune here
    with pm.Model() as m:
        #hyperprios
        sig_r = pm.Exponential("sigma_r", 1)
        sig_q = pm.Exponential("sigma_q", 1)


        #Random intercepts 
        a_rater = pm.Normal("a_rater", 0, sigma = sig_r, shape = R)
        a_question = pm.Normal("a_question", 0, sigma= sig_q, shape = Q)

        #cutpoints


        cut = pm.Normal("cutpoints", mu=0, sigma =1.5, shape = 4,
                              transform=pm.distributions.transforms.ordered,
                              initval=np.array([-2, -1, 1, 2]))# doesnt matter what values, just so we get the MC chains to start from a reasonebla place centered around 0

        eta = a_rater[rater_index] + a_question[question_index]
        
        pm.OrderedLogistic("y", eta =eta, cutpoints = cut, observed = y.astype(int) -1)

        data_return = pm.sample(
        draws=draws, tune=tune, chains=4, cores=4,
        target_accept=0.98, random_seed=seed) 

        return data_return
        



#Fit both the code quality and faitfulness using out function

data_codecorectness = fit_orderlogit(data["codequality"].values, rater_index, question_index,
                                     R,Q, seed = 1) 

data_faithfulness = fit_orderlogit(data["faithfulness"].values, rater_index, question_index,
                                     R,Q, seed = 1)


"""""""""

What do we want to get as output to make sure it worked/converged?

1)Divergence = 0 -> means we dont really have any numerical problems during sampling.
If we have divergence this basically means the sampler has a hard time expole parts of the posterior


2) r_hat = close to 1.00, If we have r_hat = 1.00 this means all chains agree.
What i understand i values up till 1.01 is usuallt "fine"

3) Dont want an effective sample sice (meaning ess_buld and ess tail should not be small).
Like a couple of hundred or more for each parameters

Deal with poor results:
- Increase the nr of tuning samples/iterations.
- Increase the accaptence rate
- We could also add more draws.

"""""""""

#Next part


"""""""""
Next part is to compute the posterior for each question. We save it in a seperate file/output.
The output should look like:


Basically what we have left do do is:

1) Each model (or score - meaning code quality and fithfulness), we take the posterios "draws"
2) For every question we average over all raters and all posterior draws to "get" the new 1-5 score, this 
is like " E["new score"] = sum_i->n {k=1,2,3,4,5} k * p(Y=y | n, cutpoints)
And here "n" is the rater bias + question effect(bias), and cutpoints are the "Learned" thesholds
3) Put the beyesian smoothes scores into some table aswell as the "Original" score so we can plot/use everything. 
We also ofcourse save it into a csv so we can work with it later.  We can have the csv as:

- Beysian smoothed corectness | betsian smothed faitfulness | Mean (real) correctness | Mean (real) faitfulness 

This csv file we can then have as the input file for when we train or evaluate the LLms in next step 
"""""""""


def scores(data, R,Q, K =5 ):

#Shape (chains, draws, R/Q/k-1)    
    a_rater = data.posterior["a_rater"].stack(sample=("chain", "draw")).values
    a_question = data.posterior["a_question"].stack(sample=("chain", "draw")).values
    cut = data.posterior["cutpoints"].stack(sample=("chain", "draw")).values
    
    cut = np.vstack([-np.inf * np.ones((1, cut.shape[1])),cut,np.inf * np.ones((1, cut.shape[1]))])

    smothed_scores =[]
    for i in range(Q):
        eta = a_rater + a_question[i]
        probabilities = []

        for j in range (K):
            top = 1/ (1 + np.exp(-(cut[j+1] - eta)))
            bot = 1 / (1+ np.exp(-(cut[j] -eta)))
            prob = top - bot
            probabilities.append(prob)

        probs = np.array(probabilities)
        score = (np.arange(1,K+1)[:, None, None] * probs).sum(axis=0)
        smothed_scores.append(score.mean())
    return np.array(smothed_scores)



#Plug in our actuall scores to get the beysian smoothed version

post_cq = scores(data_codecorectness, R, Q, K=5)
post_fa = scores(data_faithfulness,  R, Q, K=5)

#map it back to our table
map = (data[["question:","question_idx"]].drop_duplicates().sort_values("question_idx"))
map2 = map["question:"].tolist()
mean = (data.groupby("question:")[["codequality","faithfulness"]].mean().reindex(map2))

output = pd.DataFrame({ "question:": map2, "smoothed_quality" :post_cq, "smoothed_faithfulness": post_fa}).merge(mean, on = "question:")

output.to_csv("bayesian_posterior_means.csv")
