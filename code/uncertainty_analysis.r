library(rethinking)
library(dplyr)
library(ggplot2)
library(jsonlite)

# Load fitted model
code_quality_model <- readRDS(
  "bayesian-prompt-optimization/artefacts/code_quality_model.rds"
)
faithfulness_model <- readRDS(
  "bayesian-prompt-optimization/artefacts/faithfulness_model.rds"
)
n_holdout <- 5
n_raters <- 7
S <- 10000

# Produce posterior predictive draws for a new question
posterior_predict_holdouts <- function(
  model,
  n_holdout = 5,
  n_raters = 7,
  S = 5000
) {
  post <- extract.samples(model)
  a_q_new <- matrix(
    rnorm(S * n_holdout, 0, post$sigma_q), nrow = S, ncol = n_holdout
  )
  a_r_new <- matrix(
    rnorm(S * n_raters, 0, post$sigma_r), nrow = S, ncol = n_raters
  )
  eta_array <- array(NA, dim = c(S, n_raters, n_holdout))
  for (q in 1:n_holdout) {
    eta_array[, , q] <- a_r_new + a_q_new[,q]
  }
  pp_means <- matrix(NA, nrow = S, ncol = n_holdout)
  for (q in 1:n_holdout) {
    newdata <- list(
      rater = 1,
      question = 1,
      ar = as.vector(eta_array[, , q]),
      aq = rep(0, length(eta_array[, , q]))
    )
    y_sim <- sim(model, data = newdata, n = S)
    y_sim_mat <- matrix(y_sim, nrow = S, ncol = n_raters, byrow = TRUE)
    pp_means[,q] <- rowMeans(y_sim_mat)
  }
  pp_summary <- apply(
    pp_means,
    2,
    function(x) quantile(x, c(0.055, 0.5, 0.945))
  )
  pp_summary <- as.data.frame(t(pp_summary))
  colnames(pp_summary) <- c("low", "median", "high")
  pp_summary$question <- paste0("Q", 1:n_holdout)
  return(pp_summary)
}

# Prepare observed prompt scores for holdouts from validation_results.json
val_path <- "bayesian-prompt-optimization/artefacts/validation_results.json"
val_json <- jsonlite::fromJSON(val_path, simplifyVector = FALSE)

code_quality <- val_json$code_quality
qid <- names(code_quality)
val_df <- do.call(rbind, lapply(qid, function(q) {
  data.frame(
    question = q,
    original = as.numeric(code_quality[[q]]$original),
    normal   = as.numeric(code_quality[[q]]$normal),
    bayesian = as.numeric(code_quality[[q]]$bayesian),
    stringsAsFactors = FALSE
  )
}))
prompt_means_code_quality <- val_df %>%
  tidyr::pivot_longer(
    cols = c(original, normal, bayesian),
    names_to = "prompt_type",
    values_to = "mean_score"
  ) %>%
  mutate(
    prompt_type = dplyr::recode(prompt_type,
      original = "raw",
      normal   = "opt_mean",
      bayesian = "opt_bayes"
    ),
    question = as.character(question)
  )

faithfulness <- val_json$faithfulness
qid <- names(faithfulness)
val_df <- do.call(rbind, lapply(qid, function(q) {
  data.frame(
    question = q,
    original = as.numeric(faithfulness[[q]]$original),
    normal   = as.numeric(faithfulness[[q]]$normal),
    bayesian = as.numeric(faithfulness[[q]]$bayesian),
    stringsAsFactors = FALSE
  )
}))
prompt_means_faithfulness <- val_df %>%
  tidyr::pivot_longer(
    cols = c(original, normal, bayesian),
    names_to = "prompt_type",
    values_to = "mean_score"
  ) %>%
  mutate(
    prompt_type = dplyr::recode(prompt_type,
      original = "raw",
      normal   = "opt_mean",
      bayesian = "opt_bayes"
    ),
    question = as.character(question)
  )

# Visualize uncertainty in posterior predictions for code quality
pp_summary_code_quality <- posterior_predict_holdouts(code_quality_model,
                                                      n_holdout = 5,
                                                      n_raters = 7,
                                                      S = 5000)
holdout_ids <- names(code_quality)
pp_summary$question <- holdout_ids
prompt_wide <- prompt_means %>%
  tidyr::pivot_wider(names_from = prompt_type, values_from = mean_score)
plot_df <- left_join(pp_summary, prompt_wide, by = "question")
uncertainty_plot_code_quality <- ggplot(plot_df, aes(x = factor(question))) +
  geom_errorbar(aes(ymin = low, ymax = high), width = 0.2, color = "grey60") +
  geom_point(aes(y = median), size = 3, color = "black") +
  geom_point(aes(y = raw), color = "red", size = 2) +
  geom_point(aes(y = opt_mean), color = "blue", size = 2) +
  geom_point(aes(y = opt_bayes), color = "green", size = 2) +
  labs(x = "Holdout question", y = "Mean human code quality score",
       title = "Posterior predictive (89% CI) vs prompt mean scores") +
  theme_minimal()

print(uncertainty_plot_code_quality)

# Visualize uncertainty in posterior predictions for faithfulness
pp_summary_faithfulness <- posterior_predict_holdouts(faithfulness_model,
                                                      n_holdout = 5,
                                                      n_raters = 7,
                                                      S = 5000)
holdout_ids <- names(faithfulness)
pp_summary$question <- holdout_ids
prompt_wide <- prompt_means %>%
  tidyr::pivot_wider(names_from = prompt_type, values_from = mean_score)
plot_df <- left_join(pp_summary, prompt_wide, by = "question")
uncertainty_plot_faithfulness <- ggplot(plot_df, aes(x = factor(question))) +
  geom_errorbar(aes(ymin = low, ymax = high), width = 0.2, color = "grey60") +
  geom_point(aes(y = median), size = 3, color = "black") +
  geom_point(aes(y = raw), color = "red", size = 2) +
  geom_point(aes(y = opt_mean), color = "blue", size = 2) +
  geom_point(aes(y = opt_bayes), color = "green", size = 2) +
  labs(x = "Holdout question", y = "Mean human faithfulness score",
       title = "Posterior predictive (89% CI) vs prompt mean scores") +
  theme_minimal()

print(uncertainty_plot_faithfulness)
