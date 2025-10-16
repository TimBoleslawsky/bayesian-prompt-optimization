library(rethinking)
library(jsonlite)

set.seed(2405)

# --- Load & Tidy Data ------------------------------------------------------

raw_json <- jsonlite::fromJSON(
  "bayesian-prompt-optimization/data/scores.json"
)

make_long <- function(lst) {
  raters <- names(lst)
  out <- lapply(raters, function(r) {
    df <- lst[[r]]
    if (is.null(df) || length(df) == 0) return(NULL)
    df$rater <- trimws(r)
    df
  })
  do.call(rbind, out)
}

dat_long <- make_long(raw_json)
dat_long$id_int <- as.integer(dat_long$id)
dat_long$question_id <- as.integer(
  factor(dat_long$id_int, levels = sort(unique(dat_long$id_int)))
)
dat_long$rater_id    <- as.integer(
  factor(dat_long$rater,   levels = sort(unique(dat_long$rater)))
)
print("Data preview:")
print(head(dat_long))

R <- length(unique(dat_long$rater_id))
Q <- length(unique(dat_long$question_id))
N <- nrow(dat_long)

# Preserve original question IDs (from the raw JSON), in sorted order
orig_ids <- sort(unique(dat_long$id_int))

# --- Helper: Build data list for a given outcome ---------------------------

build_data_list <- function(score_vec) {
  list(
    N = N,
    score = as.integer(score_vec),  # integers 1..K
    R = R,
    Q = Q,
    rater = dat_long$rater_id,
    question = dat_long$question_id
  )
}

# --- Model Specification Template (Ordered Logistic) ----------------------

# We define the score to be drawn from an ordered logistic distribution.
# Cutpoints c[1:4] (since 5 categories) are ordered.
# We assume that eta is a linear combination of varying intercepts
# for raters and questions: eta_i = ar[rater] + aq[question].

model_alist <- alist(
  score ~ dordlogit(eta, c),
  eta <- ar[rater] + aq[question],
  ar[rater] ~ dnorm(0, sigma_r),
  aq[question] ~ dnorm(0, sigma_q),
  c ~ dnorm(0, 1.5),
  sigma_r ~ dexp(1),
  sigma_q ~ dexp(1)
)

# --- Fit Code Quality Model -------------------------------------------------

dl_code <- build_data_list(dat_long$score_codequality)

cat("Fitting code quality model...\n")
code_quality_model <- ulam(
  model_alist,
  data = dl_code,
  chains = 4,
  cores = 4,
  iter = 4000
)

cat("Code quality model fit complete.\n")

# --- Fit Faithfulness Model -------------------------------------------------

dl_faith <- build_data_list(dat_long$score_faithfulness)

cat("Fitting faithfulness model...\n")
faithfulness_model <- ulam(
  model_alist,
  data = dl_faith,
  chains = 4,
  cores = 4,
  iter = 4000
)

cat("Faithfulness model fit complete.\n")

# --- Save Models and Means for each Question ------------------------

cat("Saving models and posterior means...\n")

saveRDS(
  code_quality_model,
  "bayesian-prompt-optimization/artefacts/code_quality_model.rds"
)
saveRDS(
  faithfulness_model,
  "bayesian-prompt-optimization/artefacts/faithfulness_model.rds"
)

# Build simulation grid over internal indices used by the model
newdata <- expand.grid(
  question = 1:Q,
  rater = 1:R
)

post_sim <- sim(code_quality_model, data = newdata, n = 1000)
post_expected_code_quality <- apply(post_sim, 2, mean)
post_expected_code_quality_means <- tapply(
  post_expected_code_quality,
  newdata$question,
  mean
)

post_sim <- sim(faithfulness_model, data = newdata, n = 1000)
post_expected_faithfulness <- apply(post_sim, 2, mean)
post_expected_faithfulness_means <- tapply(
  post_expected_faithfulness,
  newdata$question,
  mean
)

posterior_means <- data.frame(
  question_id = orig_ids,
  posterior_expected_code_quality = as.numeric(
    post_expected_code_quality_means[as.character(1:Q)]
  ),
  posterior_expected_faithfulness = as.numeric(
    post_expected_faithfulness_means[as.character(1:Q)]
  ),
  normal_code_quality_means = as.numeric(tapply(
    dat_long$score_codequality,
    dat_long$id_int,
    mean
  )[as.character(orig_ids)]),
  normal_faithfulness_means = as.numeric(tapply(
    dat_long$score_faithfulness,
    dat_long$id_int,
    mean
  )[as.character(orig_ids)])
)
write.csv(
  posterior_means,
  "bayesian-prompt-optimization/artefacts/means.csv",
  row.names = FALSE
)

cat("Saving process complete.\n")
