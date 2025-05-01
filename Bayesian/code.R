# This R script loads and preprocesses data files,
# creates a list of model data, compiles and fits
# a Stan model using 'stan_model()' and 'sampling()' functions.
library(tidyverse)
library(data.table)
library(stats)
library(lubridate)
library(cmdstanr)
library(posterior)
library(rstan)
library(loo)
library(matrixStats)
options(mc.cores = parallel::detectCores())
set.seed(683328979)
# Random.org
# Timestamp: 2023-06-06 08:01:33 UTC

# Declare Functions and Variables -----------------------------------------

load("CBM/data/stan_data.RData")

# Prepare data for Stan
L <- sum(stan_data$Tsubj)  # Total number of trials
starts <- c(1, cumsum(stan_data$Tsubj) + 1)  # Starting indices
choice_concat <- numeric(L)  # Concatenated choices
reward_concat <- numeric(L)  # Concatenated rewards
week_concat <- numeric(L)  # Concatenated weeks

pos <- 1
for(i in 1:stan_data$N) {
  choice_concat[pos:(pos + stan_data$Tsubj[i] - 1)] <-
    stan_data$choice[i, 1:stan_data$Tsubj[i], 2] # Get only choice 2
  reward_concat[pos:(pos + stan_data$Tsubj[i] - 1)] <-
    stan_data$all_vars[i, 1:stan_data$Tsubj[i], 1]  # Get only reward 1
  week_concat[pos:(pos + stan_data$Tsubj[i] - 1)] <-
    stan_data$week[i, 1:stan_data$Tsubj[i]]
  pos <- pos + stan_data$Tsubj[i]
}

stan_data <- list(
  N = stan_data$N,
  L = L,
  Tsubj = stan_data$Tsubj,
  choice = choice_concat,
  reward = reward_concat,
  week = week_concat,
  starts = starts,
  mixis = 0
)
saveRDS(stan_data, "Bayesian/stan_data.RDS")

# Prepare list of models
model <- c("Bayesian/bl_opt.stan", "Bayesian/ql_opt.stan")
model_nh <- c("Bayesian/bl_nonhier.stan", "Bayesian/ql_nonhier.stan")

# Model Estimation ---------------------------------------------------------

# Generate Stan data, fit model, and save for each choice array
for (i in 1:length(model)) {
  my_model <- cmdstan_model(model[i], compile_model_methods = TRUE,
                            force_recompile = TRUE)

  rm(list = setdiff(ls(), c("model", "my_model", "stan_data", "i")))

  fit <- my_model$sample(
    data = stan_data,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 2000,
    iter_sampling = 1000,
    adapt_delta = 0.95,
    max_treedepth = 12,
    init = 0.1
  )

  # Save the loo object
  loo_results <- fit$loo(moment_match = TRUE, cores = 4)
  saveRDS(loo_results, file = paste0(
    "Bayesian/Results/",
    gsub("Bayesian/||.stan", "", model[i]),
    "_loo.RDS"
  ))
  # Save the fit object
  fit$save_object(file = paste0(
    "Bayesian/Results/",
    gsub("Bayesian/||.stan", "", model[i]),
    ".RDS"
  ))
}


# Model Diagnostic --------------------------------------------------

# Non-hierarchical models
ql_nonhier <- readRDS("Bayesian/Results/ql_nonhier.RDS")
ql_nonhier$diagnostic_summary()  # HMC diagnostics
bl_nonhier <- readRDS("Bayesian/Results/bl_nonhier.RDS")
bl_nonhier$diagnostic_summary()

# Hierarchical models
ql_fit <- readRDS("Bayesian/Results/ql_opt.RDS")
ql_fit$diagnostic_summary()
bl_fit <- readRDS("Bayesian/Results/bl_opt.RDS")
bl_fit$diagnostic_summary()


# Model Comparison --------------------------------------------------

compute_mixis_loo <- function(model, stan_data) {
  stan_data$mixis <- 1
  fit_mix <- model$sample(
    data = stan_data,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 2000,
    iter_sampling = 1000,
    adapt_delta = 0.95,
    max_treedepth = 12,
    init = 0.1
  )

  # Extract log likelihoods
  log_lik_mix <- fit_mix$draws("log_lik", format = "matrix")

  # Compute mixture estimators (as per Silva and Zanella (2022))
  l_common_mix <- rowLogSumExps(-log_lik_mix)
  log_weights <- -log_lik_mix - l_common_mix
  elpd_mixis <- logSumExp(-l_common_mix) - colLogSumExps(log_weights)

  return(elpd_mixis)
}

# For baseline model
bl_loo <- readRDS("Bayesian/Results/bl_opt_loo.RDS")
bl_model <- cmdstan_model(model[1])
bl_mixis <- compute_mixis_loo(bl_model, stan_data)

gc()  # Garbage collection to free memory

# For Q-learning model
ql_loo <- readRDS("Bayesian/Results/ql_opt_loo.RDS")
ql_model <- cmdstan_model(model[2])
ql_mixis <- compute_mixis_loo(ql_model, stan_data)


model_comparison <- list(
  baseline = list(
    loo = bl_loo,
    mixis = bl_mixis,
    # Root mean squared error (RMSE) of
    # mixture estimators relative to Pareto smoothed (PSIS)
    rmse = sqrt(mean((bl_loo$pointwise[,1]-bl_mixis)^2))
  ),
  qlearning = list(
    loo = ql_loo,
    mixis = ql_mixis,
    rmse = sqrt(mean((ql_loo$pointwise[,1]-ql_mixis)^2))
  ),
  # comparison = loo_compare(ql_loo, bl_loo),
  mixis_difference = data.frame(
    difference = sum(ql_mixis - bl_mixis),
    se = sd(ql_mixis-bl_mixis)*sqrt(length(ql_mixis))
  )
)
saveRDS(model_comparison, "Bayesian/Results/model_comparison.RDS")

# Model Summaries --------------------------------------------------

ql_sum <- qs::qread("Bayesian/Results/QLsum.qs")
bl_sum <- qs::qread("Bayesian/Results/BLsum.qs")

# Save the draws
sum <- fit$summary()
qs::qsave(x = sum, file = paste0(
  "Bayesian/Results/",
  gsub("Bayesian/||.stan", "", model[i]),
  "sum.qs"
))


# Parameter Summaries --------------------------------------------------

extract_parameter_summaries <-
  function(fit_object,
           parameter_names = c("alpha", "gamma", "tau", "cost", "ev_init")) {
    # Get all draws in a matrix format
    draws <- fit_object$draws(format = "matrix")

    transform_param <- function(param_name, values) {
      if(param_name %in% c("alpha", "gamma")) {
        # Logistic transform for alpha and gamma
        return(1/(1 + exp(-values)))
      } else if(param_name %in% c("tau", "cost")) {
        # Exponential transform for tau and cost
        return(exp(values))
      } else {
        # No transform for ev_init
        return(values)
      }
    }

    inverse_transform <- function(param_name, values) {
      if(param_name %in% c("alpha", "gamma")) {
        # Apply bounds to prevent Inf/-Inf
        values_bounded <- pmin(pmax(values, 0.0001), 0.9999)
        # Inverse of logistic: logit function
        return(log(values_bounded/(1-values_bounded)))
      } else if(param_name %in% c("tau", "cost")) {
        # Apply minimum bound to prevent -Inf
        values_bounded <- pmax(values, 0.0001)
        # Inverse of exponential: log function
        return(log(values_bounded))
      } else {
        # No transform for ev_init
        return(values)
      }
    }

    # Function to get parameter summary
    get_param_summary <- function(param_name) {
      # Get all columns matching the parameter name
      param_cols <- grep(paste0("^", param_name, "\\["), colnames(draws), value = TRUE)

      # Get hyperparameters (mu)
      if(param_name == "ev_init") {
        hyper_mu_cols <- grep(paste0("^mu_", param_name, "$"),
                              colnames(draws),
                              value = TRUE)
      } else {
        hyper_mu_cols <- grep(paste0("^mu_", param_name, "_raw$"),
                              colnames(draws),
                              value = TRUE)
      }

      # Calculate summaries for individual parameters
      if(length(param_cols) > 0) {
        param_draws <- draws[, param_cols, drop = FALSE]

        # Calculate mean for each subject, then average
        n_subjects <- ncol(param_draws)
        subject_untransformed_means <- numeric(n_subjects)
        for(j in 1:n_subjects) {
          raw_values <- inverse_transform(param_name, param_draws[, j])
          subject_untransformed_means[j] <- mean(raw_values)
        }

        # Calculate overall statistics across subjects (still in raw space)
        param_mean_raw <- mean(subject_untransformed_means)
        param_median_raw <- median(subject_untransformed_means)
        param_se_raw <- sd(subject_untransformed_means)/sqrt(n_subjects)

        pooled_stats <- list(
          # Transform back for reporting
          Mean = transform_param(param_name, param_mean_raw),
          Median = transform_param(param_name, param_median_raw),
          # For 95% CI
          CI_lower = transform_param(param_name, param_mean_raw - 1.96 * param_se_raw),
          CI_upper = transform_param(param_name, param_mean_raw + 1.96 * param_se_raw)
        )
      } else {
        pooled_stats <- NULL
      }

      # Calculate summaries for hyperparameters (already in raw scale)
      if(length(hyper_mu_cols) > 0) {
        hyper_draws <- draws[, hyper_mu_cols, drop = FALSE]
        hyper_mean_raw <- mean(hyper_draws[, 1])
        transformed_mean <- transform_param(param_name, hyper_mean_raw)
        # For CI, transform each sample then compute quantiles
        transformed_samples <- transform_param(param_name, hyper_draws[, 1])
        hyper_stats <- list(
          Mean = transformed_mean,
          CI_lower = quantile(transformed_samples, 0.025),
          CI_upper = quantile(transformed_samples, 0.975)
        )
      } else {
        hyper_stats <- NULL
      }

      return(list(
        pooled = pooled_stats,
        hyper = hyper_stats
      ))
    }

    # Get summaries for all parameters
    param_summaries <- lapply(parameter_names, get_param_summary)
    names(param_summaries) <- parameter_names

    # Create results dataframe
    results <- data.frame(
      Parameter = rep(parameter_names, each = 3),
      Statistic = rep(c("Mean", "Median", "95% CI"), length(parameter_names)),
      NonHierarchical = NA,  # To be filled with Matlab results
      Pooled = NA,
      Hyperparameters = NA
    )

    # Fill in the values
    for(i in seq_along(parameter_names)) {
      param <- parameter_names[i]
      idx <- ((i-1)*3 + 1):(i*3)

      if(!is.null(param_summaries[[param]]$pooled)) {
        results$Pooled[idx[1]] <- round(param_summaries[[param]]$pooled$Mean, 4)
        results$Pooled[idx[2]] <- round(param_summaries[[param]]$pooled$Median, 4)
        results$Pooled[idx[3]] <- sprintf("[%.4f, %.4f]",
                                          param_summaries[[param]]$pooled$CI_lower,
                                          param_summaries[[param]]$pooled$CI_upper)
      }

      if(!is.null(param_summaries[[param]]$hyper)) {
        results$Hyperparameters[idx[1]] <- round(param_summaries[[param]]$hyper$Mean, 4)
        results$Hyperparameters[idx[3]] <- sprintf("[%.4f, %.4f]",
                                                   param_summaries[[param]]$hyper$CI_lower,
                                                   param_summaries[[param]]$hyper$CI_upper)
      }
    }

    return(results)
  }


# Load your Stan fits
ql_fit <- readRDS("Bayesian/Results/ql_opt.RDS")
bl_fit <- readRDS("Bayesian/Results/bl_opt.RDS")
model_comparison <- readRDS("Bayesian/Results/model_comparison.RDS")

# Extract parameter summaries
ql_params <- extract_parameter_summaries(ql_fit)
bl_params <- extract_parameter_summaries(bl_fit)

# Calculate MixIS summary statistics
mixis_stats <- list(
  qlearning = list(
    mean_elpd = mean(model_comparison$qlearning$mixis),
    se_elpd = sd(model_comparison$qlearning$mixis)/sqrt(length(model_comparison$qlearning$mixis)),
    rmse = model_comparison$qlearning$rmse
    ),
  baseline = list(
    mean_elpd = mean(model_comparison$baseline$mixis),
    se_elpd = sd(model_comparison$baseline$mixis)/sqrt(length(model_comparison$baseline$mixis)),
    rmse = model_comparison$baseline$rmse
  )
)

# Create the complete table
create_comparison_table <- function(ql_fit, bl_fit) {
  # Get parameter summaries
  params <- extract_parameter_summaries(ql_fit)

  # # Get model fit statistics
  # ql_stats <- get_model_fit_stats(ql_fit)
  # bl_stats <- get_model_fit_stats(bl_fit)
  #
  # # Add model fit statistics rows
  # fit_stats <- data.frame(
  #   Parameter = c("Log-likelihood", "Log-likelihood", "AIC", "AIC"),
  #   Statistic = c("Q-learning", "Baseline", "Q-learning", "Baseline"),
  #   NonHierarchical = c(ql_stats$loglik, bl_stats$loglik,
  #                       ql_stats$aic, bl_stats$aic),
  #   Pooled = c(ql_stats$loglik, bl_stats$loglik,
  #              ql_stats$aic, bl_stats$aic),
  #   Hyperparameters = NA
  # )
  #
  # # Combine parameter and fit statistics
  # results <- rbind(params, fit_stats)

  return(params)
}

# Generate table
comparison_table <- create_comparison_table(ql_fit, bl_fit)

comparison_table <- create_comparison_table(ql_nonhier, bl_nonhier)


# 5. Hierarchical Models --------------------------------------------------

df[, Teacher.Rank := .GRP, by = .(Teacher.User.ID)]

## Actor Critic

# Prepare list of models
model <- c("Bayesian/Stan Files/Actor-Critic-hierarchical.stan")

# Generate Stan data, fit model, and save for each choice array
stan_data$group <- unique(
  df[, .(Classroom.ID, Teacher.User.ID, Teacher.Rank)]
)$Teacher.Rank
stan_data$G <- length(unique(df$Teacher.User.ID)) #Number of teachers

my_model <- cmdstan_model(model)

# Remove everything except my_model and stan_data
rm(list = setdiff(ls(), c("my_model", "stan_data")))

fit <- my_model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  init = function() {
    list(mu_cost = 0.8, mu_gamma = 0.8, mu_alpha = c(0.5, 0.5), mu_tau = 7.5,
         cost = rep(0.8, stan_data$G), gamma = rep(0.8, stan_data$G),
         alpha = array(0.5, dim = c(2, stan_data$G)),
         tau = rep(7.5, stan_data$G), w_0 = array(0.2, dim = c(stan_data$G, stan_data$S)),
         theta_0 = array(0, dim = c(stan_data$G, stan_data$S)))
  }
)

# Save the fit object
fit$save_output_files(dir = "Bayesian/Results/")

draws_df <- fit$draws()
try(fit$sampler_diagnostics(), silent = TRUE)
loo1 <- fit$loo()
sum1 <- fit$summary()
qs::qsave(x = fit, file = "Bayesian/Results/ACfit.qs")
qs::qsave(x = loo1, file = "Bayesian/Results/ACloo.qs")
qs::qsave(x = draws_df, file = "Bayesian/Results/ACdraws.qs")
qs::qsave(x = sum1, file = "Bayesian/Results/ACsum.qs")


files <- c(
  "Bayesian/Results/Actor-Critic-hierarchical-202403251140-1-914436.csv",
  "Bayesian/Results/Actor-Critic-hierarchical-202403251140-2-914436.csv",
  "Bayesian/Results/Actor-Critic-hierarchical-202403251140-3-914436.csv",
  "Bayesian/Results/Actor-Critic-hierarchical-202403251140-4-914436.csv"
)
csv_contents <- as_cmdstan_fit(files)
draws_df <- csv_contents$draws()
sum1 <- csv_contents$summary()
