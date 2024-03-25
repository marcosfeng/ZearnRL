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
options(mc.cores = parallel::detectCores())
set.seed(683328979)
# Random.org
# Timestamp: 2023-06-06 08:01:33 UTC

# Declare Functions and Variables -----------------------------------------

prepare_choice_array <- function(df, cols) {
  # Create zero-filled array with appropriate dimensions
  classroom_array <- unique(df$Classroom.ID)
  choice_array <- array(0, dim = c(length(classroom_array),
                                   max(df$row_n)))
  # Populate array
  for (i in 1:length(classroom_array)) {
    n <- nrow(df[Classroom.ID == classroom_array[i]])
    choice_array[i,1:n] <- unlist(
      df[Classroom.ID == classroom_array[i], ..cols]
      )
  }
  return(choice_array)
}

# Import Data -------------------------------------------------------------

action <- c("Frobenius.NNDSVD_teacher")
reward <- c("Frobenius.NNDSVD_student")

# Data and Variables
df <- read.csv(file = "Bayesian/df.csv") %>%
  mutate(across(where(is.numeric),
                ~ ifelse(. < .Machine$double.eps, 0, .))) %>%
  mutate(across(dplyr::starts_with("Frobenius.NNDSVD_teacher"),
                ~ if_else(.>median(.),1,0)))

choices <- list(
  FR = grep("Frobenius.NNDSVD_teacher", names(df), value = TRUE)
  # FRa = grep("FrobeniusNNDSVDA(.)bin", names(df), value = TRUE),
  # KL = grep("Kullback.Leibler(.)bin", names(df), value = TRUE)
)

choice = choices[["FR"]][2]
reward <- "Frobenius.NNDSVD_student1"
state <- c("Frobenius.NNDSVD_student3")

df <- df %>%
  arrange(Classroom.ID, week) %>%
  as.data.table() %>%
  .[, row_n := seq_len(.N), by = .(Classroom.ID)] %>%
  # Filter out Classroom.IDs with sd = 0 for Choices
  .[, .SD[!apply(.SD[, choice, with = FALSE], 2, sd) == 0],
    by = Classroom.ID] %>%
  # Filter out Classroom.IDs with sd = 0 for Reward
  .[, .SD[!apply(.SD[, reward, with = FALSE], 2, sd) == 0],
    by = Classroom.ID] %>%
  # Filter out Classroom.IDs with sd = 0 for States
  .[, .SD[!apply(.SD[, state, with = FALSE], 2, sd) == 0],
    by = Classroom.ID] %>%
  .[, week := week - min(week) + 1, by = .(Classroom.ID)] %>%
  .[, Duration := max(row_n), by = .(Classroom.ID)] %>%
  .[Duration > 7] %>%
  setorder(Classroom.ID, week)

choice_array <- prepare_choice_array(df, choice)
rewards <- c()
states  <- c()

# 2. Actor-Critic ---------------------------------------------------------

# Prepare list of models
model <- c("Bayesian/Stan Files/Actor-Critic.stan")

num_state_var = length(state) + 1
# Create a 3D array with the correct dimensions
state_array <-
  array(0, dim = c(length(unique(
    df$Classroom.ID
  )), max(df$row_n), num_state_var))
# Create a vector of unique Classroom.ID values
unique_ids <- unique(df$Classroom.ID)
# Fill the array with your state data
for (i in seq_along(unique_ids)) {
  current_id <- unique_ids[i]
  for (j in 1:max(df[Classroom.ID == current_id, ]$row_n)) {
    # Assuming df is ordered by Classroom.ID and row_n
    state_array[i, j,] <- c(1, as.numeric(
      df[Classroom.ID == current_id & row_n == j, ..state]))
  }
}
state_array[is.na(state_array)] <- 0

stan_data <- list(
  # Number of teachers
  N = length(unique(df$Classroom.ID)),
  # Maximum Tsubj across all teachers
  T = max(df[, row_n]),
  # Number of rows by Teacher
  Tsubj = df[, .N, by = .(Classroom.ID)][, N],
  choice = choice_array,
  # Number of choices
  outcome = replace(as.matrix(dcast(df, Classroom.ID ~ row_n,
                                     value.var = reward)), # Outcome matrix
                    is.na(as.matrix(dcast(df, Classroom.ID ~ row_n,
                                          value.var = reward))), 0)[, -1],
  week = replace(as.matrix(dcast(df, Classroom.ID ~ row_n,
                                 value.var = "week")),
                 is.na(as.matrix(dcast(df, Classroom.ID ~ row_n,
                                       value.var = "week"))), 0)[,c(-1)],
  S = num_state_var,
  # Number of states
  state = state_array
)

my_model <- cmdstan_model(model)

fit <- my_model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

# Save the fit object
fit$save_object(file = paste0(
  "Bayesian/Results/",
  gsub("Bayesian/Stan Files/||.stan", "", model),
  "-", choice, ".RDS"
))

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


# Quick diagnostics -------------------------------------------------------

# library("shinystan")
# fit$summary()
# stanfit <- rstan::read_stan_csv(fit$output_files())
# launch_shinystan(stanfit)

# library(bayesplot)
# yrep <- rstantools::posterior_predict(stanfit, "y_pred")
# ppc_hist(y, yrep)

# install.packages("pROC")
# library(pROC)

# kernel_h <- readRDS("./Bayesian/Results/Q-kernel-hierarchical-FR.RDS")
# posterior_samples <- kernel_h$summary(variables = "y_pred")

# library(DescTools)
# observed <- prepare_choice_array(df,
#                                  choices[[1]][1],
#                                  choices[[1]][2],
#                                  choices[[1]][3])
# observed_vector <- as.vector(observed)
# CalibrationPlot(posterior_samples, observed)
# ppc_hist(observed_vector, posterior_samples)

