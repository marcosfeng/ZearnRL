# This R script loads and preprocesses data files,
# creates a list of model data, compiles and fits
# a Stan model using 'stan_model()' and 'sampling()' functions.
library(tidyverse)
library(data.table)
library(stats)
library(lubridate)
library(cmdstanr)
library(rstan)
set.seed(683328979)
# Random.org
# Timestamp: 2023-06-06 08:01:33 UTC

# Declare Functions and Variables -----------------------------------------

prepare_choice_array <- function(df, col1, col2, col3) {
  # Create zero-filled array with appropriate dimensions
  classroom_array <- unique(df$Classroom.ID)
  choice_array <- array(0, dim = c(length(classroom_array), max(df$row_n), 3))
  # Populate array
  for (i in 1:length(classroom_array)) {
    n <- nrow(df[Classroom.ID == classroom_array[i]])
    choice_array[i,1:n,1] <- unlist(df[Classroom.ID == classroom_array[i], ..col1])
    choice_array[i,1:n,2] <- unlist(df[Classroom.ID == classroom_array[i], ..col2])
    choice_array[i,1:n,3] <- unlist(df[Classroom.ID == classroom_array[i], ..col3])
  }

  return(choice_array)
}

# Import Data -------------------------------------------------------------

df <- read.csv(file = "Bayesian/df.csv")

FR_cols <- grep("FrobeniusNNDSVD", names(df), value = TRUE)
FRa_cols <- grep("FrobeniusNNDSVDA", names(df), value = TRUE)
KL_cols <- grep("KullbackLeibler", names(df), value = TRUE)

df <- df %>%
  arrange(Classroom.ID, week) %>%
  group_by(Classroom.ID) %>%
  mutate(state = ifelse(Tower.Alerts.per.Tower.Completion > lag(Tower.Alerts.per.Tower.Completion), 1, 2)) %>%
  ungroup() %>% group_by(MDR.School.ID) %>%
  mutate(Badges.per.Active.User = (Badges.per.Active.User - min(Badges.per.Active.User)) /
           (max(Badges.per.Active.User) - min(Badges.per.Active.User))) %>%
  mutate(across(all_of(FR_cols), ~ifelse(. > median(.), 1, 0), .names = "{.col}bin")) %>%
  mutate(across(all_of(FRa_cols), ~ifelse(. > median(.), 1, 0), .names = "{.col}bin")) %>%
  mutate(across(all_of(KL_cols), ~ifelse(. > median(.), 1, 0), .names = "{.col}bin")) %>%
  as.data.table() %>%
  .[Classroom.ID %in% sample(unique(Classroom.ID), size = length(unique(Classroom.ID)) * 0.05)] %>%
  setorder(Classroom.ID, week) %>%
  .[, row_n := seq_len(.N), by = .(Classroom.ID)]

choices <- list(
  FR = grep("FrobeniusNNDSVD(.)bin", names(df), value = TRUE)
  # FRa = grep("FrobeniusNNDSVDA(.)bin", names(df), value = TRUE),
  # KL = grep("Kullback.Leibler(.)bin", names(df), value = TRUE)
)

# Write to csv
write.csv(df, "./Bayesian/df_subset.csv")
# df <- read_csv("Bayesian/df_subset.csv")

# 1. Q-learning -----------------------------------------------------------

# Prepare list of models
models <- c("Bayesian/Stan Files/Q-learning.stan", "Bayesian/Stan Files/Q-learning-states.stan")

# Generate Stan data, fit model, and save for each choice array
for (choice in names(choices)) {
  for (model in models) {
    print(paste0("Processing model: ", model, " with choice array: ", choice))

    choice_array <- prepare_choice_array(df,
                                         choices[[choice]][1],
                                         choices[[choice]][2],
                                         choices[[choice]][3])

    stan_data <- list(
      N = length(unique(df$Classroom.ID)), # Number of teachers
      T = max(df[, row_n]), # Maximum Tsubj across all teachers
      Tsubj = df[, .N, by = .(Classroom.ID)][,N], # Number of rows by Teacher
      choice = choice_array,
      C = dim(choice_array)[3], # Number of choices
      S = sum(!is.na(unique(df$state))), # Number of states
      outcome = replace(as.matrix(dcast(df,
                                        Classroom.ID ~ row_n,
                                        value.var = "Badges.per.Active.User")), # Outcome matrix
                        is.na(as.matrix(dcast(df,
                                              Classroom.ID ~ row_n,
                                              value.var = "Badges.per.Active.User"))), 0)[,-1],
      week = replace(as.matrix(dcast(df,
                                     Classroom.ID ~ row_n,
                                     value.var = "week")), # Week matrix
                     is.na(as.matrix(dcast(df,
                                           Classroom.ID ~ row_n,
                                           value.var = "week"))), 0)[,-1],
      state = replace(as.matrix(dcast(df,
                                      Classroom.ID ~ row_n,
                                      value.var = "state")), # State matrix
                      is.na(as.matrix(dcast(df,
                                            Classroom.ID ~ row_n,
                                            value.var = "state"))), 1)[,-1]
    )

    my_model <- cmdstan_model(model)

    fit <- my_model$sample(
      data = stan_data,
      chains = 3,
      parallel_chains = 3,
      iter_warmup = 2500,
      iter_sampling = 2500
    )

    # Save the fit object
    fit$save_object(file = paste0("Bayesian/Results/",
                                  gsub("Bayesian/Stan Files/||.stan", "", model),
                                  "-", choice, ".RDS"))
  }
}


# 2. Actor-Critic ---------------------------------------------------------
df <- df %>%
  arrange(Classroom.ID, week) %>%
  group_by(Classroom.ID) %>%
  mutate(tower_state = scale(Tower.Alerts.per.Tower.Completion),
         actst_state = scale(Active.Users...Total)) %>%
  replace_na(list(tower_state = 0)) %>%
  ungroup() %>% as.data.table()
num_state_var = 3 # Tower.Alerts.per.Tower.Completion + 1

# Prepare list of models
models <- c("Bayesian/Stan Files/Actor-Critic.stan")

# Generate Stan data, fit model, and save for each choice array
for (choice in names(choices)) {
  for (model in models) {
    print(paste0("Processing model: ", model, " with choice array: ", choice))

    choice_array <- prepare_choice_array(df,
                                         choices[[choice]][1],
                                         choices[[choice]][2],
                                         choices[[choice]][3])

    # Create a 3D array with the correct dimensions
    state_array <- array(0, dim = c(length(unique(df$Classroom.ID)), max(df$row_n), num_state_var))
    # Create a vector of unique Classroom.ID values
    unique_ids <- unique(df$Classroom.ID)
    # Fill the array with your state data
    for (i in seq_along(unique_ids)) {
      current_id <- unique_ids[i]
      for (j in 1:max(df[df$Classroom.ID == current_id,]$row_n)) {
        # Assuming df is ordered by Classroom.ID and row_n
        state_array[i, j, ] <- c(1,
                                 df[df$Classroom.ID == current_id &
                                      df$row_n == j,]$tower_state,
                                 df[df$Classroom.ID == current_id &
                                      df$row_n == j,]$actst_state)
      }
    }

    stan_data <- list(
      N = length(unique(df$Classroom.ID)), # Number of teachers
      T = max(df[, row_n]), # Maximum Tsubj across all teachers
      Tsubj = df[, .N, by = .(Classroom.ID)][,N], # Number of rows by Teacher
      choice = choice_array,
      C = dim(choice_array)[3], # Number of choices
      outcome = replace(as.matrix(dcast(df,
                                        Classroom.ID ~ row_n,
                                        value.var = "Badges.per.Active.User")), # Outcome matrix
                        is.na(as.matrix(dcast(df,
                                              Classroom.ID ~ row_n,
                                              value.var = "Badges.per.Active.User"))), 0)[,-1],
      week = replace(as.matrix(dcast(df,
                                     Classroom.ID ~ row_n,
                                     value.var = "week")), # Week matrix
                     is.na(as.matrix(dcast(df,
                                           Classroom.ID ~ row_n,
                                           value.var = "week"))), 0)[,-1],
      S = num_state_var, # Number of states
      state = state_array
    )

    my_model <- cmdstan_model(model)

    fit <- my_model$sample(
      data = stan_data,
      chains = 3,
      parallel_chains = 3,
      iter_warmup = 2500,
      iter_sampling = 2500
    )

    # Save the fit object
    fit$save_object(file = paste0("Bayesian/Results/",
                                  gsub("Bayesian/Stan Files/||.stan", "", model),
                                  "-", choice, ".RDS"))
  }
}


# 3. Kernelized Q-learning ------------------------------------------------

# Prepare list of models
models <- c("Bayesian/Stan Files/Q-learning-kernel.stan")

# Generate Stan data, fit model, and save for each choice array
for (choice in names(choices)) {
  for (model in models) {
    print(paste0("Processing model: ", model, " with choice array: ", choice))

    choice_array <- prepare_choice_array(df,
                                         choices[[choice]][1],
                                         choices[[choice]][2],
                                         choices[[choice]][3])

    stan_data <- list(
      N = length(unique(df$Classroom.ID)), # Number of teachers
      T = max(df[, row_n]), # Maximum Tsubj across all teachers
      Tsubj = df[, .N, by = .(Classroom.ID)][,N], # Number of rows by Teacher
      choice = choice_array,
      C = dim(choice_array)[3], # Number of choices
      outcome = replace(as.matrix(dcast(df,
                                        Classroom.ID ~ row_n,
                                        value.var = "Badges.per.Active.User")), # Outcome matrix
                        is.na(as.matrix(dcast(df,
                                              Classroom.ID ~ row_n,
                                              value.var = "Badges.per.Active.User"))), 0)[,-1],
      week = replace(as.matrix(dcast(df,
                                     Classroom.ID ~ row_n,
                                     value.var = "week")), # Week matrix
                     is.na(as.matrix(dcast(df,
                                           Classroom.ID ~ row_n,
                                           value.var = "week"))), 0)[,-1],
      S = length(na.omit(unique(df$state))), # Number of states
      state = replace(as.matrix(dcast(df,
                                      Classroom.ID ~ row_n,
                                      value.var = "state")), # State matrix
                      is.na(as.matrix(dcast(df,
                                            Classroom.ID ~ row_n,
                                            value.var = "state"))), 1)[,-1],
      K = 4 # Number of lags in kernel
    )

    my_model <- cmdstan_model(model)

    fit <- my_model$sample(
      data = stan_data,
      chains = 3,
      parallel_chains = 3,
      iter_warmup = 2500,
      iter_sampling = 2500
    )

    # Save the fit object
    fit$save_object(file = paste0("Bayesian/Results/",
                                  gsub("Bayesian/Stan Files/||.stan", "", model),
                                  "-", choice, ".RDS"))
  }
}


# 4. Hierarchical Models --------------------------------------------------

df[, Teacher.Rank := .GRP, by = .(Teacher.User.ID)]

## Q-learning

# Prepare list of models
models <- c("Bayesian/Stan Files/Q-kernel-hierarchical.stan")

# Generate Stan data, fit model, and save for each choice array
for (choice in names(choices)) {
  for (model in models) {
    print(paste0("Processing model: ", model, " with choice array: ", choice))

    choice_array <- prepare_choice_array(df,
                                         choices[[choice]][1],
                                         choices[[choice]][2],
                                         choices[[choice]][3])

    stan_data <- list(
      N = length(unique(df$Classroom.ID)), # Number of teachers
      T = max(df[, row_n]), # Maximum Tsubj across all teachers
      S = length(na.omit(unique(df$state))), # Number of states
      K = 4, # Number of lags in kernel
      C = dim(choice_array)[3], # Number of choices
      Tsubj = df[, .N, by = .(Classroom.ID)][,N], # Number of rows by Teacher
      choice = choice_array,
      outcome = replace(as.matrix(dcast(df,
                                        Classroom.ID ~ row_n,
                                        value.var = "Badges.per.Active.User")), # Outcome matrix
                        is.na(as.matrix(dcast(df,
                                              Classroom.ID ~ row_n,
                                              value.var = "Badges.per.Active.User"))), 0)[,-1],
      week = replace(as.matrix(dcast(df,
                                     Classroom.ID ~ row_n,
                                     value.var = "week")), # Week matrix
                     is.na(as.matrix(dcast(df,
                                           Classroom.ID ~ row_n,
                                           value.var = "week"))), 0)[,-1],
      state = replace(as.matrix(dcast(df,
                                      Classroom.ID ~ row_n,
                                      value.var = "state")), # State matrix
                      is.na(as.matrix(dcast(df,
                                            Classroom.ID ~ row_n,
                                            value.var = "state"))), 1)[,-1],
      group = unique(df[, .(Classroom.ID, Teacher.User.ID, Teacher.Rank)])$Teacher.Rank,
      number_teachers = length(unique(df$Teacher.User.ID))
    )

    my_model <- cmdstan_model(model)

    fit <- my_model$sample(
      data = stan_data,
      chains = 3,
      parallel_chains = 3,
      iter_warmup = 2500,
      iter_sampling = 2500
    )

    # Save the fit object
    fit$save_object(file = paste0("Bayesian/Results/",
                                  gsub("Bayesian/Stan Files/||.stan", "", model),
                                  "-", choice, ".RDS"))
  }
}


# 5. Logit models -----------------------------------------------------------

# Function to find lag
get_lag_value <- function(datatable, col, lag_period, n_comp = NULL) {
  # Convert to data.table if it's a data.frame
  if (is.data.frame(datatable)) {
    datatable <- as.data.table(datatable)
  }

  # Add a column for week_lag
  week_lag <- c(0, diff(datatable$week))
  set(datatable, j = "week_lag", value = week_lag)

  if (is.null(n_comp)) {
    # Update the lag column with shift function
    new_col_name <- paste0(col, "_", lag_period)
    new_col_value <- shift(datatable[[col]], lag_period, fill = 0, type = "lag")
    set(datatable, j = new_col_name, value = new_col_value)
  } else {
    for (comp in 1:n_comp) {
      # Update the lag column with shift function
      new_col_name <- paste0(col, comp, "_", lag_period)
      new_col_value <- shift(datatable[[paste0(col, comp)]], lag_period, fill = 0, type = "lag")
      set(datatable, j = new_col_name, value = new_col_value)
    }
  }

  return(datatable)
}

df <- df %>%
  mutate(state = case_when(is.na(state) ~ 0,
                           .default = state - 1))

# Non-hierarchical models
for (col in paste0("FrobeniusNNDSVD", 1:3, "bin")) {
  df <- get_lag_value(df, col, 1)
}
# Prepare data for Stan
stan_data <- list(
  C = length(unique(df$Classroom.ID)), # Number of classrooms
  N = nrow(df),
  X = df %>% select(FrobeniusNNDSVD1bin_1, Badges.per.Active.User, state) %>% as.matrix(),
  y = df %>% select(FrobeniusNNDSVD1bin, FrobeniusNNDSVD2bin, FrobeniusNNDSVD3bin) %>% as.matrix(),
  classroom = df %>%
    mutate(classroom = as.integer(factor(Classroom.ID))) %>%
    select(classroom) %>% unlist() %>% as.integer()
)

# Compile the Stan model
logistic_model <- cmdstan_model("Bayesian/Stan Files/Logit.stan")

# Fit the model
fit <- logistic_model$sample(
  data = stan_data,
  chains = 3,
  parallel_chains = 3,
  iter_warmup = 2500,
  iter_sampling = 2500
)

# Save the fit object
fit$save_object(file = "Bayesian/Results/Logit.RDS")

# Hierarchical models






#quick diagnostics
library("shinystan")
launch_shinystan(fit)
