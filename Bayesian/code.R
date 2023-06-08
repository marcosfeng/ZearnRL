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

prepare_choice_array <- function(df, col1, col2, col3){
  choices_df <- df %>%
    select(Classroom.ID, row_n, !!col1, !!col2, !!col3)
  # Combine choices into a single column and create an array
  choices_df <- choices_df %>%
    mutate(choice = ifelse(!!sym(col1) == 1, 1,
                           ifelse(!!sym(col2) == 1, 2,
                                  ifelse(!!sym(col3) == 1, 3, 4)))) %>%
    dcast(Classroom.ID ~ row_n, value.var = "choice")
  # Convert to array
  choice_array <- array(0, dim = c(nrow(choices_df), max(df$row_n), 4))
  for (i in 1:nrow(choices_df)) {
    for (j in 2:ncol(choices_df)) {
      choice_array[i, j - 1, choices_df[[i,j]]] <- 1
    }
  }
  # Remove unnecessary columns and objects
  rm(choices_df)

  return(choice_array)
}

# Import Data -------------------------------------------------------------

df <- read.csv(file = "Bayesian/df.csv")

FR_cols <- grep("FrobeniusNNDSVD", names(df), value = TRUE)
FRa_cols <- grep("FrobeniusNNDSVDA", names(df), value = TRUE)
KL_cols <- grep("Kullback.Leibler", names(df), value = TRUE)

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
  FR = grep("FrobeniusNNDSVD(.)bin", names(df), value = TRUE),
  FRa = grep("FrobeniusNNDSVDA(.)bin", names(df), value = TRUE),
  KL = grep("Kullback.Leibler(.)bin", names(df), value = TRUE)
)

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
      iter_warmup = 500,
      iter_sampling = 500
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
  mutate(tower_state = (Tower.Alerts.per.Tower.Completion -
                    lag(Tower.Alerts.per.Tower.Completion)),
         users_state = (Active.Users...Total -
                          lag(Active.Users...Total))) %>%
  replace_na(list(tower_state = 0, users_state = 0)) %>%
  ungroup() %>% as.data.table()

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
    state_array <- array(0, dim = c(length(unique(df$Classroom.ID)), max(df$row_n), 3))
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
                                      df$row_n == j,]$users_state)
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
      S = 3, # Number of states
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


# 3. Hierarchical Models --------------------------------------------------

df[, Teacher.Rank := .GRP, by = .(Teacher.User.ID)]

## Q-learning

# Prepare list of models
models <- c("Bayesian/Stan Files/Q-learning-hierarchical.stan",
            "Bayesian/Stan Files/Q-learning-states-hierarchical.stan")

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

## Actor Critic

models <- c("Bayesian/Stan Files/Actor-Critic-hierarchical.stan")

# Generate Stan data, fit model, and save for each choice array
for (choice in names(choices)) {
  for (model in models) {
    print(paste0("Processing model: ", model, " with choice array: ", choice))

    choice_array <- prepare_choice_array(df,
                                         choices[[choice]][1],
                                         choices[[choice]][2],
                                         choices[[choice]][3])

    # Create a 3D array with the correct dimensions
    state_array <- array(0, dim = c(length(unique(df$Classroom.ID)), max(df$row_n), 3))
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
                                      df$row_n == j,]$users_state)
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
      S = 3, # Number of states
      state = state_array,
      group = unique(df[, .(Classroom.ID, Teacher.User.ID, Teacher.Rank)])$Teacher.Rank,
      number_teachers = length(unique(df$Teacher.User.ID))
    )

    my_model <- cmdstan_model(model)

    fit <- my_model$sample(
      data = stan_data,
      chains = 3,
      parallel_chains = 3,
      iter_warmup = 5000,
      iter_sampling = 5000
    )

    # Save the fit object
    fit$save_object(file = paste0("Bayesian/Results/",
                                  gsub("Bayesian/Stan Files/||.stan", "", model),
                                  "-", choice, ".RDS"))
  }
}


#quick diagnostics
library("shinystan")
launch_shinystan(fit)
