# This R script loads and preprocesses data files,
# creates a list of model data, compiles and fits
# a Stan model using 'stan_model()' and 'sampling()' functions.
library(tidyverse)
library(data.table)
library(rstan)
library(stats)
library(lubridate)
library(cmdstanr)

df <- read.csv(file = "Bayesian/df.csv")
df$PC1bin <- ifelse(df$PC1 > 0, 1, 0)
df$PC2bin <- ifelse(df$PC2 > 0, 1, 0)
df$PC3bin <- ifelse(df$PC3 > 0, 1, 0)
df$AE1bin <- ifelse(df$AE1 > 0, 1, 0)
df$AE2bin <- ifelse(df$AE2 > 0, 1, 0)
df$AE3bin <- ifelse(df$AE3 > 0, 1, 0)

# Function to get lagged value
get_lag_value <- function(df, col, lag_period) {
  df %>%
    group_by(Classroom.ID) %>%
    mutate(!!paste0(col, "_", lag_period) :=
             replace_na(
               eval(sym(col), df)[match(week, (week - lag_period))],
               0)
    )
}
# Define number of lags and columns
n_lags = 1
columns <- c("PC1bin", "PC2bin", "PC3bin", "AE1bin", "AE2bin", "AE3bin")
# Add lagged variables
for (col in columns) {
  for (lag_period in 1:n_lags) {
    df <- df %>%
      arrange(Classroom.ID, week) %>%
      get_lag_value(col, lag_period) %>%
      ungroup()
  }
}
dt <- as.data.table(df)
## Get unique teacher IDs
unique_classrooms <- unique(dt$Classroom.ID)
## Sample 10% of unique teacher IDs
sampled_classrooms <- sample(unique_classrooms, size = length(unique_classrooms) * 0.05)
# Filter dt to include only sampled teachers
dt_sampled <- dt[Classroom.ID %in% sampled_classrooms]
setorder(dt_sampled, Classroom.ID, week)

dt_sampled[, `:=`(
  row_n = seq_len(.N)
), by = .(Classroom.ID)]

# Prepare the choice array
choices_df <- dt_sampled %>%
  select(Classroom.ID, row_n, PC1bin, PC2bin, PC3bin)
# Combine choices into a single column and create an array
choices_df <- choices_df %>%
  mutate(choice = ifelse(PC1bin == 1, 1,
                         ifelse(PC2bin == 1, 2,
                                ifelse(PC3bin == 1, 3, 4)))) %>%
  dcast(Classroom.ID ~ row_n, value.var = "choice")
# Convert to array
choice_array <- array(0, dim = c(nrow(choices_df), max(dt_sampled$row_n), 4))
for (i in 1:nrow(choices_df)) {
  for (j in 2:ncol(choices_df)) {
    choice_array[i, j - 1, choices_df[[i,j]]] <- 1
  }
}
# Remove unnecessary columns and objects
rm(choices_df)

week_matrix <- replace(as.matrix(dcast(dt_sampled,
                                       Classroom.ID ~ row_n,
                                       value.var = "week")), # Week matrix
                       is.na(as.matrix(dcast(dt_sampled,
                                             Classroom.ID ~ row_n,
                                             value.var = "week"))), 0)[,-1]

# Generate stan data
stan_data <- list(
  N = length(unique(dt_sampled$Classroom.ID)), # Number of teachers
  T = max(dt_sampled[, row_n]), # Maximum Tsubj across all teachers
  # S = 4, # Number of states
  Tsubj = dt_sampled[, .N, by = .(Classroom.ID)][,N], # Number of rows by Teacher
  choice = choice_array,
  outcome = replace(as.matrix(dcast(dt_sampled,
                                    Classroom.ID ~ row_n,
                                    value.var = "Badges.per.Active.User")), # Outcome matrix
                    is.na(as.matrix(dcast(dt_sampled,
                                          Classroom.ID ~ row_n,
                                          value.var = "Badges.per.Active.User"))), 0)[,-1],
  week = week_matrix
)

## Q-learning
my_model <- stan_model(file = "Bayesian/Q-learning-PCA.stan",
                       verbose = FALSE)
sample <- sampling(object = my_model,
                   data = stan_data,
                   iter = 200,
                   chains = 3,
                   cores = 3)
save(sample, file = "Bayesian/Q-learning-PCA.Rdata")


## Actor-Critic
# Generate stan data
stan_data <- list(
  N = length(unique(dt_collapsed$Teacher.User.ID)), # Number of teachers
  T = max(dt_collapsed[, row_n]), # Maximum Tsubj across all teachers
  S = 4, # Number of states
  Tsubj = dt_collapsed[, .N, by = .(Teacher.User.ID)][,N], # Number of rows by Teacher
  choice = replace(as.matrix(dcast(dt_collapsed,
                                   Teacher.User.ID ~ row_n,
                                   value.var = "choice")), # Choice matrix
                   is.na(as.matrix(dcast(dt_collapsed,
                                         Teacher.User.ID ~ row_n,
                                         value.var = "choice"))), 0)[,-1],
  outcome = replace(as.matrix(dcast(dt_collapsed,
                                    Teacher.User.ID ~ row_n,
                                    value.var = "Badges.per.Active.User")), # Outcome matrix
                    is.na(as.matrix(dcast(dt_collapsed,
                                          Teacher.User.ID ~ row_n,
                                          value.var = "choice"))), 0)[,-1],
  states = matrix() # States matrix
)
# Create a template data.table with all teacher-week combinations
dt_full <- CJ(Teacher.User.ID = unique(dt_collapsed$Teacher.User.ID),
              row_n = 1:stan_data[["T"]])
# Join dt_collapsed with dt_full
dt_full <- dt_collapsed[dt_full, on = .(Teacher.User.ID, row_n)]
# Replace NA's with appropriate values (0 for states, -1 for choice and outcome)
cols_to_replace <- c("Tower.Alerts.per.Tower.Completion",
                     "Active.Users...Total", "st_login",
                     "choice", "Badges.per.Active.User")
dt_full[, (cols_to_replace) := lapply(.SD,
                                      function(x) replace(x, is.na(x), 0)),
        .SDcols = cols_to_replace]
# Order by Teacher.User.ID and week
setorder(dt_full, Teacher.User.ID, week)
# Create states matrix
stan_data[["states"]] <- as.matrix(cbind(1,
                                         dt_full[, .(Tower.Alerts.per.Tower.Completion,
                                                     Active.Users...Total,
                                                     st_login)])) # States matrix





#diagnostics
library("shinystan")
launch_shinystan(sample)
