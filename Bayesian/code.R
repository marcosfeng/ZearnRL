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
df$PC1 <- ifelse(df$Component1 > 0, 1, 0)
df$PC2 <- ifelse(df$Component2 > 0, 1, 0)
df$PC3 <- ifelse(df$Component3 > 0, 1, 0)

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
columns <- c("PC1", "PC2", "PC3")
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

# Use pca1_1 as the first model:
## Get unique teacher IDs
unique_teachers <- unique(dt$Teacher.User.ID)
## Sample 10% of unique teacher IDs
sampled_teachers <- sample(unique_teachers, size = length(unique_teachers) * 0.10)
# Filter dt to include only sampled teachers
dt_sampled <- dt[Teacher.User.ID %in% sampled_teachers]
setorder(dt_sampled, Teacher.User.ID, week)

dt_collapsed <- dt_sampled[, .(
  Badges.per.Active.User = weighted.mean(Badges.per.Active.User, Active.Users...Total),
  Tower.Alerts.per.Tower.Completion = mean(Tower.Alerts.per.Tower.Completion),
  Active.Users...Total = sum(Active.Users...Total),
  st_login = sum(st_login),
  pca1_1 = mean(pca1_1)
), by = .(Teacher.User.ID, week)]
dt_collapsed[, `:=`(
  Tsubj = .N,
  row_n = seq_len(.N)
), by = .(Teacher.User.ID)]

# Create binary choice
dt_collapsed[, choice := ifelse(pca1_1 < 0, 0, 1)]


## Q-learning
my_model <- stan_model(file = "Bayesian/RL_Q-learning.stan",
                       verbose = FALSE)
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
                                          value.var = "choice"))), 0)[,-1]
)

sample <- sampling(object = my_model,
                   data = stan_data,
                   iter = 200,
                   chains = 10,
                   cores = 3)
save(sample, file = "Bayesian/sample.Rdata")


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
