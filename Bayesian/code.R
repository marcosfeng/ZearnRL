# This R script loads and preprocesses data files,
# creates a list of model data, compiles and fits 
# a Stan model using 'stan_model()' and 'sampling()' functions.
library(tidyverse)
library(data.table)
library(rstan)
library(stats)
library(lubridate)
library(cmdstanr)

df <- read.csv(file = "Data/df_clean.csv")
# Convert columns to appropriate data types
dt <- as.data.table(df)
# Rename variable
dt[, `:=`(
  Usage.Week = as.Date(Usage.Week),
  week = week(Usage.Week),
  poverty = factor(poverty, ordered = TRUE, exclude = c("")),
  income = factor(income, ordered = TRUE, exclude = c("")),
  charter.school = ifelse(charter.school == "Yes", 1, ifelse(charter.school == "No", 0, NA)),
  school.account = ifelse(school.account == "Yes", 1, ifelse(school.account == "No", 0, NA)),
  # Log Transform
  Minutes.per.Active.User = log(Minutes.per.Active.User + 1),
  Badges.per.Active.User = log(Badges.per.Active.User + 1),
  Tower.Alerts.per.Tower.Completion = log(Tower.Alerts.per.Tower.Completion + 1),
  User.Session = log(User.Session + 1),
  tch_min = log(tch_min + 1)
)]
# Create new variables using data.table syntax
dt[, min_week := week(min(Usage.Week)),
   by = Classroom.ID]
dt[, `:=`(
  week = ifelse(week >= min_week, week - min_week + 1, week - min_week + 53),
  n_weeks = .N,
  mean_act_st = mean(Active.Users...Total)
), by = .(Classroom.ID, Teacher.User.ID)]
dt[, `:=`(
  st_login = ifelse(Minutes.per.Active.User > 0, 1, 0),
  tch_login = ifelse(tch_min > 0, 1, 0)
), by = .(Classroom.ID, Teacher.User.ID, week)]
# Update the Grade.Level values and labels
dt <- dt[!(Grade.Level %in% c(-1, 11))] # Ignore -1 and 11
dt[, Grade.Level := factor(Grade.Level,
                           ordered = TRUE,
                           exclude = c(""))]
dt[, Grade.Level := factor(Grade.Level,
                           levels = c(0:8),
                           labels = c("Kindergarten", "1st", "2nd",
                                      "3rd", "4th", "5th",
                                      "6th", "7th", "8th"))]
dt[, Tsubj := max(week), by = Teacher.User.ID]
dt <- dt[
  n_weeks > 11 &
    Tsubj < 3*n_weeks &
    teacher_number_classes < 5 &
    Students...Total > 5 &
    mean_act_st > 3 &
    !(Grade.Level %in% c("6th","7th","8th")) &
    !(month(Usage.Week) %in% c(6, 7, 8)),
]

## PCA
df <- as.data.frame(dt)
df_pca <- df %>%
  select(c("tch_min",
           "User.Session",
           "RD.elementary_schedule":"RD.grade_level_teacher_materials")) %>%
  mutate(across(everything(), ~ifelse(is.na(.), 0, .)))
pca <- prcomp(df_pca,
              center = TRUE,
              scale. = TRUE)
df$pca1 <- pca$x[,1]
df$pca2 <- pca$x[,2]
df$pca3 <- pca$x[,3]

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
columns <- c("tch_min", "pca1", "pca2", "pca3")
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
