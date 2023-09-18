# Load the R.matlab library
library(R.matlab)

# Assuming stan_data is your existing data list
stan_data <- readRDS("~/zearn/Bayesian/stan_data.RDS")

# function to convert a single subject's data to the desired structure
convert_to_subj_struct <- function(i, stan_data) {
  list(
    actions = matrix(stan_data$choice[i, , ], nrow = stan_data$T, ncol = stan_data$C),
    outcome = matrix(stan_data$outcome[i, ], nrow = stan_data$T),
    simmed = list(
      week = matrix(stan_data$week[i, ], nrow = stan_data$T),
      state = matrix(stan_data$state[i, ], nrow = stan_data$T),
      group = stan_data$group[i],
      number_teachers = stan_data$number_teachers
    )
  )
}

# Save as a .mat file
for (i in 1:stan_data$N) {
  subj_data <- convert_to_subj_struct(i, stan_data)
  R.matlab::writeMat(paste0("CBM/data/individual/subj_", i, ".mat"),
                     actions = subj_data$actions,
                     outcome = subj_data$outcome,
                     simmed = subj_data$simmed)
}

# test <- R.matlab::readMat("CBM/example_RL_task/all_data.mat")
