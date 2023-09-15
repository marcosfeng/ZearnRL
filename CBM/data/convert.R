library(R.matlab)

# Import
stan_data <- readRDS("~/zearn/Bayesian/stan_data.RDS")

# Convert to a List of Matrices
stan_data$Tsubj <- matrix(stan_data$Tsubj, nrow = length(stan_data$Tsubj), ncol = 1)
stan_data$group <- matrix(stan_data$group, nrow = length(stan_data$group), ncol = 1)
stan_data$number_teachers <- matrix(stan_data$number_teachers, nrow = 1, ncol = 1)

# Write to .mat File
writeMat("./CBM/data/stan_data.mat",
         N = stan_data$N,
         T = stan_data$T,
         S = stan_data$S,
         K = stan_data$K,
         C = stan_data$C,
         Tsubj = stan_data$Tsubj,
         choice = stan_data$choice,
         outcome = stan_data$outcome,
         week = stan_data$week,
         state = stan_data$state,
         group = stan_data$group,
         number_teachers = stan_data$number_teachers)
