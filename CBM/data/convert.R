library(R.matlab)

# Assuming stan_data is your existing data list
stan_data <- readRDS("~/zearn/Bayesian/stan_data.RDS")

# Initialize an empty list to hold each subject's data
data_list <- vector("list", length = stan_data$N)

# Populate the list
for (i in 1:stan_data$N) {
  subj_data <- list(
    actions = stan_data$choice[i, 1:stan_data$Tsubj[i], ],
    outcome = stan_data$outcome[i, 1:stan_data$Tsubj[i]],
    state = stan_data$state[i, 1:stan_data$Tsubj[i]]
  )
  data_list[[i]] <- subj_data
}

# Save as a .mat file
writeMat("all_data.mat", data = data_list)
