options(mc.cores = 12)

`Q-learning-FR` <- readRDS("./Bayesian/Results/Q-learning-FR.RDS")
`Q-learning-states-FR` <- readRDS("./Bayesian/Results/Q-learning-states-FR.RDS")
`Actor-Critic-FR` <- readRDS("./Bayesian/Results/Actor-Critic-FR.RDS")
logit <- readRDS("./Bayesian/Results/logit.RDS")
kernel <- readRDS("./Bayesian/Results/Q-learning-kernel-FR.RDS")

qlearn_sum <- `Q-learning-FR`$summary()
qstate_sum <- `Q-learning-states-FR`$summary()
ac_sum <- `Actor-Critic-FR`$summary()
logit_sum <- logit$summary()
kernel_sum <- kernel$summary()

# LOOIC -------------------------------------------------------------------
# https://www.statology.org/negative-aic/

# Let's bind one data frame with an additional column to indicate the model
all_data <- bind_rows(
  mutate(qlearn_sum, model = "Q-learning"),
  mutate(qstate_sum, model = "State Q"),
  mutate(ac_sum, model = "Actor Critic"),
  mutate(logit_sum, model = "Logit")
)

# Filter out only the rows of interest (log_lik[1] to log_lik[210])
all_data <- filter(all_data, grepl("log_lik", variable))

# Assuming 'all_data' dataframe and 'mean' column are available
ggplot(all_data, aes(x = mean, color = model)) +
  geom_density(adjust = 1, alpha = 0.7) + # Adjusted size and added alpha for a smoother look
  labs(
    title = "Log Likelihood per Classrooms",
    x = "Log Likelihood",
    y = "Density",
    color = "Model" # Legend title
  ) +
  theme_minimal() + # Increased base font size
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5), # Centered and bolded title
    legend.position = "bottom", # Moved legend to top
    legend.box = "horizontal", # Arranged legend items horizontally
    legend.title = element_text(face = "bold") # Bolded legend title
  ) +
  scale_color_brewer(palette = "Set1") # Changed color palette to something more visually appealing


looic <- list(`Q-learning-FR`$loo(),
              `Q-learning-states-FR`$loo(),
              `Actor-Critic-FR`$loo(),
              logit$loo())

models <- c("Q-learning", "State Q-learning", "Actor-Critic", "Logit")

# Combining the extracted values into a data frame
looic_data <- data.frame(
  Model = factor(c("Q-learning", "State Q-learning", "Actor-Critic", "Logit")),
  LOOIC = c(
    looic[[1]][["estimates"]]["looic","Estimate"],
    looic[[2]][["estimates"]]["looic","Estimate"],
    looic[[3]][["estimates"]]["looic","Estimate"],
    looic[[4]][["estimates"]]["looic","Estimate"]
  ),
  SE = c(
    looic[[1]][["estimates"]]["looic","SE"],
    looic[[2]][["estimates"]]["looic","SE"],
    looic[[3]][["estimates"]]["looic","SE"],
    looic[[4]][["estimates"]]["looic","SE"]
  )
)

# Plotting
ggplot(looic_data, aes(x = Model, y = LOOIC, color = Model)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = LOOIC - SE, ymax = LOOIC + SE), width = 0.2) +
  labs(
    title = "Model Performance",
    y = "LOOIC",
    color = "Model"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "none"
  )



# Time series -------------------------------------------------------------

process_and_plot_model <- function(model_sum, model_name, stan_data) {
  prediction <- model_sum %>%
    filter(grepl("y_pred|choice_prob", variable)) %>%
    dplyr::select(variable, mean)

  prediction <- prediction %>%
    mutate(variable = str_extract(variable, "\\[.*\\]"),
           variable = str_replace_all(variable, "\\[|\\]", "")) %>%
    separate(variable, into = c("dim1", "dim2", "dim3"), sep = ",", convert = TRUE)

  prediction_3d <- array(dim = c(max(prediction$dim1),
                                 max(prediction$dim2),
                                 max(prediction$dim3)))

  for (i in 1:nrow(prediction)) {
    dim1 <- prediction$dim1[i]
    dim2 <- prediction$dim2[i]
    dim3 <- prediction$dim3[i]
    prediction_3d[dim1, dim2, dim3] <- prediction$mean[i]
  }

  if (grepl("logit", model_name, ignore.case = TRUE)) {
    # Initialize counts
    counts <- rep(0, stan_data[["C"]])

    choice_array <- array(dim = c(max(prediction$dim1),
                                  max(prediction$dim2),
                                  max(prediction$dim3)))
    for (n in 1:stan_data[["N"]]) {
      # Increment counts based on classroom occurrences
      counts[stan_data[["classroom"]][n]] <- counts[stan_data[["classroom"]][n]] + 1
      choice_array[stan_data[["classroom"]][n],
                   counts[stan_data[["classroom"]][n]], ] <-
        stan_data$y[n,]
    }

    # Create Tsubj array
    stan_data[["Tsubj"]] <- counts

  } else {
    choice_array <- stan_data$choice
  }

  # Loop through each subject
  for (subject in seq_len(dim1)) {
    # Get the value of Tsubj for this subject
    Tsubj_value <- stan_data[["Tsubj"]][subject]

    # Set the values in prediction_3d to NA
    if (Tsubj_value != max(stan_data[["Tsubj"]])) {
      prediction_3d[subject, (Tsubj_value + 1):dim2, ] <- NA
      # Set the values in choice_data to NA
      choice_array[subject, (Tsubj_value + 1):dim2, ] <- NA
    }
  }

  # Get the number of layers
  num_layers <- dim(prediction_3d)[3]
  # Custom function to calculate standard error based on the number of non-NA elements
  calc_se <- function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))

  df_compare <- data.frame()

  # Generate a plot for each layer
  for (k in 1:num_layers) {
    y_pred_avg <- apply(prediction_3d[, , k], 2, mean, na.rm = TRUE)
    y_pred_se  <- apply(prediction_3d[, , k], 2, calc_se)

    choice_data_avg <- apply(choice_array[, , k], 2, mean, na.rm = TRUE)
    choice_data_se  <- apply(choice_array[, , k], 2, calc_se)

    # Only consider weeks with valid SEs
    weeks <- seq_len(sum(!is.na(y_pred_se)))

    df_pred <- data.frame(weeks = weeks,
                          probability = y_pred_avg[weeks],
                          type = rep("Model Fit", max(weeks)),
                          se = y_pred_se[weeks],
                          action = rep(paste("Action", k), max(weeks)))

    df_real <- data.frame(weeks = weeks,
                          probability = choice_data_avg[weeks],
                          type = rep("Real Data", max(weeks)),
                          se = choice_data_se[weeks],
                          action = rep(paste("Action", k), max(weeks)))

    df_compare <- rbind(df_compare, df_pred, df_real)
  }

  p <- ggplot(df_compare, aes(x = weeks, y = probability, color = type)) +
    geom_line() +
    geom_ribbon(data = df_compare, aes(ymin = probability - se, ymax = probability + se, fill = type), alpha = 0.1) +
    labs(x = "Week", y = "Probability of a=1") +
    facet_wrap(~action, ncol = 1) +
    scale_color_manual(values = c("Model Fit" = "blue", "Real Data" = "red")) +
    scale_fill_manual(values = c("Model Fit" = "blue", "Real Data" = "red")) +
    theme_minimal() +
    theme(legend.position = "none")

  print(p)
}

q_data <- readRDS("./Bayesian/Q-learn-data.RDS")
process_and_plot_model(qlearn_sum, "Q-learning", q_data)
process_and_plot_model(qstate_sum, "Q-state", q_data)

ac_data <- readRDS("./Bayesian/Actor-Critic-data.RDS")
process_and_plot_model(ac_sum, "Actor-Critic", ac_data)

logit_data <- readRDS("./Bayesian/Logit-data.RDS")
process_and_plot_model(logit_sum, "Logit", logit_data)



# Kernel ------------------------------------------------------------------

# Let's bind one data frame with an additional column to indicate the model
all_data <- bind_rows(
  mutate(ac_sum, model = "Actor Critic"),
  mutate(logit_sum, model = "Logit"),
  mutate(kernel_sum, model = "Kernel Q"),
  mutate(qlearn_sum, model = "Q-learning")
)

# Filter out only the rows of interest (log_lik[1] to log_lik[210])
all_data <- filter(all_data, grepl("log_lik", variable))

# Assuming 'all_data' dataframe and 'mean' column are available
ggplot(all_data, aes(x = mean, color = model)) +
  geom_density(adjust = 1, alpha = 0.7) + # Adjusted size and added alpha for a smoother look
  labs(
    title = "Log Likelihood per Classrooms",
    x = "Log Likelihood",
    y = "Density",
    color = "Model" # Legend title
  ) +
  theme_minimal() + # Increased base font size
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5), # Centered and bolded title
    legend.position = "bottom", # Moved legend to top
    legend.box = "horizontal", # Arranged legend items horizontally
    legend.title = element_text(face = "bold") # Bolded legend title
  ) +
  scale_color_brewer(palette = "Set1") # Changed color palette to something more visually appealing


looic <- list(`Actor-Critic-FR`$loo(),
              logit$loo(),
              kernel$loo(),
              `Q-learning-FR`$loo())

models <- c("Actor-Critic", "Logit", "Kernel", "State Q-learning")

# Combining the extracted values into a data frame
looic_data <- data.frame(
  Model = factor(c("Actor-Critic", "Logit", "Kernel", "State Q-learning")),
  LOOIC = c(
    looic[[1]][["estimates"]]["looic","Estimate"],
    looic[[2]][["estimates"]]["looic","Estimate"],
    looic[[3]][["estimates"]]["looic","Estimate"],
    looic[[4]][["estimates"]]["looic","Estimate"]
  ),
  SE = c(
    looic[[1]][["estimates"]]["looic","SE"],
    looic[[2]][["estimates"]]["looic","SE"],
    looic[[3]][["estimates"]]["looic","SE"],
    looic[[4]][["estimates"]]["looic","SE"]
  )
)

# Plotting
ggplot(looic_data, aes(x = Model, y = LOOIC, color = Model)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = LOOIC - SE, ymax = LOOIC + SE), width = 0.2) +
  labs(
    title = "Model Performance",
    y = "LOOIC",
    color = "Model"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "none"
  )


####

process_and_plot_model(kernel_sum, "Kernel Q", q_data)


# Examples ----------------------------------------------------------------

best_fit <- function(user, data, prediction_3d, choice = 1, range = c(1:8)) {
  success <- data$choice[user,range,choice] * prediction_3d[user,range,choice]
  failures <- (1 - data$choice[user,range,choice]) * (1 - prediction_3d[user,range,choice])
  return(sum(success+failures))
}
lapply(list(101, 202, 205, 86, 7),FUN = best_fit, data = ac_data, prediction_3d = prediction_3d)

# Top best fit: 202, 159, 153
df <- df %>%
  mutate(classroom = as.integer(factor(Classroom.ID)))
participant_data <- df[df$classroom == 159, ]

# Convert data to a tidy format
tidy_data <- participant_data %>%
  select(week, state, FrobeniusNNDSVD1bin, Badges.per.Active.User) %>%
  gather(key = "variable", value = "value", -week, -state) %>%
  filter(!is.na(state))

# Convert state to a factor for shading
tidy_data$state <- as.factor(tidy_data$state)

base_plot <- ggplot(tidy_data, aes(x = week)) +
  geom_rect(aes(xmin = week - 0.5, xmax = week + 0.5, ymin = 0, ymax = 1, fill = state)) +
  scale_fill_manual(values = c("white", "grey70")) +
  theme_minimal()

# Extract parameter values from qstate_sum
params <- qstate_sum[2:13,1:2]
# Initialize Q-values
Q <- matrix(unlist(params[7:12,2]), nrow = 3, ncol = 2, byrow = TRUE)

# Initialize a dataframe to store Pr(a|s) values
pr_data <- data.frame()
tau <- as.numeric(params[params$variable == "tau","mean"])
alpha <- as.numeric(params[params$variable == "alpha","mean"])
gamma <- as.numeric(params[params$variable == "gamma","mean"])
cost1 <- as.numeric(params[params$variable == "cost[1]","mean"])
cost2 <- as.numeric(params[params$variable == "cost[2]","mean"])
cost3 <- as.numeric(params[params$variable == "cost[3]","mean"])
# Loop through the participant data
for (i in 2:nrow(participant_data)) {
  # Get current state, action, and reward
  s <- participant_data$state[i]
  a <- participant_data$KullbackLeibler1bin[i] + 1
  r <- participant_data$Badges.per.Active.User[i]

  # Calculate Pr(a|s) using softmax function
  expQ <- exp(Q / tau)
  pr <- expQ[s, a]

  # Store Pr(a|s) value
  pr_data <- rbind(pr_data, data.frame(week = participant_data$week[i],
                                       state = s,
                                       action = a,
                                       pr_a_s = pr))

  # Get next state (assuming states transition in a deterministic manner)
  s_next <- participant_data$state[i+1]

  # Update Q-value using Q-learning update rule
  Q[s, a] <- Q[s, a] + alpha * (gamma * (r - cost1) - Q[s, a])
}






# Now add the time series and dots to the base plot
plot <- base_plot +
  # geom_line(data = tidy_data, aes(y = pr_a_s, color = "Pr(a | s)"), size = 1) +
  geom_point(data = tidy_data, aes(y = ifelse(value == 1, 1, 0)), size = 1) +
  labs(y = "", color = "Variable") +
  theme(legend.position = "bottom")




Q_values <- matrix(runif(2 * 4), nrow = 2, ncol = 4)  # Placeholder, replace with actual Q-values

# Function to compute Pr(a | s) using softmax
compute_pr_a_s <- function(Q_values, tau) {
  exp_Q_values <- exp(Q_values / tau)
  sum_exp_Q_values <- rowSums(exp_Q_values)
  pr_a_s <- exp_Q_values / matrix(rep(sum_exp_Q_values, each = ncol(Q_values)), nrow = nrow(Q_values), byrow = TRUE)
  return(pr_a_s)
}

# Assuming tau is one of the estimated parameters, replace with the actual value
tau <- 0.8241379  # Placeholder, replace with actual value

# Compute Pr(a | s)
pr_a_s <- compute_pr_a_s(Q_values, tau)

# Now pr_a_s contains the probabilities Pr(a | s) for each state-action pair



# Assuming q_values is your matrix of Q-values
q_values <- matrix(runif(4), nrow = 1, ncol = 4)  # Replace this line with your actual data

# Create a new data frame for Q-values
q_data <- data.frame(
  state_action_pair = factor(1:4, labels = c("s1-a1", "s1-a2", "s2-a1", "s2-a2")),
  q_value = c(q_values)
)

# Create a new plot for Q-values
q_plot <- ggplot(q_data, aes(x = state_action_pair, y = q_value)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(y = "Q-value", x = "")

# Combine the two plots using the patchwork library
library(patchwork)

combined_plot <- q_plot / plot

print(combined_plot)


print(plot)





# Matlab model analysis ---------------------------------------------------

library(R.matlab)
library(PerformanceAnalytics)

# Read the .mat file
mlab_models <- readMat("CBM/zearn_models/hbi_model_compare.mat")
mlab_data <- readMat("CBM/data/filtered_data.mat")

# Accessing the data
# cbm.output.responsibility
responsibility <- round(mlab_models[["cbm"]][[5]][[2]][,2:3], 2)
responsibility[is.infinite(responsibility)] <- 0

# cbm.output.parameters for the kernel model
kernel_params <- mlab_models[["cbm"]][[5]][[1]][[2]][[1]]
# cbm.output.parameters for the actor-critic model
ac_params     <- mlab_models[["cbm"]][[5]][[1]][[3]][[1]]

## Correlations: Fit, parameter estimate, outcome.
mean_minutes <- sapply(mlab_data[["filtered.data"]], function(subj) mean(subj[[1]][[3]]))
# Indices of subjects for which kernel model has higher responsibility
kernel_indices <- which(responsibility[,1] > responsibility[,2])

# Indices of subjects for which actor-critic model has higher responsibility
ac_indices <- which(responsibility[,2] > responsibility[,1])

# Extract parameters and mean minutes for these subjects
kernel_params_selected <- kernel_params[kernel_indices, ]
ac_params_selected <- ac_params[ac_indices, ]

kernel_mean_minutes <- mean_minutes[kernel_indices]
ac_mean_minutes <- mean_minutes[ac_indices]

# Correlogram for Kernel Model
colnames(kernel_params_selected) <- c("alpha", "gamma", "tau", "c1", "c2", "c3")
data_kernel <- data.frame(kernel_params_selected, Minutes = kernel_mean_minutes)
colnames(ac_params_selected) <- c("alpha_w", "alpha_theta", "gamma", "tau",
                                  "w_0", "theta_0",
                                  "c1", "c2", "c3")
corr_data <- data.frame(ac_params_selected, Minutes = ac_mean_minutes)
# Apply the logistic function to alpha and gamma
corr_data$alpha_w <- 1 / (1 + exp(-corr_data$alpha_w))
corr_data$alpha_theta <- 1 / (1 + exp(-corr_data$alpha_theta))
corr_data$gamma <- 1 / (1 + exp(-corr_data$gamma))
# Apply the exponential function to tau, c1, c2, and c3
corr_data$tau <- exp(corr_data$tau)
corr_data$c1 <- exp(corr_data$c1)
corr_data$c2 <- exp(corr_data$c2)
corr_data$c3 <- exp(corr_data$c3)
chart.Correlation(corr_data, histogram=TRUE, method = "spearman", pch=19)


## Timelines: output vs. prediction

simulate_actor_critic <- function(parameters, subj_data) {
  # Extract parameters
  alpha_w <- 1 / (1 + exp(-parameters[1]))  # Learning rate for w (critic)
  alpha_theta <- 1 / (1 + exp(-parameters[2]))  # Learning rate for theta (actor)
  gamma <- 1 / (1 + exp(-parameters[3]))  # Discount factor
  tau <- exp(parameters[4])  # Inverse temperature for softmax
  cost <- exp(parameters[7:length(parameters)])  # Cost for each action

  # Initialize variables
  Tsubj <- nrow(subj_data)
  choice <- subj_data[,1:3]
  outcome <- subj_data$minutes
  state <- cbind(rep(1, Tsubj),
                 subj_data$state.alerts,
                 subj_data$state.boosts)  # Adding column of ones for intercept
  week <- subj_data$week

  C <- ncol(choice)  # Number of choices
  D <- ncol(state)  # Dimensionality of state space

  theta_init <- matrix(rep(parameters[5], D*C), nrow = D, ncol = C)
  w_init     <- matrix(rep(parameters[6], D*C), nrow = D, ncol = C)
  w <- array(NA, c(D, C, Tsubj))  # Critic's state-action value estimates
  w[,,1] = w_init
  theta <- array(NA, c(D, C, Tsubj))  # Actor's policy parameters
  theta[,,1] = theta_init
  p <- matrix(rep(NA, Tsubj*C), nrow = Tsubj, ncol = C) # Probabilities of choices

  # Loop through trials
  for (t in 1:Tsubj) {
    if (week[t] == 0) break

    s <- state[t, ]
    a <- choice[t, ]
    o <- outcome[t]

    # Actor: Compute policy (log probability of taking each action)
    product <- s %*% theta[,,t] * tau
    p[t, ] = 1 / (1 + exp(-product))

    # Critic: Compute TD error (delta)
    if (t < Tsubj) {
      w_t_next <- week[t + 1]
      s_next <- state[t + 1, ]
      PE <- gamma^(w_t_next - week[t]) * (s_next %*% w[,,t]) - (s %*% w[,,t])
    } else {
      PE <- 0  # Terminal state
    }
    delta <- (o - cost) + PE

    if (t == Tsubj | week[t+1] == 0) break
    # Update weights
    theta[,,t+1] <- theta[,,t] + alpha_theta * gamma^week[t] *
      (tau * t(t(s))) %*% ((1 + exp(product))^(-1) * delta)
    w[,,t+1] <- w[,,t] + alpha_w * t(t(s)) %*% delta
  }

  return(list(probabilities = p,
              theta = theta, w = w,
              state = state))
}

simulate_kernel_q_learning <- function(parameters, subj_data) {
  # Extract parameters
  alpha <- 1 / (1 + exp(-parameters[1]))
  gamma <- 1 / (1 + exp(-parameters[2]))
  tau <- exp(parameters[3])
  cost <- exp(parameters[4:length(parameters)])  # Cost for each action

  Tsubj <- nrow(subj_data)
  choice <- subj_data$choice
  outcome <- subj_data$outcome
  state <- (subj_data$alerts >= -0.1402) + 1  # Binarize state based on median
  K <- 4

  C <- length(cost)  # Number of choices
  S <- max(state)  # Number of states
  ev <- matrix(0, nrow = C, ncol = S)  # Expected value (Q-value)
  p <- matrix(rep(NA, Tsubj*C), nrow = Tsubj, ncol = C) # Probabilities of choices

  # Loop through trials
  for (t in 1:Tsubj) {
    a <- choice[t, ]
    o <- outcome[t]
    s <- state[t]

    # Kernel reward calculation
    for (j in 1:C) {
      # Compute log probability of the chosen action
      logit_val <- tau * ev[j, s]
      p[t, j] = 1 / (1 + exp(-logit_val))

      if (a[j] == 1) {
        ker_reward <- 0
        ker_norm <- 0
        for (t_past in 1:min(t - 1, K)) {
          jaccard_sim <- 0
          # Check if the action and state are the same in the past
          if (choice[t - t_past, j] == a[j]) {
            jaccard_sim <- 0.5
            if (state[t - t_past] == s) {
              jaccard_sim <- jaccard_sim + 0.5
            }
            # Calculate kernelized reward
            ker_reward <- ker_reward +
              gamma^(1 + t - (t - t_past)) *
              jaccard_sim * outcome[t - t_past]
            ker_norm <- ker_norm + jaccard_sim
          }
        }
        if (ker_norm > 0) {
          ker_reward <- ker_reward / ker_norm
        } else {
          ker_reward <- 0
        }

        # Update expected value (ev) if choice was made
        delta <- (ker_reward - cost[j]) - ev[j, s]
        ev[j, s] <- ev[j, s] + (alpha * delta)
      }
    }
  }

  return(list(probabilities = p,
              ev = ev,
              state = state))
}

# Loop through each subject and simulate
model_predictions <- list()
ac_estimates <- list()
kernel_estimates <- list()
for (subj in 1:length(mlab_data[["filtered.data"]])) {
  subj_data <- data.frame(actions = mlab_data[["filtered.data"]][[subj]][[1]][[1]],
                          minutes = mlab_data[["filtered.data"]][[subj]][[1]][[3]],
                          state = data.frame(alerts = mlab_data[["filtered.data"]][[subj]][[1]][[6]],
                                             boosts = mlab_data[["filtered.data"]][[subj]][[1]][[5]]),
                          week = mlab_data[["filtered.data"]][[subj]][[1]][[7]][[1]])

  ac_estimates[[subj]] <- NA
  kernel_estimates[[subj]] <- NA
  if (responsibility[subj, 1] > responsibility[subj, 2]) {
    # Use kernel model
    params <- kernel_params_selected[which(kernel_indices == subj), ]
    sim_result <- simulate_kernel_q_learning(params, subj_data)
    kernel_estimates[[subj]] <- list(ev = sim_result$ev,
                                     state = sim_result$state)
  } else if (responsibility[subj, 1] < responsibility[subj, 2]) {
    # Use actor-critic model
    params <- ac_params_selected[which(ac_indices == subj), ]
    sim_result <- simulate_actor_critic(params, subj_data)
    ac_estimates[[subj]] <- list(theta = sim_result$theta,
                                 w = sim_result$w,
                                 state = sim_result$state)
  } else {
    model_predictions[[subj]] <- NA
    next
  }
  model_predictions[[subj]] <- sim_result$probabilities
}

# Creating df of model predictions
filtered_data <- model_predictions[!sapply(model_predictions, anyNA)]
unlisted_data <- do.call(rbind, filtered_data)
# Create a data frame with the values and their indices
df <- as.data.frame(unlisted_data)
df$Subj <- rep(seq_along(filtered_data), sapply(filtered_data, nrow))
df$t <- unlist(sapply(sapply(filtered_data, nrow),seq_len))
df <- df %>% pivot_longer(cols = c(1:3),
                          names_prefix = "V",
                          names_to = "c", values_to = "mean",
                          names_transform = list(c = as.integer))
# Create the 'variable' column
df$variable <- apply(df[, c("Subj", "t", "c")], 1,
                     function(x) paste("y_pred[",
                                       paste(x, collapse = ", "),
                                       "]", sep = ""))
# Reshape the data frame to only have the 'variable' and 'value' columns
final_df <- df %>%
  select(variable, mean)


# Preparing data for process_and_plot_model
stan_data <- lapply(mlab_data[["filtered.data"]],
                    function(subj) subj[[1]][[1]])
stan_data <- stan_data[!sapply(model_predictions, anyNA)]
# 1. Create the Tsubj Vector
Tsubj <- sapply(stan_data, nrow)
# 2. Create the choice 3D Array
max_rows <- max(Tsubj)
max_cols <- max(sapply(stan_data, ncol))
# Initialize an array
choice <- array(NA, dim = c(length(stan_data), max_rows, max_cols))
# Fill the array
for (i in seq_along(stan_data)) {
  rows <- nrow(stan_data[[i]])
  cols <- ncol(stan_data[[i]])
  choice[i, 1:rows, 1:cols] <- stan_data[[i]]
}
# Combine into a list
stan_data_final <- list(Tsubj = Tsubj, choice = choice)

# Running process_and_plot_model
process_and_plot_model(model_sum = final_df,
                       model_name = "Your Model Name",
                       stan_data = stan_data_final)


## Correlations: estimated values and policies as a function of state and reward
# Initialize an empty data frame for all subjects
all_subjects_data <- data.frame()
# Loop through each subject
for (subj in 1:length(model_predictions)) {
  # Skip if any NA values are present in estimates for this subject
  if (anyNA(ac_estimates[[subj]])) next

  # Extract data for this subject
  Prob <- as.vector(t(model_predictions[[subj]]))
  W1 <- as.vector(ac_estimates[[subj]]$w[1,,])
  W2 <- as.vector(ac_estimates[[subj]]$w[2,,])
  W3 <- as.vector(ac_estimates[[subj]]$w[3,,])
  Theta1 <- as.vector(ac_estimates[[subj]]$theta[1,,])
  Theta2 <- as.vector(ac_estimates[[subj]]$theta[2,,])
  Theta3 <- as.vector(ac_estimates[[subj]]$theta[3,,])
  State <- as.vector(t(ac_estimates[[subj]]$state))
  Minutes <- as.vector(t(cbind(
    mlab_data[["filtered.data"]][[subj]][[1]][[3]],
    mlab_data[["filtered.data"]][[subj]][[1]][[3]],
    mlab_data[["filtered.data"]][[subj]][[1]][[3]])))

  # Calculate State Value for each time point t
  State_Value <- as.vector(
    sapply(1:nrow(ac_estimates[[subj]]$state),
           function(t) ac_estimates[[subj]]$state[t,] %*%
             ac_estimates[[subj]]$w[, ,t]))

  # Combine into a data frame
  subj_data <- data.frame(Prob, W1, W2, W3, Theta1, Theta2, Theta3,
                          State, State_Value, Minutes)

  # Append to the master data frame
  all_subjects_data <- rbind(all_subjects_data, subj_data)
}

all_subjects_data <- data.frame()
# Loop through each subject
for (subj in 1:length(model_predictions)) {
  # Skip if any NA values are present in estimates for this subject
  if (anyNA(ac_estimates[[subj]])) next

  # Extract data for this subject
  Prob <- as.vector(t(model_predictions[[subj]]))
  W1 <- as.vector(ac_estimates[[subj]]$w[1,,])
  W2 <- as.vector(ac_estimates[[subj]]$w[2,,])
  W3 <- as.vector(ac_estimates[[subj]]$w[3,,])
  Theta1 <- as.vector(ac_estimates[[subj]]$theta[1,,])
  Theta2 <- as.vector(ac_estimates[[subj]]$theta[2,,])
  Theta3 <- as.vector(ac_estimates[[subj]]$theta[3,,])
  State <- as.vector(t(ac_estimates[[subj]]$state))
  Minutes <- as.vector(t(cbind(
    mlab_data[["filtered.data"]][[subj]][[1]][[3]],
    mlab_data[["filtered.data"]][[subj]][[1]][[3]],
    mlab_data[["filtered.data"]][[subj]][[1]][[3]])))

  # Calculate State Value for each time point t
  State_Value <- as.vector(
    sapply(1:nrow(ac_estimates[[subj]]$state),
           function(t) ac_estimates[[subj]]$state[t,] %*%
             ac_estimates[[subj]]$w[, ,t]))

  # Combine into a data frame
  subj_data <- data.frame(Prob, W1, W2, W3, Theta1, Theta2, Theta3,
                          State, State_Value, Minutes)

  # Append to the master data frame
  all_subjects_data <- rbind(all_subjects_data, subj_data)
}

chart.Correlation(all_subjects_data, histogram=TRUE,
                  method = "spearman", pch=19)


