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
  mutate(kernel_sum, model = "Kernel Q")
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

models <- c("Actor-Critic", "Logit", "Kernel Q-learning", "State Q-learning")

# Combining the extracted values into a data frame
looic_data <- data.frame(
  Model = factor(c("Actor-Critic", "Logit", "Kernel Q-learning", "State Q-learning")),
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

