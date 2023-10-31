
# LOOIC -------------------------------------------------------------------
# https://www.statology.org/negative-aic/
library(loo)
options(mc.cores = 4)
log_lik_rl <- loo::loo(test)
loo_rl <- loo(log_lik_rl)

qlearn_sum <- `Q-learning-FR`$summary()
qstate_sum <- `Q-learning-states-FR`$summary()
logit_sum <- Logit$summary()

# Assume data1, data2, and data3 are your data frames
# Let's bind them into one data frame with an additional column to indicate the model
all_data <- bind_rows(
  mutate(qlearn_sum, model = "Q-learning"),
  mutate(qstate_sum, model = "State Q-learning"),
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


looic <- list(`Q-learning-FR`$loo(), `Q-learning-states-FR`$loo(), Logit$loo())

# Extracting the LOOIC and SE values
looic_values <- c(
  looic[[1]][["estimates"]]["looic",],
  looic[[2]][["estimates"]]["looic",],
  looic[[3]][["estimates"]]["looic",]
)

models <- c("Q-learning", "State Q-learning", "Logit")

# Combining the extracted values into a data frame
looic_data <- data.frame(
  Model = factor(c("Q-learning", "State Q-learning", "Logit")),
  LOOIC = c(
    looic[[1]][["estimates"]]["looic","Estimate"],
    looic[[2]][["estimates"]]["looic","Estimate"],
    looic[[3]][["estimates"]]["looic","Estimate"]
  ),
  SE = c(
    looic[[1]][["estimates"]]["looic","SE"],
    looic[[2]][["estimates"]]["looic","SE"],
    looic[[3]][["estimates"]]["looic","SE"]
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


fit <- readRDS("~/zearn/Bayesian/Results/Q-learning-FR.RDS")



prediction_hierarchical <- fit$summary() %>%
  filter(grepl("y_pred", variable)) %>%
  dplyr::select(variable, mean)

test <- prediction_hierarchical %>%
  mutate(variable = str_extract(variable, "\\[.*\\]"),
         variable = str_replace_all(variable, "\\[|\\]", "")) %>%
  separate(variable, into = c("dim1", "dim2", "dim3"), sep = ",", convert = TRUE)

prediction_hierarchical_3d <- array(dim = c(max(prediction_hierarchical$dim1),
                                            max(prediction_hierarchical$dim2),
                                            max(prediction_hierarchical$dim3)))

for (i in 1:nrow(prediction_hierarchical)) {
  dim1 <- prediction_hierarchical$dim1[i]
  dim2 <- prediction_hierarchical$dim2[i]
  dim3 <- prediction_hierarchical$dim3[i]
  prediction_hierarchical_3d[dim1, dim2, dim3] <- prediction_hierarchical$mean[i]
}

# Loop through each subject
for (subject in seq_len(dim1)) {
  # Get the value of Tsubj for this subject
  Tsubj_value <- stan_data[["Tsubj"]][subject]

  # Set the values in prediction_hierarchical_3d to NA
  if (Tsubj_value != max(stan_data[["Tsubj"]])) {
    prediction_hierarchical_3d[subject, (Tsubj_value + 1):dim2, ] <- NA
    # Set the values in choice_data to NA
    choice_data[subject, (Tsubj_value + 1):dim2, ] <- NA
  }
}

# Get the number of layers
num_layers <- dim(prediction_hierarchical_3d)[3]
# Custom function to calculate standard error based on the number of non-NA elements
calc_se <- function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))

df_compare <- data.frame()

# Generate a plot for each layer
for (k in 1:num_layers) {
  y_pred_avg <- apply(prediction_hierarchical_3d[, , k], 2, mean, na.rm = TRUE)
  y_pred_se  <- apply(prediction_hierarchical_3d[, , k], 2, calc_se)

  choice_data_avg <- apply(choice_data[, , k], 2, mean, na.rm = TRUE)
  choice_data_se  <- apply(choice_data[, , k], 2, calc_se)

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
  theme_bw()

print(p)



# Examples ----------------------------------------------------------------

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



qkernel_sum <-
