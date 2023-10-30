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
# Assuming pr_a_s is the probability Pr(a | s)
tidy_data$pr_a_s <- runif(nrow(tidy_data))  # Replace this line with your actual data


base_plot <- ggplot(tidy_data, aes(x = week)) +
  geom_rect(aes(xmin = week - 0.5, xmax = week + 0.5, ymin = 0, ymax = 1, fill = state)) +
  scale_fill_manual(values = c("white", "grey70")) +
  theme_minimal()

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
