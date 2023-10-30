# Actions
df_melted <- melt(df, measure.vars = c("FrobeniusNNDSVD1", "FrobeniusNNDSVD2", "FrobeniusNNDSVD3"))
# Change the variable names to Component 1, Component 2, Component 3 for better labeling
df_melted$variable <- factor(df_melted$variable,
                             levels = c("FrobeniusNNDSVD1", "FrobeniusNNDSVD2", "FrobeniusNNDSVD3"),
                             labels = c("Component 1", "Component 2", "Component 3"))

# Tower Alerts
# df_melted <- melt(df, measure.vars = c("Tower.Alerts.per.Tower.Completion"))
# # Change the variable names to Tower Alerts for better labeling
# df_melted$variable <- factor(df_melted$variable,
#                              levels = c("Tower.Alerts.per.Tower.Completion"),
#                              labels = c("Tower Alerts"))

# Calculate the median for each variable
medians <- aggregate(value ~ variable, data = df_melted, FUN = median)

# Filter the data so that for each variable, only data up to twice the median is included
df_filtered <- df_melted %>%
  group_by(variable) %>%
  mutate(value = if_else(value > 3*median(value, na.rm = TRUE),
                         3*median(value, na.rm = TRUE),
                         value))

# Define custom label function for scientific notation with 2 decimal places
label_sci_2 <- function(x) {
  parse(text = scales::scientific_format(digits = 2)(x))
}

# Now create the plot
p <- ggplot(df_filtered, aes(x = value, fill = variable)) +
  geom_histogram(bins = 20, alpha = 0.5, position = 'identity') +
  geom_vline(data = medians, aes(xintercept = value, color = variable), linetype="dashed", linewidth=1) +
  facet_wrap(~ variable, scales = 'free', nrow = 3) +
  scale_x_continuous(labels = label_sci_2) +
  theme_minimal() +
  theme(legend.position = "none") +  # Removed legends
  labs(x = "Value", y = "Frequency")  # Removed fill and color legends

# Print the plot
print(p)

