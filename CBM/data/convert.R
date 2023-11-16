# Load the R.matlab library
library(tidyverse)
library(data.table)
library(R.matlab)

# Declare Functions and Variables -----------------------------------------

prepare_choice_array <- function(df, col1, col2, col3) {
  # Create zero-filled array with appropriate dimensions
  classroom_array <- unique(df$Classroom.ID)
  choice_array <- array(0, dim = c(length(classroom_array), max(df$row_n), 3))
  # Populate array
  for (i in 1:length(classroom_array)) {
    n <- nrow(df[Classroom.ID == classroom_array[i]])
    choice_array[i,1:n,1] <- unlist(df[Classroom.ID == classroom_array[i], ..col1])
    choice_array[i,1:n,2] <- unlist(df[Classroom.ID == classroom_array[i], ..col2])
    choice_array[i,1:n,3] <- unlist(df[Classroom.ID == classroom_array[i], ..col3])
  }

  return(choice_array)
}

prepare_variable_array <- function(df, all_vars) {
  # Create zero-filled array with appropriate dimensions
  classroom_array <- unique(df$Classroom.ID)
  variable_array <- array(0, dim = c(length(classroom_array), max(df$row_n), length(all_vars)))

  # Populate array
  for (i in 1:length(classroom_array)) {
    for (j in 1:length(all_vars)) {
      # Subset the data frame for the specific Classroom.ID
      subset_df <- df[df$Classroom.ID == classroom_array[i], ]

      # Get the number of rows for this Classroom.ID
      n <- nrow(subset_df)

      # Populate the array
      variable_array[i, 1:n, j] <- as.numeric(subset_df[[all_vars[j]]])
    }
  }
  return(variable_array)
}


# function to convert a single subject's data to the desired structure
convert_to_subj_struct <- function(i, stan_data) {
  list(
    actions = matrix(stan_data$choice[i, c(1:stan_data$Tsubj[i]), ],
                     nrow = stan_data$Tsubj[i], ncol = stan_data$C),
    all_vars = matrix(stan_data$all_vars[i, c(1:stan_data$Tsubj[i]),],
                      nrow = stan_data$Tsubj[i], ncol = dim(stan_data$all_vars)[3]),
    simmed = list(
      week = matrix(stan_data$week[i, c(1:stan_data$Tsubj[i])], nrow = stan_data$Tsubj[i]),
      # group = stan_data$group[i],
      number_teachers = stan_data$number_teachers
    )
  )
}

# Import Data -------------------------------------------------------------

df <- read.csv(file = "Bayesian/df.csv")

FR_cols <- grep("FrobeniusNNDSVD", names(df), value = TRUE)

df <- df %>%
  arrange(Classroom.ID, week) %>%
  group_by(MDR.School.ID) %>%
  mutate(across(all_of(FR_cols), ~ifelse(. > median(.), 1, 0), .names = "{.col}bin")) %>%
  as.data.table() %>%
  .[Classroom.ID %in% sample(unique(Classroom.ID), size = length(unique(Classroom.ID)) * 0.10)] %>%
  setorder(Classroom.ID, week) %>%
  .[, row_n := seq_len(.N), by = .(Classroom.ID)]

# List of all potential state and reward variables
all_vars <- c("Active.Users...Total", "Minutes.per.Active.User", "Badges.per.Active.User",
              "Boosts.per.Tower.Completion", "Tower.Alerts.per.Tower.Completion")

# Normalize the variables in 'all_vars'
df <- df %>%
  mutate(across(all_of(all_vars), scale))

# List of all potential action variables
choices <- list(
  FR = grep("FrobeniusNNDSVD(.)bin", names(df), value = TRUE)
)

for (choice in names(choices)) {
  print(paste0("Processing choice array: ", choice))

  stan_data <- list(
    N = length(unique(df$Classroom.ID)), # Number of teachers
    Tsubj = df[, .N, by = .(Classroom.ID)][,N], # Number of rows by Teacher
    choice = prepare_choice_array(df,
                                  choices[[choice]][1],
                                  choices[[choice]][2],
                                  choices[[choice]][3]),
    C = length(choices[[choice]]),
    all_vars = prepare_variable_array(df, all_vars),
    week = as.matrix(dcast(df,
                           Classroom.ID ~ row_n,
                           value.var = "week"))[,c(-1)],
    number_teachers = length(unique(df$Teacher.User.ID))
  )

  # Save the full data as a .mat file
  saveRDS(stan_data, "CBM/data/full_stan_data.RDS")
}

# Save as a .mat file for each subject
for (i in 1:stan_data$N) {
  subj_data <- convert_to_subj_struct(i, stan_data)
  R.matlab::writeMat(paste0("CBM/data/individual/", "/subj_", i, ".mat"),
                     actions = subj_data$actions,
                     activest = subj_data$all_vars[,1],
                     minutes = subj_data$all_vars[,2],
                     badges = subj_data$all_vars[,3],
                     boosts = subj_data$all_vars[,4],
                     alerts = subj_data$all_vars[,5],
                     simmed = subj_data$simmed)
}

