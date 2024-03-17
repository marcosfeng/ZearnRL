# Load the R.matlab library
library(tidyverse)
library(data.table)
library(R.matlab)

# Declare Functions and Variables -----------------------------------------

prepare_choice_array <- function(df, cols) {
  # Create zero-filled array with appropriate dimensions
  classroom_array <- unique(df$Classroom.ID)
  choice_array <- array(0, dim = c(length(classroom_array),
                                   max(df$row_n),
                                   length(cols)))
  # Populate array
  for (i in 1:length(classroom_array)) {
    n <- nrow(df[Classroom.ID == classroom_array[i]])
    choice_array[i,1:n,] <- as.matrix(
      df[Classroom.ID == classroom_array[i], ..cols]
    )
  }

  return(choice_array)
}

prepare_variable_array <- function(df, all_vars) {
  # Create zero-filled array with appropriate dimensions
  classroom_array <- unique(df$Classroom.ID)
  variable_array <- array(0, dim = c(length(classroom_array),
                                     max(df$row_n),
                                     length(all_vars)))
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

df <- read.csv(file = "Bayesian/df.csv") %>%
  mutate(across(where(is.numeric),
                ~ ifelse(. < .Machine$double.eps, 0, .))) %>%
  mutate(across(dplyr::starts_with("Frobenius.NNDSVD_teacher"),
                ~ if_else(.>median(.),1,0)))

FR_cols <- grep("Frobenius.NNDSVD_teacher", names(df), value = TRUE)

# List of all potential state and reward variables
# all_vars <- c("Active.Users...Total", "Minutes.per.Active.User", "Badges.per.Active.User",
#               "Boosts.per.Tower.Completion", "Tower.Alerts.per.Tower.Completion")
all_vars <- grep("Frobenius.NNDSVD_student", names(df), value = TRUE)

df <- df %>%
  arrange(Classroom.ID, week) %>%
  as.data.table() %>%
  .[Classroom.ID %in%
      sample(unique(Classroom.ID),
             size = length(unique(Classroom.ID)) * 0.10)
    ] %>%
  setorder(Classroom.ID, week) %>%
  .[, row_n := seq_len(.N), by = .(Classroom.ID)] %>%
  # Filter out Classroom.IDs with sd = 0 for Scaffolding
  .[, .SD[!apply(.SD[, FR_cols[2], with = FALSE], 2, sd) == 0],
    by = Classroom.ID] %>%
  # Filter out Classroom.IDs with sd = 0 for Activities
  .[, .SD[!apply(.SD[, FR_cols[3], with = FALSE], 2, sd) == 0],
    by = Classroom.ID]

stan_data <- list(
  N = length(unique(df$Classroom.ID)), # Number of teachers
  Tsubj = df[, .N, by = .(Classroom.ID)][,N], # Number of rows by Teacher
  choice = prepare_choice_array(df, FR_cols),
  C = length(FR_cols),
  all_vars = prepare_variable_array(df, all_vars),
  week = as.matrix(dcast(df,
                         Classroom.ID ~ row_n,
                         value.var = "week"))[,c(-1)],
  number_teachers = length(unique(df$Teacher.User.ID))
)

# Save as a .mat file for each subject
for (i in 1:stan_data$N) {
  subj_data <- convert_to_subj_struct(i, stan_data)
  R.matlab::writeMat(paste0("CBM/data/individual/", "/subj_", i, ".mat"),
                     NNDSVD_teacher1 = subj_data$actions[,1],
                     NNDSVD_teacher2 = subj_data$actions[,2],
                     NNDSVD_teacher3 = subj_data$actions[,3],
                     NNDSVD_teacher4 = subj_data$actions[,4],
                     NNDSVD_student1 = subj_data$all_vars[,1],
                     NNDSVD_student2 = subj_data$all_vars[,2],
                     NNDSVD_student3 = subj_data$all_vars[,3],
                     NNDSVD_student4 = subj_data$all_vars[,4],
                     simmed = subj_data$simmed)
}
