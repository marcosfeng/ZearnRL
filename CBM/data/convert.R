# Load the R.matlab library
library(tidyverse)
library(data.table)
library(R.matlab)
set.seed(37909890)
# © 1998-2023 RANDOM.ORG
# Timestamp: 2023-10-04 18:38:28 UTC

# Declare Functions and Variables -----------------------------------------

remove_holiday_weeks <- function(data) {
  # Define holiday weeks (Thanksgiving and two weeks of Christmas)
  holiday_weeks <- c("2019-11-25", "2019-12-23", "2019-12-30")

  # Filter out holiday weeks
  data %>%
    filter(!date %in% holiday_weeks)
}

remove_inactive_periods <- function(data) {
  active_months <- data %>%
    mutate(month = floor_date(as.Date(date), "month")) %>%
    group_by(Classroom.ID, month) %>%
    summarise(monthly_activity = sum(rowSums(
      across(starts_with("Frobenius.NNDSVD_teacher"), ~ . > 0)
    )) > 0, .groups = "drop") %>%
    filter(monthly_activity)

  data %>%
    mutate(month = floor_date(as.Date(date), "month")) %>%
    semi_join(active_months, by = c("Classroom.ID", "month")) %>%
    select(-month)
}

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
      number_teachers = stan_data$number_teachers,
      ID = stan_data$ID[i]
    )
  )
}

# Import Data -------------------------------------------------------------

minimum_weeks <- 12
df <- read.csv(file = "./Data/df.csv") %>%
  mutate(across(where(is.numeric),
                ~ ifelse(. < .Machine$double.eps, 0, .))) %>%
  mutate(date = as.Date(date, format = "%Y-%m-%d") - 7) %>%
  remove_inactive_periods() %>%
  remove_holiday_weeks() %>%
  group_by(Classroom.ID) %>%
  filter(if_any(dplyr::starts_with("Frobenius.NNDSVD_teacher"),
                ~ sum(.>0) > minimum_weeks/2)) %>%
  # Filter out Classroom.IDs with less than 12 total rows of data
  filter(n() >= minimum_weeks) %>%
  ungroup() %>%
  mutate(across(dplyr::starts_with("Frobenius.NNDSVD_teacher"),
                ~ if_else(.>0,1,0)))

FR_cols <- grep("Frobenius.NNDSVD_teacher", names(df), value = TRUE)

# List of all potential state and reward variables
# all_vars <- c("Active.Users...Total", "Minutes.per.Active.User", "Badges.per.Active.User",
#               "Boosts.per.Tower.Completion", "Tower.Alerts.per.Tower.Completion")
all_vars <- grep("Frobenius.NNDSVD_student", names(df), value = TRUE)

df <- df %>%
  arrange(Classroom.ID, week) %>%
  as.data.table()
setorder(df, Classroom.ID, week)
# # Filter out Classroom.IDs with sd = 0 for Choices
# df <- df[, .SD[!any(apply(.SD[, FR_cols, with = FALSE], 2, sd) == 0)],
#          by = Classroom.ID]
# Filter out Classroom.IDs with sd = 0 for Scaffolding
df <- df[, .SD[!apply(.SD[, FR_cols[2], with = FALSE], 2, sd) == 0],
         by = Classroom.ID]
# Filter out Classroom.IDs with sd = 0 for Activities
df <- df[, .SD[!apply(.SD[, FR_cols[3], with = FALSE], 2, sd) == 0],
         by = Classroom.ID]
# Label the weeks
df[, row_n := seq_len(.N), by = .(Classroom.ID)]
# Sample 10% of the data
df <- df[Classroom.ID %in%
           sample(unique(Classroom.ID),
                  size = length(unique(Classroom.ID)) * 0.10)]

stan_data <- list(
  N = length(unique(df$Classroom.ID)), # Number of teachers
  Tsubj = df[, .N, by = .(Classroom.ID)][,N], # Number of rows by Teacher
  choice = prepare_choice_array(df, FR_cols),
  C = length(FR_cols),
  all_vars = prepare_variable_array(df, all_vars),
  week = as.matrix(dcast(df,
                         Classroom.ID ~ row_n,
                         value.var = "week"))[,c(-1)],
  number_teachers = length(unique(df$Teacher.User.ID)),
  ID = unique(df$Classroom.ID)
)
# save(stan_data, file = "CBM/data/stan_data.RData")

# Save as a .mat file for each subject
for (i in 1:stan_data$N) {
  subj_data <- convert_to_subj_struct(i, stan_data)
  R.matlab::writeMat(paste0("CBM/data/individual_all/", "/subj_", i, ".mat"),
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


# # Define the directory containing the .mat files
# data_dir <- "CBM/data/individual"
# # List all .mat files in the directory
# mat_files <- list.files(data_dir, pattern = "\\.mat$", full.names = TRUE)
# # Initialize a list to hold matched Classroom.IDs
# matched_classroom_ids <- vector("list", length(mat_files))
# # Loop through each .mat file
# for (i in seq_along(mat_files)) {
#   # Read the .mat file
#   mat_data <- readMat(mat_files[i])
#
#   # Find the matching Classroom.ID based on the unique identifier
#   for (j in 1:stan_data$N) {
#     if (stan_data$Tsubj[j] == length(mat_data[[1]])) {
#       if (all(stan_data$choice[j,1:stan_data$Tsubj[j],1] == mat_data[[1]]) &
#           all(stan_data$choice[j,1:stan_data$Tsubj[j],2] == mat_data[[2]]) &
#           all(stan_data$choice[j,1:stan_data$Tsubj[j],3] == mat_data[[3]]) &
#           all(stan_data$choice[j,1:stan_data$Tsubj[j],4] == mat_data[[4]]) &
#           all(stan_data$all_vars[j,1:stan_data$Tsubj[j],1] == mat_data[[5]]) &
#           all(stan_data$all_vars[j,1:stan_data$Tsubj[j],2] == mat_data[[6]]) &
#           all(stan_data$all_vars[j,1:stan_data$Tsubj[j],3] == mat_data[[7]]) &
#           all(stan_data$all_vars[j,1:stan_data$Tsubj[j],4] == mat_data[[8]])) {
#         matched_classroom_ids[[i]] <- stan_data$ID[j]
#         break
#       }
#     }
#   }
# }
# classrooms <- data.frame(Classroom.ID = unlist(matched_classroom_ids))
# save(classrooms, file = "CBM/data/classrooms.RData")
