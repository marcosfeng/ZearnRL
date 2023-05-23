# Load necessary libraries
library(dplyr)
library(tidyr)
library(lubridate)

# Importing files
teacher_usage <- read.csv("Data/Teacher Usage - Time Series 2020-10-09T1635.csv")
classroom_student_usage <- read.csv("Data/Classroom Student Usage - Time Series 2020-10-09T1616.csv")
classroom_info <- read.csv("Data/Classroom Info 2020-10-09T1617.csv")
classroom_teacher_lookup <- read.csv("Data/Classroom-Teacher Lookup 2020-10-09T1618.csv")
school_info <- read.csv("Data/School Info 2020-10-09T1619.csv")

# Data cleaning

# Delete classrooms with multiple teachers
classroom_teacher_lookup <- classroom_teacher_lookup %>%
  group_by(Teacher.User.ID) %>%
  mutate(teacher_number_classes = n()) %>%
  ungroup() %>%
  group_by(Classroom.ID) %>%
  filter(n_distinct(Teacher.User.ID) == 1) %>%
  ungroup()

# Feature engineering

# Extract week, year, wday, hour, and whour information for teacher_usage
teacher_usage <- teacher_usage %>%
  mutate(week = week(Usage.Time),
         year = year(Usage.Time),
         wday = wday(Usage.Time),
         hour = hour(Usage.Time),
         whour = (wday - 1) * 24 + hour)

# Summarize teacher behavior per week
teacher_usage_total <- teacher_usage %>%
  mutate(Event.Type = ifelse(Event.Type == "Resource Downloaded",
                             paste("RD.", Curriculum.Resource.Category, sep = ""), Event.Type)) %>%
  group_by(Adult.User.ID, Event.Type, week, year) %>%
  summarise(Freq = n()) %>%
  pivot_wider(id_cols = c(Adult.User.ID, week, year),
              names_from = Event.Type,
              values_from = Freq) %>%
  ungroup() %>%
  mutate(across(everything(), ~ifelse(is.na(.), 0, .)))

teacher_usage_total <- teacher_usage %>%
  group_by(Adult.User.ID, week, year) %>%
  summarise(tch_min = sum(Minutes.on.Zearn...Total)) %>%
  full_join(teacher_usage_total, by = c("Adult.User.ID", "week", "year"))

# Data merging

# Merge classroom_student_usage with classroom_teacher_lookup
df <- classroom_student_usage %>%
  inner_join(classroom_teacher_lookup,
             by = c("Classroom.ID"),
             multiple = "all") %>%
  mutate(week = week(Usage.Week),
         year = year(Usage.Week)) %>%
  right_join(classroom_info, by = "Classroom.ID",
             multiple = "all",
             relationship = "many-to-many")  %>%
  right_join(school_info, by = "MDR.School.ID",
             multiple = "all") %>%
  # Delete classrooms with multiple schools
  group_by(Classroom.ID) %>%
  filter(n_distinct(MDR.School.ID) == 1) %>%
  ungroup()

# Merge df with teacher_usage_total, classroom_info, and school_info
df <- df %>%
  inner_join(teacher_usage_total,
             by = c("Teacher.User.ID" = "Adult.User.ID", "week", "year"),
             multiple = "all") %>%
  select(-c(NCES.ID,
            School.Name,
            District.Rollup.Name,
            School.Address...City,
            School.Address...County,
            School.Address...State)) %>%
  mutate(
    Minutes.per.Active.User = as.numeric(Minutes.per.Active.User),
    Minutes.per.Active.User = if_else(is.na(Minutes.per.Active.User),
                                      0, Minutes.per.Active.User),
    Badges.per.Active.User = if_else(is.na(Badges.per.Active.User),
                                     0, Badges.per.Active.User),
    Tower.Alerts.per.Tower.Completion = if_else(is.na(Tower.Alerts.per.Tower.Completion),
                                                0, Tower.Alerts.per.Tower.Completion),
    Boosts.per.Tower.Completion = if_else(is.na(Boosts.per.Tower.Completion),
                                          0, Boosts.per.Tower.Completion)) %>%
  rename(
    poverty = Demographics...Poverty.Level,
    income = Demographics...Zipcode.Median.Income,
    charter.school = Is.Charter..Yes...No.,
    school.account = MDR.School.has.School.Account..Yes...No.,
    zipcode = School.Address...Zipcode
  )

write_csv(df, "Data/df_clean.csv")
