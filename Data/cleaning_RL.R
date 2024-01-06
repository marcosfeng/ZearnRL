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


# Feature engineering

teacher_usage <- teacher_usage %>%
  mutate(
    # Noting week, and year to summarize teacher behavior:
    week = lubridate::isoweek(Usage.Time),
    year = lubridate::year(Usage.Time),
    # Transforming Event.Type for "Resource Downloaded"
    Event.Type = ifelse(Event.Type == "Resource Downloaded",
                        paste0("RD.", Curriculum.Resource.Category),
                        Event.Type)
  ) %>%
  mutate(year = ifelse(week == 1 & year == 2019, 2020, year))

# Summarize teacher behavior per week
teacher_usage_total <- teacher_usage %>%
  count(Adult.User.ID, Event.Type, week, year) %>%
  pivot_wider(names_from = Event.Type, values_from = n,
              values_fill = list(n = 0)) %>%
  full_join(teacher_usage %>%
              group_by(Adult.User.ID, week, year) %>%
              summarize(Minutes.on.Zearn...Total = sum(Minutes.on.Zearn...Total)),
            by = c("Adult.User.ID", "week", "year"))

# Data cleaning

# Classroom exclusion
classroom_teacher_lookup <- classroom_teacher_lookup %>%
  group_by(Teacher.User.ID) %>%
  mutate(teacher_number_classes = n()) %>%
  ungroup() %>%
  group_by(Classroom.ID) %>%
  # Delete classrooms with multiple teachers
  filter(n_distinct(Teacher.User.ID) == 1) %>%
  ungroup() %>%
  inner_join(classroom_info %>%
               group_by(Classroom.ID) %>%
               # Delete classrooms with multiple schools
               filter(n_distinct(MDR.School.ID) == 1) %>%
               ungroup(), by = "Classroom.ID")

# Data merging
classroom_student_usage <- classroom_student_usage %>%
  inner_join(classroom_teacher_lookup %>%
               select(Classroom.ID, Teacher.User.ID),
             by = c("Classroom.ID")) %>%
  mutate(week = lubridate::isoweek(Usage.Week),
         year = lubridate::year(Usage.Week)) %>%
  mutate(year = ifelse(week == 1 & year == 2019, 2020, year))

df <- teacher_usage_total %>%
  filter(Adult.User.ID %in% unique(classroom_teacher_lookup$Teacher.User.ID)) %>%
  left_join(classroom_teacher_lookup,
            by = c("Adult.User.ID" = "Teacher.User.ID"),
            relationship = "many-to-many") %>%
  full_join(classroom_student_usage,
            by = c("Classroom.ID", "Adult.User.ID" = "Teacher.User.ID",
                   "week", "year"))

# Merge school info
df <- df %>%
  left_join(school_info %>%
              select(MDR.School.ID, District.Rollup.ID,
                     Demographics...Zipcode.Median.Income,
                     Demographics...Poverty.Level,
                     Is.Charter..Yes...No.,
                     MDR.School.has.School.Account..Yes...No.,
                     School.Address...Zipcode),
            by = "MDR.School.ID") %>%
  rename(
    poverty = Demographics...Poverty.Level,
    income = Demographics...Zipcode.Median.Income,
    charter.school = Is.Charter..Yes...No.,
    school.account = MDR.School.has.School.Account..Yes...No.,
    zipcode = School.Address...Zipcode
  ) %>%
  mutate(charter.school = case_when(charter.school == "No" ~ 0,
                                    charter.school == "Yes" ~ 1,
                                    .default = NA),
         school.account = case_when(school.account == "No" ~ 0,
                                    school.account == "Yes" ~ 1,
                                    .default = NA)) %>%
  rename(Teacher.User.ID = Adult.User.ID)

write.csv(df, "Data/df_clean.csv")
