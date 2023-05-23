library("rstan") # observe startup messages
library("tidyverse")
library("cmdstanr")

setwd("~/shared_folder")
load("RL_data_Zearn-notscaled20220715.Rdata")
# load("~/shared_folder/LauGlimcher_RL/LauGlimcher20220927.Rdata")
load("subjects.Rdata")
Tsubj  <-  Tsubj[order( Tsubj$Adult.User.ID),]

df <- mfmb_logged %>%
  filter(Adult.User.ID %in% Tsubj$Adult.User.ID) %>%
  group_by(Adult.User.ID) %>%
  #mutate(minutes_lagged = dplyr::lag(Minutes.on.Zearn...Total)) %>%
  mutate(minutes_lagged = lead(Minutes.on.Zearn...Total)) %>%
  # Transforming data so first week of use = 1 for each teacher
  mutate(week_num = week_num - min(week_num) + 1) %>%
  mutate(Tsubj = max(week_num))

# create choice[N,T]
choice <- df %>%
  select(Adult.User.ID, week_num, minutes_lagged) %>%
  mutate(minutes_lagged = ifelse(minutes_lagged == 0, 0, 1)) %>%
  group_by(week_num) %>%
  spread(week_num, minutes_lagged)

# create reward[N,T]
reward <- df %>%
  select(Adult.User.ID, week_num, Badges.per.Active.User) %>%
  group_by(week_num) %>%
  spread(week_num, Badges.per.Active.User)

## Determine State matrix
towers <- df %>%
  select(Adult.User.ID, week_num, Tower.Alerts.per.Tower.Completion) %>%
  group_by(week_num) %>%
  spread(week_num, Tower.Alerts.per.Tower.Completion)
minutes <- df %>%
  select(Adult.User.ID, week_num, Minutes.per.Active.User) %>%
  group_by(week_num) %>%
  spread(week_num, Minutes.per.Active.User)
# Create state variables[N*T, S]
towers <- towers[,c(1:(max(df$week_num) + 1))] %>%
  pivot_longer(cols = c(2:(max(df$week_num) + 1))) %>%
  rename(week_num = name) %>%
  mutate(week_num = as.numeric(week_num))
minutes <- minutes[,c(1:(max(df$week_num) + 1))] %>%
  pivot_longer(cols = c(2:(max(df$week_num) + 1))) %>%
  rename(week_num = name) %>%
  mutate(week_num = as.numeric(week_num))

states <- towers %>%
  rename(tower = value) %>%
  left_join(minutes, by = c("Adult.User.ID", "week_num")) %>%
  rename(minutes = value)

###
# fill in NAs with 1 for now (lowest payoff)
choice[is.na(choice)] <- 0
#Sort to match reward
choice <- choice[order(choice$Adult.User.ID),]
# fill in NAs with 0 for now
reward[is.na(reward)] <- 0
#Sort to match effort
reward <- reward[order(reward$Adult.User.ID),]
# fill in NAs with 0 for now
states[is.na(states)] <- 0
#Sort to match effort
states <- states[order(states$Adult.User.ID),]

#Number of states
S = ncol(states[,-c(1,2)])

model_data <- list( N = nrow(Tsubj), #number of teachers
                    T       = as.integer(max(df$week_num)), # number of weeks total
                    S       = S,
                    Tsubj   = as.integer(unlist(Tsubj[,c(-1)])), # number of weeks each teacher used Zearn
                    choice  = as.matrix(choice[,c(-1)]),
                    outcome = as.matrix(reward[,c(-1)]),
                    states  = as.matrix(states[,-c(1,2)]))

#### Clean everything except model_data
rm(list=setdiff(ls(), "model_data"))

# my_model <- stan_model(file = "RL_AC_traces.stan")
my_model <- stan_model(file = "RL_hierarchical_AC_traces_FAST.stan")
sample <- sampling(object = my_model,
                   data = model_data,
                   iter = 10000,
                   chains = 4,
                   cores = 4)

parameters <- summary(sample)

save(sample, file = "Results_Zearn_AC_Hierarchical20220726.Rdata")


# my_model <- cmdstan_model("RL_hierarchical_AC_traces.stan")
# 
# sample <- my_model$sample(data = model_data,
#                           iter_warmup =1000,
#                           iter_sampling = 1000,
#                           chains = 1,
#                           parallel_chains = 1,
#                           refresh = 100)
# stanfit <- rstan::read_stan_csv(sample$output_files())











