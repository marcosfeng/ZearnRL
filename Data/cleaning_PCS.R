#-----Load necessary libraries and data-----:
library(data.table)

# Import Data
classroom_student_usage <- fread("Data/Classroom Student Usage - Time Series 2020-10-09T1616.csv")
classroom_teacher_lookup <- fread("Data/Classroom-Teacher Lookup 2020-10-09T1618.csv")
teacher_usage <- fread("Data/Teacher Usage - Time Series 2020-10-09T1635.csv")
classroom_info <- fread("Data/Classroom Info 2020-10-09T1617.csv")
setnames(teacher_usage, "Adult User ID", "Adult.User.ID")
setnames(classroom_teacher_lookup, "Teacher User ID", "Adult.User.ID")

# Prepare data
teacher_usage[, hour := lubridate::hour(`Usage Time`)]
teacher_usage[, Date := as.Date(`Usage Time`)]

# Sequence of dates for the overall grid expand
date.seq <- seq.Date(min(teacher_usage$Date), max(teacher_usage$Date), by = "day")

# First and Last Log ins by time of day
teacher_first_login <- teacher_usage[, .(first_login = min(hour)), by = .(Adult.User.ID, Date)]
teacher_last_login <- teacher_usage[, .(last_login = max(hour)), by = .(Adult.User.ID, Date)]

teacher_usage <- teacher_usage[, .(logged = 1), by = .(Adult.User.ID, Date)]

df <- CJ(Adult.User.ID = unique(teacher_usage$Adult.User.ID),
         Date = date.seq)
df <- df[teacher_usage, logged := logged, on = .(Adult.User.ID, Date)]
df[is.na(logged), logged := 0]

####################################
df <- df[order(Date)]

#-----Code datetime variables-----:
df[,Month:=as.numeric(month(Date))]
df[,dayofweek:=as.factor(wday(Date))]
df[,time_dow:=1:.N, list(Adult.User.ID, dayofweek)]
df[,time:=1:.N, Adult.User.ID]

#-----Create previous visits/frequency-----:
df[,prev_visits:=cumsum(logged) - logged, Adult.User.ID]
df[,prev_dow_visits:=cumsum(logged) - logged, list(Adult.User.ID, dayofweek)]
df[,prev_freq:=prev_visits/time]
df[,prev_dow_freq:=prev_dow_visits/time_dow]
df[,mean_visits:=mean(logged), Adult.User.ID]

#-----Streak & time_lag-----:
##Streak:
# df <- df[order(time_of_day, Date)]
df <- df[order(Date)]
df[,lag.logged:=lag(logged, 1), Adult.User.ID] 
df[is.na(lag.logged), lag.logged:=0]
df[,streak:=NULL]
df[lag.logged==0, streak:=as.double(0)]
df[,start_streak:=as.numeric(logged == 1 & lag.logged == 0)]
df[,start_streak:=cumsum(start_streak), Adult.User.ID]
df[is.na(streak), streak:=as.double(1:.N), list(Adult.User.ID, start_streak)]
# Weekday Streak
##Streak:
# df <- df[order(time_of_day, Date)]
df <- df[order(Date)]
df[dayofweek != "Sunday" & dayofweek != "Saturday",wk.lag.logged:=lag(logged, 1), Adult.User.ID] 
df[is.na(wk.lag.logged), wk.lag.logged:=0]
df[,wk.streak:=NULL]
df[wk.lag.logged==0 & dayofweek != "Sunday" & dayofweek != "Saturday", wk.streak:=as.double(0)]
df[,start_wk.streak:=as.numeric(logged == 1 & wk.lag.logged == 0)]
df[dayofweek != "Sunday" & dayofweek != "Saturday",start_wk.streak:=cumsum(start_wk.streak), Adult.User.ID]
df[is.na(wk.streak) & dayofweek != "Sunday" & dayofweek != "Saturday", wk.streak:=as.double(1:.N), list(Adult.User.ID, start_wk.streak)]
##Time lag:
df[,time_lag:=NULL]
df[,start_streak_lag:=lag(start_streak, 1), Adult.User.ID]
df[lag.logged==1, time_lag:=1]
df[lag.logged==0, time_lag:=as.double(1:.N)+1, list(Adult.User.ID, start_streak_lag)]
df[is.na(start_streak_lag), time_lag:=NA]
##StreakDow
df[,attended_lag_dow:=lag(logged, 1), list(Adult.User.ID, dayofweek)] 
df[is.na(attended_lag_dow), attended_lag_dow:=0]
df[,streak_dow:=NULL]
df[attended_lag_dow==0, streak_dow:=as.double(0)]
df[,start_streak_dow:=as.numeric(logged == 1 & attended_lag_dow == 0)]
df[,start_streak_dow:=cumsum(start_streak_dow), list(Adult.User.ID, dayofweek)]
df[is.na(streak_dow), streak_dow:=as.double(1:.N), list(Adult.User.ID, start_streak_dow, dayofweek)]

##Prevweek visits:
levels(df$dayofweek) <- c( "Sunday", "Monday", "Tuesday", "Wednesday",
                          "Thursday", "Friday", "Saturday")
df[,dow_num:=as.numeric(dayofweek)]
df[,week_num:=as.numeric(dow_num == 1)]
df = df[order(Adult.User.ID, Date)]
df[,week_num:=cumsum(week_num), Adult.User.ID]
df[,week_visits:=sum(logged), list(Adult.User.ID, week_num)]

week_vs = df[,.(week_visits = sum(logged)), list(Adult.User.ID, week_num)]
week_vs[,prev_week_visits:=lag(week_visits), Adult.User.ID]
week_vs[is.na(prev_week_visits), prev_week_visits:=0]

df = merge(df, week_vs[,-c("week_visits")], by = c("Adult.User.ID", "week_num"), all.x = TRUE)
df[,prev_dow_visit:=lag(logged), list(Adult.User.ID, dow_num)]
df[is.na(prev_dow_visit),prev_dow_visit:=0]
## Previous minutes spent:
df <- df[order(Date)]

df[,prev_min_zearn:=lag(Minutes.on.Zearn...Total), Adult.User.ID]
df[is.na(prev_min_zearn), prev_min_zearn:=0]

# Start with the first logged==1
df = df[order(Date)]
df[,real_time:=seq(1,.N),by=Adult.User.ID]

index = df[, .SD[which.max(logged)], by = Adult.User.ID]

df = merge(df, index[,c("Adult.User.ID","real_time")], by = c("Adult.User.ID"), all.x = TRUE)

df[, first_time:=real_time.x-real_time.y]
df <- df[!(first_time<0)]

# End with the last logged==1
df = df[order(-Date)]
index = df[, .SD[which.max(logged)], by = Adult.User.ID]
df = merge(df, index[,c("Adult.User.ID","first_time")], by = c("Adult.User.ID"), all.x = TRUE)

df[, last_time:=first_time.x-first_time.y]
df <- df[!(last_time>0)]

# Remove the scraps
df[, c("real_time.y","first_time.x", "first_time.y", "last_time"):=NULL]  # remove these columns
df = df[order(Date)]
setnames(df, "real_time.x", "real_time")

# Merge First and Last logins
df <- merge(x = df, y = teacher_first_login, by = c("Adult.User.ID","Date"))
df <- merge(x = df, y = teacher_last_login, by = c("Adult.User.ID","Date"))

save(df, file = "Zearn_PCS_cleaned20210517.RData")





