library(tidyverse)
library(dplyr)

dat <- read.csv("data/merged_crash_data-11-4-17.csv", header = TRUE, sep = ",")
dat[["incidence"]][is.na(dat[["incidence"]])] <- 0
fit <- glm(incidence ~ total_population + housing_units + house_income, data = dat, family = gaussian())
summary(fit)

# ggplot(data, aes(x = total_population, y = incidence)) +
#   stat_summary(fun.y = "mean", geom = "bar")
