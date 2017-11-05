library(tidyverse)
library(dplyr)

crash.data <- read.csv("data/merged_crash_data-11-4-17.csv", header = TRUE, sep = ",")
crash.data[["incidence"]][is.na(crash.data[["incidence"]])] <- 0
fit <- glm(incidence ~ total_population + housing_units + house_income, data = crash.data, family = gaussian())
summary(fit)

ggplot(crash.data, aes(x = total_population, y = incidence)) +
  stat_summary(fun.y = "mean", geom = "bar") +
  theme_bw()
