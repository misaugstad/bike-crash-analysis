library(tidyverse)
library(dplyr)

crash.data <- read.csv("data/merged_crash_data.csv", header = TRUE, sep = ",")
crash.data[["incidence"]][is.na(crash.data[["incidence"]])] <- 0
crash.data <- crash.data %>% rename(total_population = B00001e1,
                                    housing_units = B00002e1,
                                    house_income = B19001e1,
                                    accidents = incidence)
fit <- glm(accidents ~ total_population + housing_units + house_income,
           data = crash.data,
           family = gaussian())
summary(fit)

ggplot(crash.data, aes(x = total_population, y = accidents)) +
  stat_summary(fun.y = "mean", geom = "bar") +
  theme_bw()
