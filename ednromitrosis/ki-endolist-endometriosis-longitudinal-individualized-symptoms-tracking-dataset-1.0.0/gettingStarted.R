rm(list = ls())
setwd("~/KI/Endometriosis - Documents/Main-Project-EndoPatterns/Papers/Paper-1-Study-design/PhysioNet/Submission_v2/")

library(tidyr)
library(janitor)
library(ggplot2)

# Read onboarding info 
onboarding.file <- "Patient_onboarding_info.csv"
onboarding.dict <- "Onboarding_codes_dictionary.csv"

baseline.info <- read.csv(onboarding.file, check.names = F)
baseline.dict <- read.csv(onboarding.dict, check.names = F)

# Process alternative therapy column
alternative.dict <- baseline.dict[baseline.dict$`Column name` == "Alternative therapy", ]
alternarive.therapies <- baseline.info %>% 
  select(`User id`, `Alternative therapy`) %>%
  filter(nchar(`Alternative therapy`) > 0) %>% 
  separate_longer_delim(cols = "Alternative therapy", delim = ", ") %>% 
  rename(Code = "Alternative therapy") %>% 
  mutate(Description = alternative.dict$Description[match(Code, alternative.dict$Code)])

tabyl(alternarive.therapies$Description) %>% arrange(-n) %>% View()

# Read user data
user59.data <- read.csv("User59.csv", check.names = F)

# Plot headache reported values by cycle number
user59.data <- user59.data %>% filter(!is.na(`Headache (0-no pain, 4-worst pain)`))
ggplot(data = user59.data, aes(x = `Cycle day`, y = `Headache (0-no pain, 4-worst pain)`, colour = `Cycle number`)) + 
  geom_jitter(height = 0.05, width = 0) + 
  scale_color_gradient(low = "blue", high = "red") + 
  theme_bw()