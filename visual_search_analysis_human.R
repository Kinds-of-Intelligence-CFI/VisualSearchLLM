## Visual search analysis

library("fixest");library("rstatix");library("lmerTest");library("tidyverse"); 
library("ggplot2");library("data.table"); library("marginaleffects"); 
library("margins");library("modelsummary");library("parameters");library("readr");
library("lme4"); library("lfe");library("psych")

setwd("C:/Users/jep84/Documents/projects/visual-search/")

##### Experiment 1: Two Among Five #####
df <- read_csv("humanResults/e1_numbers_processed.csv")
df$PID <- factor(df$PID)
df$version <- factor(df$shape_type, levels = c('5','2'))
df$condition <- factor(df$colour_type, levels = c('Efficient disjunctive','Inefficient disjunctive','Conjunctive'))
df$quadrant <- factor(df$quadrant, levels = c('Quadrant 1','Quadrant 2','Quadrant 3','Quadrant 4'))
df$distractor_bin <- factor(df$distractor_bin, levels = c('1–4','5–8','9–16','17–32','33–64','65–99'))
df$accuracy <- as.numeric(df$accuracy)

df_clean <- df %>% drop_na(accuracy)

ANOVA_ACC <- anova_test(
  data = df_clean, 
  dv = accuracy, 
  wid = PID,
  within = c(condition, version, quadrant, distractor_bin),
  effect.size = "pes"
)
get_anova_table(ANOVA_ACC, correction = c("auto"))

Describe_ACC <- describeBy(
  df$accuracy,
  group = list(df$condition, df$quadrant),
  mat = TRUE,
  digits = 3
)
Describe_ACC

post_hoc_distractor <- df_clean %>%
  group_by(condition) %>%
  pairwise_t_test(
    accuracy ~ distractor_bin, 
    paired = TRUE, 
    p.adjust.method = "bonferroni"
  )

post_hoc_condition <- df_clean %>%
  group_by(distractor_bin) %>%
  pairwise_t_test(
    accuracy ~ condition, 
    paired = TRUE, 
    p.adjust.method = "bonferroni"
  )

post_hoc_condition <- df_clean %>%
  pairwise_t_test(
    accuracy ~ condition, 
    paired = TRUE, 
    p.adjust.method = "bonferroni"
  )

post_hoc_quadrant <- df_clean %>%
  group_by(condition) %>%
  pairwise_t_test(
    accuracy ~ quadrant, 
    paired = TRUE, 
    p.adjust.method = "bonferroni"
  )

##### Experiment 2: Light Priors #####
df <- read_csv("humanResults/e2_light_priors_processed.csv")
df$PID <- factor(df$PID)
df$condition <- factor(df$gradient_type, levels = c('vertical','horizontal'))
df$inversion <- factor(df$inversion_type, levels = c('original','inverted'))
df$quadrant <- factor(df$quadrant, levels = c('Quadrant 1','Quadrant 2','Quadrant 3','Quadrant 4'))
df$distractor_bin <- factor(df$distractor_bin, levels = c('1–4','5–8','9–12','13–16','17–20','21–24','25-32','33-49'))
df$accuracy <- as.numeric(df$accuracy)

df_clean <- df %>% drop_na(accuracy)

ANOVA_ACC <- anova_test(
  data = df_clean, 
  dv = accuracy, 
  wid = PID,
  within = c(condition, inversion, quadrant, distractor_bin),
  effect.size = "pes"
)
get_anova_table(ANOVA_ACC, correction = c("auto"))

Describe_ACC <- describeBy(
  df$accuracy,
  group = list(df$quadrant),
  mat = TRUE,
  digits = 3
)
Describe_ACC

post_hoc_inversion <- df_clean %>%
  group_by(condition) %>%
  pairwise_t_test(
    accuracy ~ quadrant, 
    paired = TRUE, 
    p.adjust.method = "bonferroni"
  )

##### Experiment 3: Circle Sizes #####
df <- read_csv("humanResults/e3_circle_sizes_processed.csv")
df$PID <- factor(df$PID)
df$condition <- factor(df$target_size, levels = c('small','medium','large'))
df$quadrant <- factor(df$quadrant, levels = c('Quadrant 1','Quadrant 2','Quadrant 3','Quadrant 4'))
df$distractor_bin <- factor(df$distractor_bin, levels = c('1–4','5–8','9–12','13–16','17–20','21–24',
                                                          '25-28','29-32','33-36','37-40','41-44','45-49'))
df$accuracy <- as.numeric(df$accuracy)

df_clean <- df %>% drop_na(accuracy)

ANOVA_ACC <- anova_test(
  data = df_clean, 
  dv = accuracy, 
  wid = PID,
  within = c(condition, quadrant, distractor_bin),
  effect.size = "pes"
)
get_anova_table(ANOVA_ACC, correction = c("auto"))

post_hoc_condition <- df_clean %>%
  pairwise_t_test(
    accuracy ~ condition, 
    paired = TRUE, 
    p.adjust.method = "bonferroni"
  )

Describe_ACC <- describeBy(
  df$accuracy,
  group = list(df$condition),
  mat = TRUE,
  digits = 3
)
Describe_ACC

post_hoc_distractor <- df_clean %>%
  group_by(condition) %>%
  pairwise_t_test(
    accuracy ~ distractor_bin, 
    paired = TRUE, 
    p.adjust.method = "bonferroni"
  )
