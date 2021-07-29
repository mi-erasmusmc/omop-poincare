library(magrittr)
library(dplyr)

test <- read.csv("~/Downloads/vocabulary_download_v5_{7f67c39f-b6ae-444e-997e-87a9f6b05e86}_1627400111595/CONCEPT_ANCESTOR.csv", sep='\t')

dist1 <- test %>%
  filter(min_levels_of_separation == 1 & max_levels_of_separation == 1) %>%
  mutate(dist = min_levels_of_separation) %>%
  select(ancestor_concept_id, descendant_concept_id, dist)
write.csv(dist1, file="~/Desktop/dist1.csv", row.names=FALSE)

dist2 <- test %>%
  filter(min_levels_of_separation == 1 & max_levels_of_separation == 1 |
           min_levels_of_separation == 2 & max_levels_of_separation == 2) %>%
  mutate(dist = min_levels_of_separation) %>%
  select(ancestor_concept_id, descendant_concept_id, dist)
write.csv(dist2, file="~/Desktop/dist2.csv", row.names=FALSE)


dist3 <- test %>%
  filter(min_levels_of_separation == 1 & max_levels_of_separation == 1 |
           min_levels_of_separation == 2 & max_levels_of_separation == 2 |
           min_levels_of_separation == 3 & max_levels_of_separation == 3) %>%
  mutate(dist = min_levels_of_separation) %>%
  select(ancestor_concept_id, descendant_concept_id, dist)
write.csv(dist3, file="~/Desktop/dist3.csv", row.names=FALSE)


dist4 <- test %>%
  filter(min_levels_of_separation == 1 & max_levels_of_separation == 1 |
           min_levels_of_separation == 2 & max_levels_of_separation == 2 |
           min_levels_of_separation == 3 & max_levels_of_separation == 3 |
           min_levels_of_separation == 4 & max_levels_of_separation == 4) %>%
  mutate(dist = min_levels_of_separation) %>%
  select(ancestor_concept_id, descendant_concept_id, dist)
write.csv(dist4, file="~/Desktop/dist4.csv", row.names=FALSE)


dist5 <- test %>%
  filter(min_levels_of_separation == 1 & max_levels_of_separation == 1 |
           min_levels_of_separation == 2 & max_levels_of_separation == 2 |
           min_levels_of_separation == 3 & max_levels_of_separation == 3 |
           min_levels_of_separation == 4 & max_levels_of_separation == 4 |
           min_levels_of_separation == 5 & max_levels_of_separation == 5) %>%
  mutate(dist = min_levels_of_separation) %>%
  select(ancestor_concept_id, descendant_concept_id, dist)
write.csv(dist5, file="~/Desktop/dist5.csv", row.names=FALSE)


dist6 <- test %>%
  filter(min_levels_of_separation == 1 & max_levels_of_separation == 1 |
           min_levels_of_separation == 2 & max_levels_of_separation == 2 |
           min_levels_of_separation == 3 & max_levels_of_separation == 3 |
           min_levels_of_separation == 4 & max_levels_of_separation == 4 |
           min_levels_of_separation == 5 & max_levels_of_separation == 5 |
           min_levels_of_separation == 6 & max_levels_of_separation == 6) %>%
  mutate(dist = min_levels_of_separation) %>%
  select(ancestor_concept_id, descendant_concept_id, dist)
write.csv(dist6, file="~/Desktop/dist6.csv", row.names=FALSE)


refOriginal = read.table("~/Downloads/vocabulary_download_v5_{7f67c39f-b6ae-444e-997e-87a9f6b05e86}_1627400111595/CONCEPT.csv", sep="\t", quote = "", fill = TRUE, header = TRUE)

ref <- refOriginal %>%
  select(concept_id, concept_name, concept_class_id, standard_concept) %>%
  filter(concept_class_id == "Clinical Finding")

write.csv(ref, file="~/Desktop/ref.csv", row.names=FALSE)

nSample <- 10
delta <- 1
depth <- 6

set.seed(1000)

temp1 <- dist1 %>%
  filter(ancestor_concept_id == 4274025) %>% # This is disease, the super parent
  slice_sample(n = nSample)
man <- temp1

for (i in 1:depth) {
  temp2 <- dist1 %>%
    filter(ancestor_concept_id %in% temp1$descendant_concept_id) %>%
    slice_sample(n = nSample*i*delta)

  man <- bind_rows(man, temp2)
  temp1 <- temp2
}

# Some code to join references onto concept ids
dist1Sample <- man %>%
  left_join(ref, by = c("descendant_concept_id" = "concept_id")) %>%
  mutate(descendant_concept_name = concept_name) %>%
  select(-c(concept_class_id, standard_concept, concept_name))
