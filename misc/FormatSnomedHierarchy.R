library(magrittr)
library(dplyr)

test <- read.csv("C:/Users/luish/Downloads/CONCEPT_ANCESTOR.csv", sep='\t')

dist1 <- test %>%
  filter(min_levels_of_separation == 1 & max_levels_of_separation == 1) %>%
  mutate(weight = min_levels_of_separation) %>%
  select(c(descendant_concept_id, ancestor_concept_id, weight)) %>%
  rename(id1 = descendant_concept_id, id2 = ancestor_concept_id)
write.csv(dist1, file="C:/Users/luish/Desktop/dist1.csv", row.names=FALSE)

# 443580 - systolic heart failure
# 316139 - heart failure
# 313217 - atrial fibrillation
# 4103183 - cardiac finding
# 441840 - clinical finding
# 4182210 - dementia
# 139900 - urticaria
# 80809 - rheumatoid arthritis
set <- c(443580, 316139, 313217, 4103183, 441840, 4182210, 139900, 80809)
output <- test %>%
  filter(min_levels_of_separation != 0 & max_levels_of_separation != 0) %>%
  filter(descendant_concept_id %in% set) %>%
  filter(ancestor_concept_id %in% set)

data_ordered <- output[order(output$min_levels_of_separation, output$max_levels_of_separation, decreasing = FALSE), ]
data_highest <- data_ordered[!duplicated(data_ordered$descendant_concept_id), ]


# data <- data.frame(ancestor_concept_id = numeric(), descendant_concept_id = numeric(),
#                    min_levels_of_separation = numeric(), max_levels_of_separation = numeric())


write.csv(data_highest, file="C:/Users/luish/Downloads/pars.csv", row.names=FALSE)



# dist2 <- test %>%
#   filter(min_levels_of_separation == 1 & max_levels_of_separation == 1 |
#            min_levels_of_separation == 2 & max_levels_of_separation == 2) %>%
#   mutate(dist = min_levels_of_separation) %>%
#   select(ancestor_concept_id, descendant_concept_id, dist)
# write.csv(dist2, file="~/Desktop/dist2.csv", row.names=FALSE)
#
#
# dist3 <- test %>%
#   filter(min_levels_of_separation == 1 & max_levels_of_separation == 1 |
#            min_levels_of_separation == 2 & max_levels_of_separation == 2 |
#            min_levels_of_separation == 3 & max_levels_of_separation == 3) %>%
#   mutate(dist = min_levels_of_separation) %>%
#   select(ancestor_concept_id, descendant_concept_id, dist)
# write.csv(dist3, file="~/Desktop/dist3.csv", row.names=FALSE)
#
#
# dist4 <- test %>%
#   filter(min_levels_of_separation == 1 & max_levels_of_separation == 1 |
#            min_levels_of_separation == 2 & max_levels_of_separation == 2 |
#            min_levels_of_separation == 3 & max_levels_of_separation == 3 |
#            min_levels_of_separation == 4 & max_levels_of_separation == 4) %>%
#   mutate(dist = min_levels_of_separation) %>%
#   select(ancestor_concept_id, descendant_concept_id, dist)
# write.csv(dist4, file="~/Desktop/dist4.csv", row.names=FALSE)
#
#
# dist5 <- test %>%
#   filter(min_levels_of_separation == 1 & max_levels_of_separation == 1 |
#            min_levels_of_separation == 2 & max_levels_of_separation == 2 |
#            min_levels_of_separation == 3 & max_levels_of_separation == 3 |
#            min_levels_of_separation == 4 & max_levels_of_separation == 4 |
#            min_levels_of_separation == 5 & max_levels_of_separation == 5) %>%
#   mutate(dist = min_levels_of_separation) %>%
#   select(ancestor_concept_id, descendant_concept_id, dist)
# write.csv(dist5, file="~/Desktop/dist5.csv", row.names=FALSE)
#
#
# dist6 <- test %>%
#   filter(min_levels_of_separation == 1 & max_levels_of_separation == 1 |
#            min_levels_of_separation == 2 & max_levels_of_separation == 2 |
#            min_levels_of_separation == 3 & max_levels_of_separation == 3 |
#            min_levels_of_separation == 4 & max_levels_of_separation == 4 |
#            min_levels_of_separation == 5 & max_levels_of_separation == 5 |
#            min_levels_of_separation == 6 & max_levels_of_separation == 6) %>%
#   mutate(dist = min_levels_of_separation) %>%
#   select(ancestor_concept_id, descendant_concept_id, dist)
# write.csv(dist6, file="~/Desktop/dist6.csv", row.names=FALSE)


refOriginal = read.table("C:/Users/luish/Downloads/CONCEPT.csv", sep="\t", quote = "", fill = TRUE, header = TRUE)

ref <- refOriginal %>%
  select(concept_id, concept_name, concept_class_id, standard_concept) %>%
  filter(concept_class_id == "Clinical Finding")
write.csv(ref, file="C:/Users/luish/Desktop/ref.csv", row.names=FALSE)

nSample <- 100 # was 10
delta <- 1
depth <- 10

set.seed(1001)

##########
# temp1 <- dist1 %>%
#   filter(ancestor_concept_id == 316139) %>%
#   left_join(ref, by = c("descendant_concept_id" = "concept_id")) %>%
#   mutate(descendant_concept_name = concept_name) %>%
#   select(-c(concept_class_id, standard_concept, concept_name)) %>%
#   slice_sample(n = nSample)
#########

# Abdominal pain: 21522001
# Clinical finding: 441840
# Cardiovascular finding: 4023995

####   rename(id1 = descendant_concept_id, id2 = ancestor_concept_id)
temp1 <- dist1 %>%
  filter(id2 == 4023995) %>% # 4274025 This is disease, the super parent
  slice_sample(n = nSample)
man <- temp1

for (i in 1:depth) {
  temp2 <- dist1 %>%
    filter(id2 %in% temp1$id1) %>%
    slice_sample(n = nSample*delta)
    # slice_sample(n = nSample*i*delta)

  man <- bind_rows(man, temp2)
  temp1 <- temp2
}
################### same code as above, but no sampling
temp1 <- dist1 %>%
  filter(ancestor_concept_id == 316139)
man <- temp1

for (i in 1:100) {
  temp2 <- dist1 %>%
    filter(ancestor_concept_id %in% temp1$descendant_concept_id)
  # slice_sample(n = nSample*i*delta)

  man <- bind_rows(man, temp2)
  temp1 <- temp2
}

# # Some code to join references onto concept ids
# dist1Sample <- man %>%
#   left_join(ref, by = c("descendant_concept_id" = "concept_id")) %>%
#   mutate(descendant_concept_name = concept_name) %>%
#   select(-c(concept_class_id, standard_concept, concept_name))

dist1Sample <- man %>%
  select(c(id1, id2, weight))
  # rename(id1 = descendant_concept_id, id2 = ancestor_concept_id)

write.csv(dist1Sample, file="~/Desktop/dist1HeartDis.csv", row.names=FALSE, quote = FALSE)


