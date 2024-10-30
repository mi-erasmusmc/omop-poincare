library(magrittr)
library(dplyr)

test <- read.csv("D:/git/KnowledgeGraph/data/CONCEPT_ANCESTOR.csv", sep='\t')

# conceptsFromModel <- read.csv("~/Desktop/ohdsi-dlc-download/models/strategusOutput_ehr/DeepPatientLevelPredictionModule_3/models/folder-1-1/covariateImportance.csv")

data <- PatientLevelPrediction::loadPlpData("D:/git/KnowledgeGraph/data/targetId_11454_L1/")

set <- data$covariateData$covariateRef %>%
  collect() %>%
  filter(analysisId == 102) %>%
  unlist()

####
# set <- conceptsFromModel %>%
#   filter(analysisId == 102) %>% # only from condition table
#   select(conceptId) %>%
#   # sample_n(10) %>%
#   unlist()

################################################################################

clinical_finding_concept_id <- 441840
set <- c(set, clinical_finding_concept_id)

# ensure no concepts in set are invalid, this will give false as there is no ancestry available
# data <- c(data_output_internal$ancestor_concept_id, data_output_internal$descendant_concept_id)
# set %in% data

################################################################################

output_internal <- test %>%
  filter(min_levels_of_separation != 0 & max_levels_of_separation != 0) %>% # remove the self-reference, we do not need it
  filter(descendant_concept_id %in% set)

set <- c(output_internal$ancestor_concept_id, output_internal$descendant_concept_id)
set <- unique(set)
######

# another ancestor exists within the set
output_internal <- test %>%
  filter(min_levels_of_separation != 0 & max_levels_of_separation != 0) %>% # remove the self-reference, we do not need it
  filter(descendant_concept_id %in% set) %>% # descendant is in our set
  filter(ancestor_concept_id %in% set) # and at the same time also the ancestor is in our set, btw, since we manually added clinical finding all will be selected, but we know which ones have clinical finding

# uncomment this and below to add clinical finding connections
# to_add_later <- output_internal %>%
#   filter(ancestor_concept_id == clinical_finding_concept_id)


# order and keep only the one with smallest level of separation
data_ordered <- output_internal[order(output_internal$min_levels_of_separation, output_internal$max_levels_of_separation, decreasing = FALSE), ]
data_output_internal <- data_ordered[!duplicated(data_ordered$descendant_concept_id), ]
# data_output_internal <- bind_rows(data_output_internal, to_add_later)
write.csv(data_output_internal, file="D:/git/KnowledgeGraph/data/opehr_concepts_11454.csv", row.names=FALSE)

# next line is a test to take all min levle of speration == 1, rather than removing all duplicates of descendant_concept_id, which means there are now multiple ways to rome?
data_output_internal <- data_ordered %>%
  dplyr::filter(min_levels_of_separation == 1)

################################################################################
# REF FILE GENERATION

refOriginal = read.table("D:/git/KnowledgeGraph/data/CONCEPT.csv", sep="\t", quote = "", fill = TRUE, header = TRUE)

ref <- refOriginal %>%
  select(concept_id, concept_name, domain_id, standard_concept) %>%
  # filter(concept_class_id == "Clinical Finding")
  filter(domain_id == "Condition" | domain_id == "Observation" | domain_id == "Measurement" | domain_id == "Spec Anatomic Site" | domain_id == "Procedure" | domain_id == "Relationship")

write.csv(ref, file="D:/git/KnowledgeGraph/data/ref.csv", row.names=FALSE)
