library(magrittr)
library(dplyr)
library(DBI)
library(duckdb)
library(arrow)

# the below is much faster if you have a database with the vocabulary instead of only a csv. Plus
# csv's don't have any types so there is chance of errors
connection <- dbConnect(duckdb::duckdb(), "/home/egill/database/database.duckdb")
edges <- dbGetQuery(connection, "SELECT ancestor_concept_id, descendant_concept_id, min_levels_of_separation
 FROM concept_ancestor INNER JOIN concept ON concept_ancestor.descendant_concept_id = concept.concept_id
WHERE concept.domain_id in ('Condition', 'Observation', 'Measurement', 'Spec Anatomic Site', 'Procedure', 'Relationship')")
# test <- read.csv("./data/CONCEPT_ANCESTOR.csv")

data <- PatientLevelPrediction::loadPlpData("/home/egill/github/KnowledgeGraph/data/targetId_11454_L1/")

set <- data$covariateData$covariateRef %>%
  collect() %>%
  filter(analysisId == 102) %>%
  select(conceptId) %>%
  unlist()

clinical_finding_concept_id <- 441840

set <- c(set, clinical_finding_concept_id)
set <- as.character(set)
################################################################################

output_internal <- edges %>%
  filter(min_levels_of_separation != 0) %>% # remove the self-reference, we do not need it
  filter(min_levels_of_separation == 1) %>%
  select(ancestor_concept_id, descendant_concept_id)
  # %>%
  # filter(descendant_concept_id %in% set)
nodes <- dbGetQuery(connection, "SELECT * FROM concept WHERE
domain_id in ('Condition', 'Observation', 'Measurement', 'Spec Anatomic Site', 'Procedure', 'Relationship')
and standard_concept = 'S'")

arrow::write_feather(nodes, "/home/egill/github/omop-poincare/data/snomed_nodes.arrow")
arrow::write_feather(output_internal, "/home/egill/github/omop-poincare/data/snomed_edges.arrow")

output_internal %>% inner_join(nodes, join_by("ancestor_concept_id"=="concept_id"))

# set <- c(output_internal$ancestor_concept_id, output_internal$descendant_concept_id)
# set <- unique(set)

library(igraph)
# Create a directed graph from the full edge list
g_full <- graph_from_data_frame(output_internal, directed = TRUE)



# Get the list of vertex names in the graph
graph_vertices <- V(g_full)$name

# Check for vertices in 'set' that are missing in the graph
invalid_nodes <- set[!(set %in% graph_vertices)]

set <- set[set %in% graph_vertices]

set <- as.character(set)

invalid_nodes <- set[!(set %in% graph_vertices)]



# Find all relevant nodes: nodes of interest and their ancestors
all_relevant_nodes <- unique(unlist(lapply(set, function(node) {
  subcomponent(g_full, node, mode = "in")
})))

relevant_nodes <- as.character(all_relevant_nodes)

# Identify relevant vertices in the graph
# valid_vertices <- V(g_full)[V(g_full)$name %in% relevant_nodes]

# Create a subgraph with only the relevant vertices
sub_g <- induced_subgraph(g_full, V(g_full) %in% relevant_nodes)

# Extract the edge list from the subgraph
# edge_list <- as_data_frame(sub_g, what = "edges")
edge_list <- as_data_frame(g_full, what = "edges")
# 
# # Filter the edge list based on relevant nodes
# reduced_edges <- output_internal[
#   output_internal$ancestor_concept_id %in% all_relevant_nodes &
#     output_internal$descendant_concept_id %in% all_relevant_nodes,
# ]



# # order and keep only the one with smallest level of separation
# data_ordered <- output_internal[order(output_internal$min_levels_of_separation, output_internal$max_levels_of_separation, decreasing = FALSE), ]
# data_output_internal <- data_ordered[!duplicated(data_ordered$descendant_concept_id), ]
# data_output_internal <- bind_rows(data_output_internal, to_add_later)
write.csv(edge_list, file="./data/opehr_concepts_11454.csv", row.names=FALSE)

# next line is a test to take all min levle of speration == 1, rather than removing all duplicates of descendant_concept_id, which means there are now multiple ways to rome?
# data_output_internal <- data_ordered %>%
#   dplyr::filter(min_levels_of_separation == 1)

################################################################################
# REF FILE GENERATION

# refOriginal <- read.table("D:/git/KnowledgeGraph/data/CONCEPT.csv", sep="\t", quote = "", fill = TRUE, header = TRUE)
refOriginal <- dbGetQuery(connection, "SELECT * FROM concept")

ref <- refOriginal %>%
  select(concept_id, concept_name, domain_id, standard_concept) %>%
  # filter(concept_class_id == "Clinical Finding")
  filter(domain_id == "Condition" | domain_id == "Observation" | domain_id == "Measurement" | domain_id == "Spec Anatomic Site" | domain_id == "Procedure" | domain_id == "Relationship")

write.csv(ref, file="./data/ref.csv", row.names=FALSE)
