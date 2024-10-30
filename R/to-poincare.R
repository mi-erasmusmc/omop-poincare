library(dplyr)
library(tidyr)
library(Andromeda)
library(FeatureExtraction)
library(readr)

options(andromedaTempFolder = "D:/temp")
options(arrow.int64_downcast = FALSE)

# plpDataDir <- "D:/git/omop-poincare/data/data_original/truven_mdcr"
plpDataDir <- "D:/git/omop-poincare/data/data_poincare/truven_mdcr"

data_in <- PatientLevelPrediction::loadPlpData(plpDataDir)

covariates <- data_in$covariateData$covariates %>%
  collect()
covariateRef <- data_in$covariateData$covariateRef %>%
  collect()
analysisRef <- data_in$covariateData$analysisRef %>%
  collect()

lab <- readr::read_tsv("D:/git/omop-poincare/output/tf_proj_lab.tsv", col_names = FALSE)
vec <- readr::read_tsv("D:/git/omop-poincare/output/tf_proj_vec.tsv", col_names = FALSE)
colnames(lab) <- c("covariateId")
colnames(vec) <- 1:ncol(vec)

embedding <- bind_cols(lab, vec)
embedding <- embedding %>%
  dplyr::mutate(covariateId=paste0(covariateId,999)) %>%
  dplyr::mutate(covariateId = bit64::as.integer64(covariateId))

# embedding <- left_join(covariates, embedding, by=c('covariateId'='covariateId'))

covariates_embedding <- left_join(covariates, embedding, by='covariateId') %>%
  group_by(rowId, .drop = FALSE) %>%
  summarise_at(vars(colnames(vec)), list(mean))
  
covariates_embedding_final <- covariates_embedding %>%
  pivot_longer(cols=colnames(vec),
               names_to='covariateId',
               values_to='covariateValue') %>%
  mutate(covariateId = bit64::as.integer64(paste0(covariateId, 999)))

covariateRef_final <- data.frame(
  covariateId=bit64::as.integer64(paste0(1:ncol(vec), 999)),
  covariateName=paste0(1:ncol(vec)),
  analysisId=999,
  conceptId=1:ncol(vec)
)

#### remove this once we have correct ipci data
# covariates_embedding_final <- covariates_embedding_final %>%
#   dplyr::filter(!is.na(covariateValue))


data <- Andromeda::andromeda(covariates = covariates_embedding_final,
                     covariateRef = covariateRef_final,
                     analysisRef = analysisRef)

result <- list(covariateData = data,
               timeRef = readRDS(file.path(plpDataDir, "timeRef.rds")),
               cohorts = readRDS(file.path(plpDataDir, "cohorts.rds")),
               outcomes = readRDS(file.path(plpDataDir, "outcomes.rds")),
               metaData = readRDS(file.path(plpDataDir, "metaData.rds")))

fileName <- file.path(plpDataDir, "covariates")

fileNamesInZip <- utils::unzip(fileName, list = TRUE)$Name
# sqliteFilenameInZip <- fileNamesInZip[grepl(".sqlite$", fileNamesInZip)]
rdsFilenameInZip <- fileNamesInZip[grepl(".rds$", fileNamesInZip)]

andromedaTempFolder <- Andromeda:::.getAndromedaTempFolder()
# .checkAvailableSpace()

tempDir <- tempfile(tmpdir = andromedaTempFolder)
dir.create(tempDir)
# on.exit(unlink(tempDir, recursive = TRUE))
zip::unzip(fileName, exdir = tempDir)

##############
# attributes <- readRDS(file.path(tempDir, rdsFilenameInZip))
# for (name in names(attributes)) {
#   attr(result$covariateData, name) <- attributes[[name]]
# }
##############
class(result$covariateData) <- "CovariateData"
attr(class(result$covariateData), "package") <- "Andromeda"

class(result) <- "plpData"

# attr(covariateData, "metaData") <- NULL

PatientLevelPrediction::savePlpData(result, file.path("./data", "data_poincare", "truven_mdcr"))

################
