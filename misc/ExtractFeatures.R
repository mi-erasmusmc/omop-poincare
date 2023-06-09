library(PatientLevelPrediction)

connectionDetails <- Eunomia::getEunomiaConnectionDetails()
Eunomia::createCohorts(connectionDetails)

conceptIds <- readRDS("~/Downloads/concept_ids.rds")

subType <- "all"
windowStart <- -365
windowEnd <- 0
analysisId = 999 # make sure this is three digits or it will be zero-padded and confuse henrik
sqlFileName <- "DomainConcept.sql"
analyses <- list()
analyses[[1]] <- FeatureExtraction::createAnalysisDetails(
  analysisId = analysisId,
  sqlFileName = sqlFileName,
  includedCovariateConceptIds = conceptIds,
  # includedCovariateIds = 4285898023,
  parameters = list(
    analysisId = analysisId,
    analysisName = sprintf("Condition concepts in days %d - %d", windowStart, windowEnd),
    startDay = windowStart,
    endDay = windowEnd,
    subType = subType,
    domainId = "Condition",
    domainTable	= "condition_occurrence",
    domainConceptId = "condition_concept_id",
    domainStartDate = "condition_start_date",
    domainEndDate = "condition_start_date"
  )
)

covariateSettings <- FeatureExtraction::createDetailedCovariateSettings(
  analyses = analyses
)

databaseDetails <- PatientLevelPrediction::createDatabaseDetails(
  connectionDetails = connectionDetails,
  cdmDatabaseSchema = "main",
  cohortDatabaseSchema = "main",
  cohortTable = "cohort",
  targetId = 4,
  outcomeIds = 3,
  outcomeDatabaseSchema = "main",
  outcomeTable = "cohort",
  cdmDatabaseName = "eunomia"
)

plpData <- PatientLevelPrediction::getPlpData(
  databaseDetails = databaseDetails,
  restrictPlpDataSettings = PatientLevelPrediction::createRestrictPlpDataSettings(),
  covariateSettings = covariateSettings
)

savePlpData(plpData = plpData, "plpData_dementia_poincare")

View(dplyr::collect(plpData$covariateData$covariates))

