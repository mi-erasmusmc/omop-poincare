library(PatientLevelPrediction)
# get connection and data from Eunomia
connectionDetails <- Eunomia::getEunomiaConnectionDetails()
Eunomia::createCohorts(connectionDetails)

covSet <- FeatureExtraction::createCovariateSettings(
  useDemographicsGender = T,
  useDemographicsAgeGroup = T,
  endDays = -1
)

subType <- "all"
windowStart <- c(-365, -180, -30)
windowEnd <- c(0, 0, 0)
analysisIdOffset = 930
sqlFileName <- "DomainConcept.sql"
analyses <- list()
analyses[[1]] <- FeatureExtraction::createAnalysisDetails(
  analysisId = analysisIdOffset+1,
  sqlFileName = sqlFileName,
  includedCovariateConceptIds = c(81893, 4182210, 316139, 313217),
  # includedCovariateIds = c(81893930),
  parameters = list(
    analysisId = analysisIdOffset,
    analysisName = sprintf("Condition concepts in days %d - %d", windowStart[1], windowEnd[1]),
    startDay = windowStart[1],
    endDay = windowEnd[1],
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

covSet <- covariateSettings

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

restrictPlpDataSettings <- PatientLevelPrediction::createRestrictPlpDataSettings(
  firstExposureOnly = T,
  washoutPeriod = 365
)

plpData <- PatientLevelPrediction::getPlpData(
  databaseDetails = databaseDetails,
  restrictPlpDataSettings = restrictPlpDataSettings,
  covariateSettings = covSet
)

View(dplyr::collect(plpData$covariateData$covariates))
