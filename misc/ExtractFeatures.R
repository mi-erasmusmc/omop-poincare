library(PatientLevelPrediction)

# ---------------------------------------------------------------------------- #                                                                              #
#         INPUT: DATABASE SETTINGS                                             #                                                                              #
# ---------------------------------------------------------------------------- #                                                                              #
BASE_URL <- Sys.getenv("BASE_URL")
USER <- Sys.getenv("USER")
PASSWORD <- Sys.getenv("PASSWORD")
DB_PASSWORD <- Sys.getenv("DB_PASSWORD")
SERVER <- Sys.getenv("SERVER")
EXTRA_SETTINGS <- Sys.getenv("EXTRA_SETTINGS")
PORT <- 5439

cohortDatabaseSchema <- "scratch_hjohn2"
cohortTable <- "poincare_cohort"
pathToDriver <- "~/JdbcDrivers"
DATA_SAVE_DIR <- './results'

# 11931 - Dementia target - v7
# 6243 - dementia outcome - v1
analysis <- list(target=11931, outcome=6243)

set_gerda <- readRDS(file.path("./data", "set_gerda.RDS"))
set_mdcr <- readRDS(file.path("./data", "set_mdcr.RDS"))
set_opehr <- readRDS(file.path("./data", "set_opehr.RDS"))
set_opses <- readRDS(file.path("./data", "set_opses.RDS"))

databases <- list(mdcr=list(name="truven_mdcr", version="v2322", sample=1000000, conceptIds=set_gerda), 
                  opehr=list(name="optum_ehr", version="v2247", sample=1000000, conceptIds=set_mdcr),
                  iqger=list(name="ims_germany", version="v2352", sample=1000000, conceptIds=set_opehr),
                  opses=list(name="optum_extended_ses", version='v2327', sample=1000000, conceptIds=set_opses))

# ---------------------------------------------------------------------------- #                                                                              #
#         INPUT: COVARIATE SETTINGS                                            #                                                                              #
# ---------------------------------------------------------------------------- #
conceptIds <- readRDS("~/Downloads/concept_ids.rds")

getCovariateSettings <- function(conceptIds) {
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
  
  return(covariateSettings)
}


# ---------------------------------------------------------------------------- #                                                                              #
#         CREATE COHORT TABLES                                                 #                                                                              #
# ---------------------------------------------------------------------------- # 
allIds <- unique(unlist(analysis))

ROhdsiWebApi::authorizeWebApi(baseUrl = BASE_URL,
                              authMethod = 'windows',
                              webApiUsername = USER,
                              webApiPassword = PASSWORD)

cohortTableNames <- CohortGenerator::getCohortTableNames(cohortTable = cohortTable)

cohortsToCreate <- ROhdsiWebApi::exportCohortDefinitionSet(
  baseUrl = BASE_URL, 
  cohortIds = allIds,
  generateStats = F
)

# create cohorts and store in cohort tables in scratch space
for (database in databases) {
  server <- paste0(SERVER,'/', database$name)
  connectionDetails <- DatabaseConnector::createConnectionDetails(dbms ='redshift',
                                                                  server = server,
                                                                  port = PORT,
                                                                  user = USER,
                                                                  password = DB_PASSWORD,
                                                                  extraSettings = "ssl=true&sslfactor=com.amazon.redshift.ssl.NonValidatingFactory",
                                                                  pathToDriver = pathToDriver)
  
  CohortGenerator::createCohortTables(connectionDetails = connectionDetails,
                                      cohortDatabaseSchema = cohortDatabaseSchema,
                                      cohortTableNames = cohortTableNames)
  cdmDatabaseSchema <- paste0("cdm_", database$name, "_", database$version)
  cohortsGenerated <- CohortGenerator::generateCohortSet(connectionDetails = connectionDetails,
                                                         cdmDatabaseSchema = cdmDatabaseSchema,
                                                         cohortDatabaseSchema = cohortDatabaseSchema,
                                                         cohortTableNames = cohortTableNames,
                                                         cohortDefinitionSet = cohortsToCreate)
}

# ---------------------------------------------------------------------------- #                                                                              #
#         EXTRACT FEATURES                                                     #                                                                              #
# ---------------------------------------------------------------------------- #
fix64bit <- function(plpData) {
  plpData$covariateData$covariateRef <- plpData$covariateData$covariateRef |>
    dplyr::mutate(covariateId=bit64::as.integer64(covariateId))
  plpData$covariateData$covariates <- plpData$covariateData$covariates |>
    dplyr::mutate(rowId = as.integer(rowId),
                  covariateId = bit64::as.integer64(covariateId))
  plpData$cohorts <- plpData$cohorts |> dplyr::mutate(rowId=as.integer(rowId))
  plpData$outcomes <- plpData$outcomes |> dplyr::mutate(rowId = as.integer(rowId))
  
  return(plpData)
}

# get plpData and save in folders with name of database
for (database in databases) {
  server <- paste0(SERVER,'/', database$name)
  connectionDetails <- DatabaseConnector::createConnectionDetails(dbms ='redshift',
                                                                  server = server,
                                                                  port = PORT,
                                                                  user = USER,
                                                                  password = DB_PASSWORD,
                                                                  extraSettings = EXTRA_SETTINGS,
                                                                  pathToDriver = pathToDriver)
  
  cdmDatabaseSchema <- paste0("cdm_", database$name, "_", database$version)
  databaseDetails <- PatientLevelPrediction::createDatabaseDetails(
    connectionDetails = connectionDetails,
    cdmDatabaseSchema = cdmDatabaseSchema,
    cdmDatabaseName = database$name,
    cohortDatabaseSchema = cohortDatabaseSchema,
    cohortTable = cohortTable,
    outcomeDatabaseSchema = cohortDatabaseSchema,
    outcomeTable = cohortTable,
    targetId = analysis$target,
    outcomeIds = analysis$outcome
  )
  
  
  plpData <- PatientLevelPrediction::getPlpData(
    databaseDetails=databaseDetails,
    covariateSettings = getCovariateSettings(database$conceptIds),
    restrictPlpDataSettings = PatientLevelPrediction::createRestrictPlpDataSettings(
      sampleSize = database$sample
    )
  )
  
  plpData <- fix64bit(plpData)
  
  saveDir <- file.path(DATA_SAVE_DIR, database$name)
  if (!dir.exists(saveDir)) {
    dir.create(saveDir, recursive=T)
  }
  PatientLevelPrediction::savePlpData(plpData, file=saveDir)
}
