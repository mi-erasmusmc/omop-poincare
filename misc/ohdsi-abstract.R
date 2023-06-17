library(dplyr)
library(tidyr)
library(Andromeda)
library(FeatureExtraction)
library(readr)
library(PatientLevelPrediction)

options(andromedaTempFolder = "D:/temp")
options(arrow.int64_downcast = FALSE)


populationSettings <- PatientLevelPrediction::createStudyPopulationSettings(
  binary = T, 
  includeAllOutcomes = T, 
  firstExposureOnly = T, 
  washoutPeriod = 365, 
  removeSubjectsWithPriorOutcome = F, 
  priorOutcomeLookback = 99999, 
  requireTimeAtRisk = T, 
  minTimeAtRisk = 1, 
  riskWindowStart = 1, 
  startAnchor = 'cohort start', 
  endAnchor = 'cohort start', 
  riskWindowEnd = 1825
)

analyses <- c(
  "GERDA",
  "MDCR",
  "OPEHR",
  "OPSES",
  "IPCI"
)

analysesIds <- list(targetId = c(0, 0, 0, 0, 0), outcomeId = c(6243, 6243, 6243, 6243, 1032))

plpDataDirFull <- c(
  "D:/git/omop-poincare/data/data_original/ims_germany",
  "D:/git/omop-poincare/data/data_original/truven_mdcr",
  "D:/git/omop-poincare/data/data_original/optum_ehr",
  "D:/git/omop-poincare/data/data_original/optum_extended_ses",
  "D:/git/omop-poincare/data/data_original/emc_ipci_old"
)

plpDataDirFullPoincare <- c(
  "D:/git/omop-poincare/data/data_poincare/ims_germany",
  "D:/git/omop-poincare/data/data_poincare/truven_mdcr",
  "D:/git/omop-poincare/data/data_poincare/optum_ehr",
  "D:/git/omop-poincare/data/data_poincare/optum_extended_ses",
  "D:/git/omop-poincare/data/data_poincare/emc_ipci_old_pruned"
)

#------------------------------------------------------------------------------#
# Full Models - CONCEPT DATA
#------------------------------------------------------------------------------#

getPlpSettingsFull <- function(i) {
  result <- list(
    plpData = plpDataDirFull[i],
    outcomeId = analysesIds$outcomeId[i],
    analysisId = paste0(analyses[i],"_","Full"),
    analysisName = paste0(analyses[i],"_","Full"),
    populationSettings = populationSettings,
    splitSettings = createDefaultSplitSetting(splitSeed = 1000,
                                              testFraction = 0.25,
                                              trainFraction = 0.75),
    sampleSettings = createSampleSettings(),
    featureEngineeringSettings = createFeatureEngineeringSettings(),
    preprocessSettings = createPreprocessSettings(),
    modelSettings = setRidgeRegression(seed = 1000),
    logSettings = createLogSettings(),
    executeSettings = createExecuteSettings(runSplitData = T,
                                            runSampleData = F,
                                            runfeatureEngineering = F,
                                            runPreprocessData = F,
                                            runModelDevelopment = T,
                                            runCovariateSummary = T),
    saveDirectory = file.path('D:/git/omop-poincare/output/models-L2')
  )
  
  return(result)
}

plpSettingsFull <- lapply(1:length(analyses), getPlpSettingsFull)

plpWrapperFull <- function(settings){
  options(arrow.int64_downcast = FALSE)
  
  plpData <- PatientLevelPrediction::loadPlpData(settings$plpData)
  settings$plpData <- plpData
  
  do.call(PatientLevelPrediction::runPlp, settings)
}

cluster <- ParallelLogger::makeCluster(numberOfThreads = length(analyses))
ParallelLogger::clusterRequire(
  cluster,
  c("PatientLevelPrediction", "Andromeda", "FeatureExtraction")
)

ParallelLogger::clusterApply(
  cluster = cluster, 
  x = plpSettingsFull, 
  fun = plpWrapperFull, 
  stopOnError = FALSE,
  progressBar = TRUE)

ParallelLogger::stopCluster(cluster)

#------------------------------------------------------------------------------#
# External Validation Full Models - CONCEPT DATA
#------------------------------------------------------------------------------#

plpModelsFull <- c("D:/git/omop-poincare/output/models-L2/GERDA_Full/plpResult/model",
                   "D:/git/omop-poincare/output/models-L2/MDCR_Full/plpResult/model",
                   "D:/git/omop-poincare/output/models-L2/OPEHR_Full/plpResult/model",
                   "D:/git/omop-poincare/output/models-L2/OPSES_Full/plpResult/model",
                   "D:/git/omop-poincare/output/models-L2/IPCI_Full/plpResult/model")


getValidationSettingsFull <- function(i, j) {
  settings <- list()
  
  result <- list(
    plpModel = plpModelsFull[i],
    plpData = plpDataDirFull[j],
    population = rep(NULL, length(analyses)),
    settings = PatientLevelPrediction::createValidationSettings(runCovariateSummary = F)
  )
  
  settings$targetOutcomeId <- analysesIds$outcomeId[j]
  settings$targetDatabaseName <- analyses[j]
  settings$settings <- result
  
  return(settings)
}

plpValidationSettingsFull <- unlist(apply(outer(1:length(analyses), 1:length(analyses),
                                                Vectorize(getValidationSettingsFull, SIMPLIFY = FALSE)), 1, as.list), recursive = FALSE)

plpValWrapperFull <- function(param){
  options(arrow.int64_downcast = FALSE)
  
  plpData <- PatientLevelPrediction::loadPlpData(param$settings$plpData)
  populationSettings <- PatientLevelPrediction::createStudyPopulationSettings(
    binary = T, 
    includeAllOutcomes = T, 
    firstExposureOnly = T, 
    washoutPeriod = 365, 
    removeSubjectsWithPriorOutcome = F, 
    priorOutcomeLookback = 99999, 
    requireTimeAtRisk = T, 
    minTimeAtRisk = 1, 
    riskWindowStart = 1, 
    startAnchor = 'cohort start', 
    endAnchor = 'cohort start', 
    riskWindowEnd = 1825
  )
  
  population <- PatientLevelPrediction::createStudyPopulation(
    plpData,
    outcomeId = param$targetOutcomeId,
    populationSettings = populationSettings)
  plpModel = PatientLevelPrediction::loadPlpModel(param$settings$plpModel)
  
  param$settings$plpData <- plpData
  param$settings$population = population
  param$settings$plpModel = plpModel
  
  result <- do.call(PatientLevelPrediction:::externalValidatePlp, param$settings)
  PatientLevelPrediction::savePlpResult(
    result,
    dirPath = file.path("D:/git/omop-poincare", "output", 'external-L2', plpModel$trainDetails$analysisId, param$targetDatabaseName))
}

cluster <- ParallelLogger::makeCluster(numberOfThreads = 10)
ParallelLogger::clusterRequire(cluster, c("PatientLevelPrediction", "Andromeda", "FeatureExtraction"))

plpValidationResult <- ParallelLogger::clusterApply(
  cluster = cluster, 
  x = plpValidationSettingsFull, 
  fun = plpValWrapperFull, 
  stopOnError = FALSE,
  progressBar = TRUE)

ParallelLogger::stopCluster(cluster)



#------------------------------------------------------------------------------#
# Full Models - POINCARE DATA
#------------------------------------------------------------------------------#

getPlpSettingsFull <- function(i) {
  result <- list(
    plpData = plpDataDirFullPoincare[i],
    outcomeId = analysesIds$outcomeId[i],
    analysisId = paste0(analyses[i],"_","Poincare"),
    analysisName = paste0(analyses[i],"_","Poincare"),
    populationSettings = populationSettings,
    splitSettings = createDefaultSplitSetting(splitSeed = 1000,
                                              testFraction = 0.25,
                                              trainFraction = 0.75),
    sampleSettings = createSampleSettings(),
    featureEngineeringSettings = createFeatureEngineeringSettings(),
    preprocessSettings = createPreprocessSettings(),
    modelSettings = setRidgeRegression(seed = 1000),
    logSettings = createLogSettings(),
    executeSettings = createExecuteSettings(runSplitData = T,
                                            runSampleData = F,
                                            runfeatureEngineering = F,
                                            runPreprocessData = F,
                                            runModelDevelopment = T,
                                            runCovariateSummary = T),
    saveDirectory = file.path('D:/git/omop-poincare/output/models-L2-poincare')
  )
  
  return(result)
}

plpSettingsFull <- lapply(1:length(analyses), getPlpSettingsFull)

plpWrapperFull <- function(settings){
  options(arrow.int64_downcast = FALSE)
  
  plpData <- PatientLevelPrediction::loadPlpData(settings$plpData)
  settings$plpData <- plpData
  
  do.call(PatientLevelPrediction::runPlp, settings)
}

cluster <- ParallelLogger::makeCluster(numberOfThreads = length(analyses))
ParallelLogger::clusterRequire(
  cluster,
  c("PatientLevelPrediction", "Andromeda", "FeatureExtraction")
)

ParallelLogger::clusterApply(
  cluster = cluster, 
  x = plpSettingsFull, 
  fun = plpWrapperFull, 
  stopOnError = FALSE,
  progressBar = TRUE)

ParallelLogger::stopCluster(cluster)


#------------------------------------------------------------------------------#
# External Validation Full Models - POINCARE DATA
#------------------------------------------------------------------------------#

plpModelsFullPoincare <- c("D:/git/omop-poincare/output/models-L2-poincare/GERDA_Full/plpResult/model",
                   "D:/git/omop-poincare/output/models-L2-poincare/MDCR_Full/plpResult/model",
                   "D:/git/omop-poincare/output/models-L2-poincare/OPEHR_Full/plpResult/model",
                   "D:/git/omop-poincare/output/models-L2-poincare/OPSES_Full/plpResult/model",
                   "D:/git/omop-poincare/output/models-L2-poincare/IPCI_Full/plpResult/model")


getValidationSettingsFull <- function(i, j) {
  settings <- list()
  
  result <- list(
    plpModel = plpModelsFull[i],
    plpData = plpDataDirFullPoincare[j],
    population = rep(NULL, length(analyses)),
    settings = PatientLevelPrediction::createValidationSettings(runCovariateSummary = F)
  )
  
  settings$targetOutcomeId <- analysesIds$outcomeId[j]
  settings$targetDatabaseName <- analyses[j]
  settings$settings <- result
  
  return(settings)
}

plpValidationSettingsFull <- unlist(apply(outer(1:length(analyses), 1:length(analyses),
                                                Vectorize(getValidationSettingsFull, SIMPLIFY = FALSE)), 1, as.list), recursive = FALSE)

plpValWrapperFull <- function(param){
  options(arrow.int64_downcast = FALSE)
  
  plpData <- PatientLevelPrediction::loadPlpData(param$settings$plpData)
  populationSettings <- PatientLevelPrediction::createStudyPopulationSettings(
    binary = T, 
    includeAllOutcomes = T, 
    firstExposureOnly = T, 
    washoutPeriod = 365, 
    removeSubjectsWithPriorOutcome = F, 
    priorOutcomeLookback = 99999, 
    requireTimeAtRisk = T, 
    minTimeAtRisk = 1, 
    riskWindowStart = 1, 
    startAnchor = 'cohort start', 
    endAnchor = 'cohort start', 
    riskWindowEnd = 1825
  )
  
  population <- PatientLevelPrediction::createStudyPopulation(
    plpData,
    outcomeId = param$targetOutcomeId,
    populationSettings = populationSettings)
  plpModel = PatientLevelPrediction::loadPlpModel(param$settings$plpModel)
  
  param$settings$plpData <- plpData
  param$settings$population = population
  param$settings$plpModel = plpModel
  
  result <- do.call(PatientLevelPrediction:::externalValidatePlp, param$settings)
  PatientLevelPrediction::savePlpResult(
    result,
    dirPath = file.path("D:/git/omop-poincare", "output", 'external-L2-poincare', plpModel$trainDetails$analysisId, param$targetDatabaseName))
}

cluster <- ParallelLogger::makeCluster(numberOfThreads = 10)
ParallelLogger::clusterRequire(cluster, c("PatientLevelPrediction", "Andromeda", "FeatureExtraction"))

plpValidationResult <- ParallelLogger::clusterApply(
  cluster = cluster, 
  x = plpValidationSettingsFull, 
  fun = plpValWrapperFull, 
  stopOnError = FALSE,
  progressBar = TRUE)

ParallelLogger::stopCluster(cluster)










#####################################################################################
library(PatientLevelPrediction)
# library(DeepPatientLevelPrediction)

# plpDataTesting <- PatientLevelPrediction::loadPlpData("poincareData")

plpDataDir_ipci_original <- "D:/git/omop-poincare/data/ipci_original/emc_ipci"
plpDataDir_ipci_poincare <- "D:/git/omop-poincare/data/ipci_poincare_abstract"
plpDataDir_gerda_original <- "D:/git/omop-poincare/data/ims_germany"
plpDataDir_gerda_poincare <- "D:/git/omop-poincare/data/gerda_poincare_abstract"


plpData <- PatientLevelPrediction::loadPlpData(plpDataDir_original)

settings <- list(
  plpData = plpData,
  outcomeId = analysesIds$outcomeId[1],
  analysisId = paste0(analyses[1],"_","L1-No-Poincare"),
  analysisName = paste0(analyses[1],"_","L1-No-Poincare"),
  populationSettings = populationSettings,
  splitSettings = createDefaultSplitSetting(splitSeed = 1000, testFraction = 0.25, trainFraction = 0.75),
  sampleSettings = createSampleSettings("underSample"),
  featureEngineeringSettings = createFeatureEngineeringSettings(),
  preprocessSettings = createPreprocessSettings(),
  modelSettings = setRidgeRegression(seed = 1000),
  logSettings = createLogSettings(),
  executeSettings = createExecuteSettings(runSplitData = T,
                                          runSampleData = F,
                                          runfeatureEngineering = F,
                                          runPreprocessData = F,
                                          runModelDevelopment = T,
                                          runCovariateSummary = T),
  saveDirectory = file.path('D:/git/omop-poincare/output'),
  limitPop = NULL
)

result1 <- do.call(PatientLevelPrediction::runPlp, settings)
result1 <- readRDS("D:/git/omop-poincare/output/IPCI_L1-No-Poincare/plpResult/runPlp.rds")




plpData <- PatientLevelPrediction::loadPlpData(plpDataDir_gerda_original)

###### CREATE A SPLIT BASED ON HIERARCHY

# library(magrittr)
# library(dplyr)
# 
# test <- read.csv("C:/Users/luish/Downloads/CONCEPT_ANCESTOR.csv", sep='\t')
# trainCovariateId <- result1$covariateSummary %>%
#   # filter(abs(covariateValue) >= 0.0) %>%
#   select(covariateId, conceptId)
# 
# output_internal <- test %>%
#   filter(min_levels_of_separation != 0 & max_levels_of_separation != 0) %>% # remove the self-reference, we do not need it
#   filter(descendant_concept_id %in% trainCovariateId$conceptId) %>% # descendant is in our set
#   filter(ancestor_concept_id %in% trainCovariateId$conceptId)
# 
# ancestor_set <- output_internal$ancestor_concept_id
# # descendant_set <- output_internal$descendant_concept_id
# 
# set <- unique(ancestor_set)
# 
# trainCovariateId_new <- result1$covariateSummary %>%
#   filter(conceptId %in% set) %>%
#   select(covariateId)
# 
# trainRowIds <- plpData$covariateData$covariates %>%
#   collect() %>%
#   filter(covariateId %in% trainCovariateId_new$covariateId) %>%
#   select(rowId) %>%
#   filter(duplicated(rowId) == FALSE)
#   

settings <- list(
  plpData = plpData,
  outcomeId = analysesIds$outcomeId[2],
  analysisId = paste0(analyses[2],"_","GB-No-Poincare"),
  analysisName = paste0(analyses[2],"_","GB-No-Poincare"),
  populationSettings = populationSettings,
  splitSettings = createDefaultSplitSetting(splitSeed = 1000, testFraction = 0.25, trainFraction = 0.75),
  sampleSettings = createSampleSettings("underSample"),
  featureEngineeringSettings = createFeatureEngineeringSettings(),
  preprocessSettings = createPreprocessSettings(),
  modelSettings = setGradientBoostingMachine(seed = 1000, nthread = 10),
  logSettings = createLogSettings(),
  executeSettings = createExecuteSettings(runSplitData = T,
                                          runSampleData = F,
                                          runfeatureEngineering = F,
                                          runPreprocessData = F,
                                          runModelDevelopment = T,
                                          runCovariateSummary = T),
  saveDirectory = file.path('D:/git/omop-poincare/output'),
  limitPop = NULL # trainRowIds$rowId
)

result2 <- do.call(PatientLevelPrediction::runPlp, settings)


####### "externally" validate now, well more or less

options(arrow.int64_downcast = FALSE)

plpData <- PatientLevelPrediction::loadPlpData(plpDataDir_ipci_original)
populationSettings <- PatientLevelPrediction::createStudyPopulationSettings(
  binary = T, 
  includeAllOutcomes = T, 
  firstExposureOnly = T, 
  washoutPeriod = 365, 
  removeSubjectsWithPriorOutcome = F, 
  priorOutcomeLookback = 99999, 
  requireTimeAtRisk = T, 
  minTimeAtRisk = 1, 
  riskWindowStart = 1, 
  startAnchor = 'cohort start', 
  endAnchor = 'cohort start', 
  riskWindowEnd = 1825
)

population <- PatientLevelPrediction::createStudyPopulation(
  plpData,
  outcomeId = analysesIds$outcomeId[1],
  populationSettings = populationSettings)
plpModel = PatientLevelPrediction::loadPlpModel(file.path('D:/git/omop-poincare/output/IPCI_GB-No-Poincare/plpResult/model'))

# population <- population %>%
#   filter(!(rowId %in% trainRowIds$rowId))

param <- list()
param$settings$plpData <- plpData
param$settings$population <- population
param$settings$plpModel <- plpModel

result_final_original <- do.call(PatientLevelPrediction:::externalValidatePlp, param$settings)

####### 4. TRAIN AND"externally" validate now, THE POINCARE MODEL

options(arrow.int64_downcast = FALSE)

plpData <- PatientLevelPrediction::loadPlpData(plpDataDir_gerda_poincare)

settings <- list(
  plpData = plpData,
  outcomeId = analysesIds$outcomeId[2],
  analysisId = paste0(analyses[2],"_","GB-Poincare"),
  analysisName = paste0(analyses[2],"_","GB-Poincare"),
  populationSettings = populationSettings,
  splitSettings = createDefaultSplitSetting(splitSeed = 1000, testFraction = 0.25, trainFraction = 0.75),
  sampleSettings = createSampleSettings("underSample"),
  featureEngineeringSettings = createFeatureEngineeringSettings(),
  preprocessSettings = createPreprocessSettings(),
  modelSettings = setGradientBoostingMachine(seed = 1000, nthread = 10),
  logSettings = createLogSettings(),
  executeSettings = createExecuteSettings(runSplitData = T,
                                          runSampleData = F,
                                          runfeatureEngineering = F,
                                          runPreprocessData = F,
                                          runModelDevelopment = T,
                                          runCovariateSummary = T),
  saveDirectory = file.path('D:/git/omop-poincare/output'),
  limitPop = NULL # trainRowIds$rowId
)

result3 <- do.call(PatientLevelPrediction::runPlp, settings)


########

plpData <- PatientLevelPrediction::loadPlpData(plpDataDir_ipci_poincare)
populationSettings <- PatientLevelPrediction::createStudyPopulationSettings(
  binary = T, 
  includeAllOutcomes = T, 
  firstExposureOnly = T, 
  washoutPeriod = 365, 
  removeSubjectsWithPriorOutcome = F, 
  priorOutcomeLookback = 99999, 
  requireTimeAtRisk = T, 
  minTimeAtRisk = 1, 
  riskWindowStart = 1, 
  startAnchor = 'cohort start', 
  endAnchor = 'cohort start', 
  riskWindowEnd = 1825
)

population <- PatientLevelPrediction::createStudyPopulation(
  plpData,
  outcomeId = analysesIds$outcomeId[1],
  populationSettings = populationSettings)
plpModel = PatientLevelPrediction::loadPlpModel(file.path('D:/git/omop-poincare/output/IPCI_GB-Poincare/plpResult/model'))

# population <- population %>%
#   filter(!(rowId %in% trainRowIds$rowId))

param <- list()
param$settings$plpData <- plpData
param$settings$population <- population
param$settings$plpModel <- plpModel

result_final_original <- do.call(PatientLevelPrediction:::externalValidatePlp, param$settings)
