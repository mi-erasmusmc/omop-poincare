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
  "D:/git/omop-poincare/data/data_original/emc_ipci"
)

plpDataDirFullPoincare <- c(
  "D:/git/omop-poincare/data/data_poincare/ims_germany",
  "D:/git/omop-poincare/data/data_poincare/truven_mdcr",
  "D:/git/omop-poincare/data/data_poincare/optum_ehr",
  "D:/git/omop-poincare/data/data_poincare/optum_extended_ses",
  "D:/git/omop-poincare/data/data_poincare/emc_ipci"
)

modelSettings <- setGradientBoostingMachine(seed = 1000, nthread = 10)
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
    modelSettings = modelSettings,
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
    modelSettings = modelSettings,
    logSettings = createLogSettings(),
    executeSettings = createExecuteSettings(runSplitData = T,
                                            runSampleData = F,
                                            runfeatureEngineering = F,
                                            runPreprocessData = F,
                                            runModelDevelopment = T,
                                            runCovariateSummary = F),
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

plpModelsFullPoincare <- c(
  "D:/git/omop-poincare/output/models-L2-poincare/GERDA_Poincare/plpResult/model",
  "D:/git/omop-poincare/output/models-L2-poincare/MDCR_Poincare/plpResult/model",
  "D:/git/omop-poincare/output/models-L2-poincare/OPEHR_Poincare/plpResult/model",
  "D:/git/omop-poincare/output/models-L2-poincare/OPSES_Poincare/plpResult/model",
  "D:/git/omop-poincare/output/models-L2-poincare/IPCI_Poincare/plpResult/model"
)


getValidationSettingsFull <- function(i, j) {
  settings <- list()
  
  result <- list(
    plpModel = plpModelsFullPoincare[i],
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
