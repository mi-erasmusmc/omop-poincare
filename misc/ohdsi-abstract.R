library(dplyr)
library(tidyr)
library(Andromeda)
library(FeatureExtraction)
library(readr)

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
  "IPCI",
  "GERDA"
)

analysesIds <- list(targetId = c(1115, 0), outcomeId = c(1032, 6243))
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
