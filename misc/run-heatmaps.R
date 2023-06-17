library(magrittr)
library(heatmaply)
source(file = "./misc/heatmap-helper.R")

getAUROC <- function(inPlpResult) {
  inPlpResult$performanceEvaluation$evaluationStatistics %>%
    dplyr::filter(metric == "AUROC" & (evaluation == "Test" | evaluation == "Validation")) %>%
    dplyr::select(value) %>%
    unlist()
}

getEavg <- function(inPlpResult) {
  inPlpResult$performanceEvaluation$evaluationStatistics %>%
    dplyr::filter(metric == "Eavg" & (evaluation == "Test" | evaluation == "Validation")) %>%
    dplyr::select(value) %>%
    unlist() %>%
    as.double()
}

getEmax <- function(inPlpResult) {
  inPlpResult$performanceEvaluation$evaluationStatistics %>%
    dplyr::filter(metric == "Emax" & (evaluation == "Test" | evaluation == "Validation")) %>%
    dplyr::select(value) %>%
    unlist() %>%
    as.double()
}

gerda_gerda_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/models-L2/GERDA_Full/plpResult")
gerda_ipci_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/GERDA_Full/IPCI")
gerda_mdcr_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/GERDA_Full/MDCR")
gerda_opehr_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/GERDA_Full/OPEHR")
gerda_opses_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/GERDA_Full/OPSES")

ipci_gerda_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/IPCI_Full/GERDA")
ipci_ipci_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/models-L2/IPCI_Full/plpResult")
ipci_mdcr_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/IPCI_Full/MDCR")
ipci_opehr_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/IPCI_Full/OPEHR")
ipci_opses_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/IPCI_Full/OPSES")

mdcr_gerda_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/MDCR_Full/GERDA")
mdcr_ipci_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/MDCR_Full/IPCI")
mdcr_mdcr_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/models-L2/MDCR_Full/plpResult")
mdcr_opehr_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/MDCR_Full/OPEHR")
mdcr_opses_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/MDCR_Full/OPSES")

opehr_gerda_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/OPEHR_Full/GERDA")
opehr_ipci_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/OPEHR_Full/IPCI")
opehr_mdcr_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/OPEHR_Full/MDCR")
opehr_opehr_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/models-L2/OPEHR_Full/plpResult")
opehr_opses_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/OPEHR_Full/OPSES")

opses_gerda_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/OPSES_Full/GERDA")
opses_ipci_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/OPSES_Full/IPCI")
opses_mdcr_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/OPSES_Full/MDCR")
opses_opehr_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/external-L2/OPSES_Full/OPEHR")
opses_opses_full <- PatientLevelPrediction::loadPlpResult("D:/git/omop-poincare/output/models-L2/OPSES_Full/plpResult")

data_auroc_model <- matrix(
  c(getAUROC(mdcr_mdcr_full),
    getAUROC(mdcr_gerda_full),
    getAUROC(mdcr_opses_full),
    getAUROC(mdcr_opehr_full), # MDCR
    getAUROC(mdcr_ipci_full),
    getAUROC(gerda_mdcr_full),
    getAUROC(gerda_gerda_full), # GERDA
    getAUROC(gerda_opses_full),
    getAUROC(gerda_opehr_full),
    getAUROC(gerda_ipci_full),
    getAUROC(opses_mdcr_full),
    getAUROC(opses_gerda_full), # OPSES
    getAUROC(opses_opses_full),
    getAUROC(opses_opehr_full),
    getAUROC(opses_ipci_full),
    getAUROC(opehr_mdcr_full),
    getAUROC(opehr_gerda_full), # OPEHR
    getAUROC(opehr_opses_full),
    getAUROC(opehr_opehr_full),
    getAUROC(opehr_ipci_full),
    getAUROC(ipci_mdcr_full),
    getAUROC(ipci_gerda_full), # IPCI
    getAUROC(ipci_opses_full),
    getAUROC(ipci_opehr_full),
    getAUROC(ipci_ipci_full)),
  ncol = 5, byrow = TRUE)

drawAurocHeatmap(data_auroc_model, title = "FULL - L2 - AUROC")