package training

import config.readYamlConfig
import constants.PATH_TO_YAML_CONFIG
import logging.ProjectLogger
import kotlin.time.measureTime

/**
 * Provides various training pipelines (e.g. for ensemble classifiers or logistic regression).
 */
fun trainingPipelineRunner(algorithm: String) {
    val cfg = readYamlConfig(filePath = PATH_TO_YAML_CONFIG)

    // TODO: Replace if with when

    if (algorithm in listOf("decisionTree", "randomForest", "adaBoost", "gradientBoosting")) {
        val duration = measureTime { EnsembleTrainingPipeline(cfg = cfg).execute() }
        ProjectLogger.logger.info("Ensemble training pipeline duration: ${duration.inWholeSeconds} seconds")
    } else if (algorithm == "logisticRegression") {
        val duration = measureTime { LogisticRegressionTrainingPipeline(cfg = cfg).execute() }
        ProjectLogger.logger.info("Logistic Regression training pipeline duration: ${duration.inWholeSeconds} seconds")
    } else if (algorithm == "deepLearningClassifier") {
        val duration = measureTime { DeepLearningTrainingPipeline(cfg = cfg).execute() }
        ProjectLogger.logger.info("Deep Learning training pipeline duration: ${duration.inWholeSeconds} seconds")
    } else {
        ProjectLogger.logger.info("No valid algorithm specified in config file")
    }
}