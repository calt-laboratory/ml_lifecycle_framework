package training

import config.readYamlConfig
import constants.PATH_TO_YAML_CONFIG
import logging.ProjectLogger.logger
import kotlin.time.measureTime

/**
 * Provides various training pipelines (e.g. for ensemble classifiers or logistic regression).
 */
fun trainingPipelineRunner(algorithm: String) {
    val cfg = readYamlConfig(filePath = PATH_TO_YAML_CONFIG)

    // TODO: Replace if with when

    if (algorithm in listOf("decisionTree", "randomForest", "adaBoost", "gradientBoosting")) {
        executePipeline(pipeline = EnsembleTrainingPipeline(cfg = cfg))
    } else if (algorithm == "logisticRegression") {
        executePipeline(pipeline = LogisticRegressionTrainingPipeline(cfg = cfg))
    } else if (algorithm == "deepLearningClassifier") {
        executePipeline(pipeline = DeepLearningTrainingPipeline(cfg = cfg))
    } else {
        logger.info("No valid algorithm specified in config file")
    }
}


fun executePipeline(pipeline: TrainingPipeline) {
    val duration = measureTime { pipeline.execute() }
    logger.info("${pipeline::class.simpleName} duration: ${duration.inWholeSeconds} seconds")
}