package training

import config.Algorithm
import config.Config
import config.ensembleAlgorithms
import logging.ProjectLogger.logger
import kotlin.time.measureTime

/**
 * Provides various training pipelines (e.g. for ensemble classifiers or logistic regression).
 */
fun trainingPipelineRunner(cfg: Config) {
    // factory method
    val pipeline = getPipeline(cfg = cfg)

    val duration = measureTime { pipeline.execute() }

    logger.info("${pipeline::class.simpleName} duration: ${duration.inWholeSeconds} seconds")
}

private fun getPipeline(cfg: Config): TrainingPipeline {
    return when (cfg.train.algorithm) {
        in ensembleAlgorithms -> EnsembleTrainingPipeline(cfg = cfg)
        Algorithm.LOGISTIC_REGRESSION -> LogisticRegressionTrainingPipeline(cfg = cfg)
        Algorithm.DEEP_LEARNING_CLASSIFIER -> DeepLearningTrainingPipeline(cfg = cfg)
        else -> throw IllegalArgumentException("No valid algorithm specified in config file")
    }
}
