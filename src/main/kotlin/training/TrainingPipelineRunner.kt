package training

import config.Algorithm
import config.Config
import config.ensembleAlgorithms
import logging.ProjectLogger.logger
import kotlin.time.measureTime


/**
 * Runs one or more training pipelines based on the algorithms list specified in the config file.
 * @param cfg Configurations
 */
fun trainingPipelineRunner(cfg: Config) {

    // TODO: Parallelize training pipelines
    val duration = measureTime {
        for (algorithm in cfg.train.algorithms) {
            when (algorithm) {
                in ensembleAlgorithms -> EnsembleTrainingPipeline(cfg = cfg, algorithm = algorithm)
                Algorithm.LOGISTIC_REGRESSION -> LogisticRegressionTrainingPipeline(cfg = cfg, algorithm = algorithm)
                Algorithm.DEEP_LEARNING_CLASSIFIER -> DeepLearningTrainingPipeline(cfg = cfg, algorithm = algorithm)
                else -> throw IllegalArgumentException("No valid algorithm specified in config file")
            }.execute()
        }
    }
    logger.info("Training pipelines duration: $duration")
}
