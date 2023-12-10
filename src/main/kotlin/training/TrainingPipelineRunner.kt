package training

import config.Algorithm
import config.Config
import config.ensembleAlgorithms
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import logging.ProjectLogger.logger
import kotlin.time.measureTime


/**
 * Runs one or more training pipelines based on the algorithms list specified in the config file.
 * @param cfg Configurations
 */
fun trainingPipelineRunner(cfg: Config) = runBlocking {

    val duration = measureTime {
        cfg.train.algorithms.map { algorithm ->
            async(Dispatchers.Default) {
                when (algorithm) {
                    in ensembleAlgorithms -> EnsembleTrainingPipeline(cfg = cfg, algorithm = algorithm)
                    Algorithm.LOGISTIC_REGRESSION -> LogisticRegressionTrainingPipeline(cfg = cfg, algorithm = algorithm)
                    Algorithm.DEEP_LEARNING_CLASSIFIER -> DeepLearningTrainingPipeline(cfg = cfg, algorithm = algorithm)
                    else -> throw IllegalArgumentException("No valid algorithm specified in config file")
                }.execute()
            }
        }.awaitAll()
    }
    logger.info("Training pipelines duration: $duration")
}
