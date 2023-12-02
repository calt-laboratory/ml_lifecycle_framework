
import config.readYamlConfig
import constants.PATH_TO_YAML_CONFIG
import logging.ProjectLogger.logger
import training.multipleTrainingPipelineRunner
import training.trainingPipelineRunner
import kotlin.time.measureTime


fun main() {

    val cfg = readYamlConfig(filePath = PATH_TO_YAML_CONFIG)

    val singleRunner = RunnerType.TRAINING_PIPELINE_RUNNER
    val multipleTrainingPipelineRunner = RunnerType.MULTIPLE_TRAINING_PIPELINE_RUNNER

    if (cfg.train.runner == singleRunner.runnerName) {
        trainingPipelineRunner(algorithm = cfg.train.algorithm)
    } else if (cfg.train.runner == multipleTrainingPipelineRunner.runnerName) {
        val duration = measureTime { multipleTrainingPipelineRunner(algorithmsList = cfg.train.multipleAlgorithms) }
        logger.info("Multiple training pipeline duration: ${duration.inWholeSeconds} seconds")
    }


}

enum class RunnerType(val runnerName: String) {
    TRAINING_PIPELINE_RUNNER("trainingPipelineRunner"),
    MULTIPLE_TRAINING_PIPELINE_RUNNER("multipleTrainingPipelineRunner")
}
