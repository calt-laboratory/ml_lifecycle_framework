import config.readYamlConfig
import constants.PATH_TO_YAML_CONFIG
import training.multipleTrainingPipelineRunner
import training.trainingPipelineRunner

fun main() {

    val cfg = readYamlConfig(filePath = PATH_TO_YAML_CONFIG)

    if (cfg.train.runner == "trainingPipelineRunner") {
        trainingPipelineRunner(algorithm = cfg.train.algorithm)
    } else if (cfg.train.runner == "multipleTrainingPipelineRunner") {
        multipleTrainingPipelineRunner(algorithms = cfg.train.multipleAlgorithms)
    }
}
