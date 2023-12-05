import config.Config
import training.trainingPipelineRunner


fun main() {
    val cfg = Config.fromYaml()
    trainingPipelineRunner(cfg = cfg)
}
