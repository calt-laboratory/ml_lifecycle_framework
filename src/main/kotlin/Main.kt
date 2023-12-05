
import config.Config
import training.trainingPipelineRunner
import kotlin.concurrent.thread

fun main() {
    val cfg = Config.fromYaml()

    val threads = cfg.train.algorithms.map { _ ->
        thread {
            trainingPipelineRunner(cfg = cfg)
        }
    }

    threads.forEach { it.join() }
}
