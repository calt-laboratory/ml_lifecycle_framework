package training

import kotlin.concurrent.thread


fun multipleTrainingPipelineRunner(algorithmsList: List<String>) {
    val threads = algorithmsList.map { algorithm -> thread { trainingPipelineRunner(algorithm = algorithm) } }
    threads.forEach { it.join() }
}
