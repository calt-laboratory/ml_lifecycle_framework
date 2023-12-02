package training


fun multipleTrainingPipelineRunner(algorithms: List<String>) {
    for (algorithm in algorithms) {
        trainingPipelineRunner(algorithm = algorithm)
    }
}