package mlflow

import constants.MLFLOW_TRACKING_URI
import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.MlflowContext

fun logMlflowInformation(
    client: MlflowClient,
    runID: String?,
    metricKey: String?,
    metricValue: Double?,
    paramKey: String?,
    paramValue: String?,
    tagKey: String?,
    tagValue: String?,
    ) {

    if (metricKey != null && metricValue != null) {
        client.logMetric(runID, metricKey, metricValue)
    }

    if (paramKey != null && paramValue != null) {
        client.logParam(runID, paramKey, paramValue)
    }

    if (tagKey != null && tagValue != null) {
        client.setTag(runID, tagKey, tagValue)
    }
}


fun getMlflowClient() : MlflowClient {
    return MlflowContext(MLFLOW_TRACKING_URI).client
}


fun createOrGetMlflowExperiment(name: String, mlflowClient: MlflowClient) : Pair<MlflowClient, String?> {

    var experimentID: String? = null
    try {
        experimentID = mlflowClient.getExperimentByName(name).get().experimentId
        println("Experiment $name was found")
    } catch (e: NoSuchElementException) {
        experimentID = mlflowClient.createExperiment(name)
        println("New experiment $name created")
    }

    val runInfo = mlflowClient.createRun(experimentID)

    return Pair(mlflowClient, runInfo.runId)
}


