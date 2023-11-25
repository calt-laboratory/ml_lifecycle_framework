package mlflow

import constants.MLFLOW_TRACKING_URI
import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.MlflowClientException
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
    return try {
        MlflowContext(MLFLOW_TRACKING_URI).client
    } catch (e: IllegalArgumentException) {
        println("MLflow tracking URI is not set. Using local host.")
        val localHost = "http://127.0.0.1:5000"
        MlflowContext(localHost).client
    }
}


fun createOrGetMlflowExperiment(name: String, mlflowClient: MlflowClient) : Pair<MlflowClient, String?> {

    var experimentID: String? = null
    try {
        experimentID = mlflowClient.getExperimentByName(name).get().experimentId
        println("Experiment $name was found")
    } catch (e: Exception) {
        when (e) {
            is NoSuchElementException -> {
                experimentID = mlflowClient.createExperiment(name)
                println("New experiment $name created: $e")
            }
            is MlflowClientException -> {
                println("MLflow tracking server is not running. Connection to Mlflow Tracking URI failed $e")
            }
            else -> {
                println("Unknown exception: $e")
            }
        }
    }

    val runInfo = mlflowClient.createRun(experimentID)

    return Pair(mlflowClient, runInfo.runId)
}


