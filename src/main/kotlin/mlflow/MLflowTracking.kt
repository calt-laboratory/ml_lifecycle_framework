package mlflow

import constants.MLFLOW_TRACKING_URI
import httpServices.isMlflowServerRunning
import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.MlflowContext
import runCommand
import java.util.concurrent.TimeUnit


/**
 * Logs a metric, a parameter and a tag to MLflow.
 */
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
        println("Logged $metricKey: $metricValue in MLflow")
    }

    if (paramKey != null && paramValue != null) {
        client.logParam(runID, paramKey, paramValue)
        println("Logged $paramKey: $paramValue in MLflow")
    }

    if (tagKey != null && tagValue != null) {
        client.setTag(runID, tagKey, tagValue)
        println("Logged $tagKey: $tagValue in MLflow")
    }
}

/**
 * Gets a Mlflow client based on the MLFLOW_TRACKING_URI and checks if the MLflow server is running.
 * @return Pair of MlflowClient and Boolean (if the MLflow server is running)
 */
fun getMlflowClient() : Pair<MlflowClient, Boolean> {
    return try {
        val client = MlflowContext(MLFLOW_TRACKING_URI).client
        println("MLflow tracking URI is set to $MLFLOW_TRACKING_URI")
        val isMlflowServerRunning = isMlflowServerRunning(MLFLOW_TRACKING_URI)
        Pair(client, isMlflowServerRunning)
    } catch (e: IllegalArgumentException) {
        println("MLflow tracking URI is not set. Using local host.")
        val localHost = "http://127.0.0.1:5000"
        val isMlflowServerRunning = isMlflowServerRunning(localHost)
        Pair(MlflowContext(localHost).client, isMlflowServerRunning)
    }
}

/**
 * Gets a Mlflow experiment by name or creates a new one if it does not exist.
 */
fun getOrCreateMlflowExperiment(
    name: String,
    mlflowClient: MlflowClient,
    isMlflowServerRunning: Boolean,
    ) : Pair<MlflowClient, String?>
{
    if (!isMlflowServerRunning) {
        startMlflowServer()
    }
    var experimentID: String

    try {
        experimentID = mlflowClient.getExperimentByName(name).get().experimentId
        println("Experiment '$name' was found")
    } catch (e: NoSuchElementException) {
        experimentID = mlflowClient.createExperiment(name)
        println("New experiment '$name' created")
    }

    val runInfo = mlflowClient.createRun(experimentID)
    return Pair(mlflowClient, runInfo.runId)
}

/**
 * Starts a MLflow Tracking using a cli command based on the operating system (Linux, Mac, Windows).
 */
fun startMlflowServer() {
    val os = System.getProperty("os.name").lowercase()
    println("Starting MLflow Tracking server...")
    if (os.contains("linux") or os.contains("mac")) {
        println("Using operating system: $os")
        "mlflow server".runCommand()
        println("MLflow Tracking server is running")
    } else if (os.contains("windows")) {
        println("Using operating system: $os")
        Runtime.getRuntime().exec("mlflow server").waitFor(25, TimeUnit.SECONDS)
        println("MLflow Tracking server is running")
    }
}
