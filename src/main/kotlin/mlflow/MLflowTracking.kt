package mlflow

import config.Algorithm
import constants.MLFLOW_TRACKING_URI
import datetime.createTimeStamp
import httpServices.isMlflowServerRunning
import logging.ProjectLogger.logger
import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.MlflowContext
import runCommand


/**
 * Logs metrics, a parameter and a tag to MLflow.
 */
fun logMlflowInformation(
    client: MlflowClient,
    runID: String,
    metrics: Map<String, Double?>,
    paramKey: String?,
    paramValue: String?,
    tagKey: String?,
    tagValue: String?,
    ) {

    metrics.forEach { (key, value) ->
        if (value != null) {
            client.logMetric(runID, key, value)
            logger.info("Logged '$key: $value' in MLflow")
        }
    }

    if (paramKey != null && paramValue != null) {
        client.logParam(runID, paramKey, paramValue)
        logger.info("Logged '$paramKey: $paramValue' in MLflow")
    }

    if (tagKey != null && tagValue != null) {
        client.setTag(runID, tagKey, tagValue)
        logger.info("Logged '$tagKey: $tagValue' in MLflow")
    }
}


/**
 * Defines a custom MLflow run name for the current run.
 */
fun defineMLflowRunName(client: MlflowClient, runID: String, algorithm: Algorithm) {
    val runName = createTimeStamp() + "_$algorithm"
    client.setTag(runID, "mlflow.runName", runName)
}


/**
 * Gets a Mlflow client based on the MLFLOW_TRACKING_URI and checks if the MLflow server is running.
 * @return Pair of MlflowClient and Boolean (if the MLflow server is running)
 */
fun getMlflowClient() : Pair<MlflowClient, Boolean> {
    return try {
        val client = MlflowContext(MLFLOW_TRACKING_URI).client
        logger.info("MLflow tracking URI is set to $MLFLOW_TRACKING_URI")
        val isMlflowServerRunning = isMlflowServerRunning(MLFLOW_TRACKING_URI)
        Pair(client, isMlflowServerRunning)
    } catch (e: IllegalArgumentException) {
        logger.info("MLflow tracking URI is not set. Using local host.")
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
    ) : String
{
    if (!isMlflowServerRunning) {
        startMlflowServer()
    }

    var experimentID: String
    try {
        experimentID = mlflowClient.getExperimentByName(name).get().experimentId
        logger.info("MLflow experiment '$name' was found")
    } catch (e: NoSuchElementException) {
        experimentID = mlflowClient.createExperiment(name)
        logger.info("New experiment '$name' created")
    }

    val runInfo = mlflowClient.createRun(experimentID)
    return runInfo.runId
}


/**
 * Starts a MLflow Tracking using a cli command based on the operating system (Linux, Mac, Windows).
 */
fun startMlflowServer() {
    val os = System.getProperty("os.name").lowercase()
    logger.info("Starting MLflow Tracking server...")

    if (os.contains("linux") or os.contains("mac")) {
        logger.info("Using operating system: $os")
        "mlflow server".runCommand()
        logger.info("MLflow Tracking server is running")
    } else if (os.contains("windows")) {
        logger.info("Using operating system: $os")
        try {
            Runtime.getRuntime().exec("mlflow server")
            logger.info("MLflow Tracking server is running")
        } catch (e: java.io.IOException) {
            val errorMessage = "Error while starting MLflow server: ${e.message}"
            logger.error(errorMessage)
            logger.info("Trying to install mlflow and start MLflow server again...")
            Runtime.getRuntime().exec("pip install mlflow")
            Runtime.getRuntime().exec("mlflow server")
            logger.info("MLflow Tracking server is running")
        }
    } else {
        logger.error("Unsupported operating system: $os")
    }
}
