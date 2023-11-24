package mlflow

import org.mlflow.api.proto.Service.RunStatus
import org.mlflow.tracking.MlflowContext

fun logMlflowInfos(
    metricKey: String?,
    metricValue: Double?,
    paramKey: String?,
    paramValue: String?,
    tagKey: String?,
    tagValue: String?,
    experimentName: String,
    ) {

    RunStatus.FINISHED


    val MLFLOW_TRACKING_URI = "http://localhost:5000"

    val mlflowContext = MlflowContext(MLFLOW_TRACKING_URI)
    val client = mlflowContext.client

    val experimentID = client.createExperiment(experimentName)
    val runInfo = client.createRun(experimentID)

    if (metricKey != null && metricValue != null) {
        client.logMetric(runInfo.runId, metricKey, metricValue)
    }

    if (paramKey != null && paramValue != null) {
        client.logParam(runInfo.runId, paramKey, paramValue)
    }

    if (tagKey != null && tagValue != null) {
        client.setTag(runInfo.runId, tagKey, tagValue)
    }
}
