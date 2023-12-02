package httpServices

import logging.ProjectLogger.logger
import org.http4k.client.OkHttp
import org.http4k.core.Method
import org.http4k.core.Request
import org.http4k.core.Status


/**
 * Checks if MLflow tracking server is running.
 * @param mlflowTrackingUri MLflow tracking URI
 * @return Boolean if the MLflow tracking server is running
 */
fun isMlflowServerRunning(mlflowTrackingUri: String) : Boolean {
    return try {
        val response = OkHttp()(Request(Method.GET, mlflowTrackingUri))
        logger.info("MLflow tracking server is running")
        response.status == Status.OK
    } catch (e: Exception) {
        logger.info("MLflow tracking server is not running")
        logger.info("Exception: $e")
        false
    }
}

fun stopMlflowServer() {
    val client = OkHttp()
    val request = Request(Method.POST, "http://127.0.0.1:5000/shutdown")
    client(request)
}

