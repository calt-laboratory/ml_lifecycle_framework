package httpServices

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
        println("MLflow tracking server is running")
        response.status == Status.OK
    } catch (e: Exception) {
        println("Exception: $e")
        println("MLflow tracking server is not running")
        false
    }
}
