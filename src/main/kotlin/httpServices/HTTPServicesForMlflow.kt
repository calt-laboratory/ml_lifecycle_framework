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
    val response = OkHttp()(Request(Method.GET, mlflowTrackingUri))
    return response.status == Status.OK
}

fun stopMlflowServer() {
    val client = OkHttp()
    val request = Request(Method.POST, "http://127.0.0.1:5000/shutdown")
    client(request)
}

