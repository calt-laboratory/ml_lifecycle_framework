package azure

import com.azure.storage.blob.BlobClient
import com.azure.storage.blob.BlobClientBuilder

@Deprecated(message = "Does not work by now.")
fun getBlobClient(storageConnectionString: String, blobName: String): BlobClient {

    val blobClient = BlobClientBuilder()
        .connectionString(storageConnectionString)
        .blobName(blobName)
        .buildClient()

    return blobClient
}

fun downloadFileFromBlob(blobClient: BlobClient, filePath: String) {
    blobClient.downloadToFile(filePath)
}

fun uploadFileToBlob(blobClient: BlobClient, filePath: String) {
    blobClient.uploadFromFile(filePath, true)
}
