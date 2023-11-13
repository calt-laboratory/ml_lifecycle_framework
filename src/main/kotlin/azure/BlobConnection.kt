package azure

import com.azure.storage.blob.BlobClient
import com.azure.storage.blob.BlobServiceClientBuilder


/**
 * Returns a BlobClient object to access a blob in a container.
 * @param storageConnectionString Connection string to Azure Storage Account
 * @param blobContainerName Name of the blob container
 * @param fileName Name of the blob
 */
fun getBlobClientConnection(storageConnectionString: String, blobContainerName: String, fileName: String): BlobClient {

    // Build connection to Azure Storage Account
    val blobServiceClient = BlobServiceClientBuilder()
        .connectionString(storageConnectionString)
        .buildClient()

    // Get access to container level by Blob container name
    val containerClient = blobServiceClient.getBlobContainerClient(blobContainerName)

    // Get access to Blob level
    return containerClient.getBlobClient(fileName)
}

/**
 * Downloads a file from Blob level.
 * @param blobClient BlobClient object to access a blob in a container
 * @param filePath Path to the file to which the blob should be downloaded
 */
fun downloadFileFromBlob(blobClient: BlobClient, filePath: String) {
    blobClient.downloadToFile(filePath)
}

/**
 * Uploads a file to Blob level.
 * @param blobClient BlobClient object to access a blob in a container
 * @param filePath Path to the file to which the blob should be uploaded
 */
fun uploadFileToBlob(blobClient: BlobClient, filePath: String) {
    blobClient.uploadFromFile(filePath, true)
}
