package aws

import aws.sdk.kotlin.runtime.auth.credentials.StaticCredentialsProvider
import aws.sdk.kotlin.services.s3.S3Client
import aws.sdk.kotlin.services.s3.model.GetObjectRequest
import aws.smithy.kotlin.runtime.content.writeToFile
import constants.S3_BUCKET_REGION
import java.io.File

suspend fun downloadFileFromS3(bucketName: String, keyName: String, path: String) {
    val request = GetObjectRequest {
        key = keyName
        bucket = bucketName
    }

    val credentials = StaticCredentialsProvider {
        accessKeyId = System.getenv("AWS_ACCESS_KEY_ID")
        secretAccessKey = System.getenv("AWS_SECRET_ACCESS_KEY")
    }

    S3Client
        .fromEnvironment {
            region = S3_BUCKET_REGION
            credentialsProvider = credentials
        }.use { s3 ->
        s3.getObject(request) { response ->
            val myFile = File(path)
            response.body?.writeToFile(myFile)
            println("Successfully downloaded file from S3 $bucketName to $path")
        }
    }
}