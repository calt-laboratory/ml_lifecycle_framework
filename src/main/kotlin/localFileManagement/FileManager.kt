package localFileManagement

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import logging.ProjectLogger
import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import org.jetbrains.kotlinx.dataframe.io.readCSV
import org.jetbrains.kotlinx.dataframe.io.writeCSV
import java.io.File
import java.util.Date


/**
 * Reads CSV file and returns a dataframe.
 * @param path Path to the CSV file
 * @return Unprocessed data frame
 */
fun readCSVAsKotlinDF(path: String): DataFrame<*> {
    return DataFrame.readCSV(path)
}


/**
 * Stores a Kotlin dataframe as CSV file.
 * @param df Kotlin dataframe to store
 * @path Path where to store the CSV file
 */
fun storeKotlinDFAsCSV(df: DataFrame<*>, path: String) {
    df.writeCSV(path = path)
}


/**
 * Stores a Kotlin dataframe as CSV file asynchronously.
 * @param df Kotlin dataframe to store
 * @path Path where to store the CSV file
 */
suspend fun storeKotlinDFAsCSVAsync(df: DataFrame<*>, path: String) = withContext(Dispatchers.IO) {
    storeKotlinDFAsCSV(df = df, path = path)
}


/**
 * Reads asynchronously a CSV file and returns the data as a Kotlin dataframe.
 * @param path Path to the CSV file
 * @return Kotlin dataframe
 */
suspend fun readCSVAsKotlinDFAsync(path: String): DataFrame<*> {
    return withContext(Dispatchers.IO) {
        readCSVAsKotlinDF(path = path)
    }
}


/**
 * Stores a target column from a Kotlin dataframe as CSV file.
 * @param target column from a Kotlin dataframe
 * @path Path where to store the CSV file
 */
fun storeTargetAsCSV(target: DataColumn<*>, path: String) {
    val df = target.toDataFrame()
    df.writeCSV(path = path)
}


/**
 * Reads a CSV file as Smile dataframe.
 * @param path Path to the CSV file
 * @return Smile dataframe
 */
fun readCSVAsSmileDF(path: String): smile.data.DataFrame  {
    return smile.read.csv(path)
}


/**
 * Reads asynchronously a CSV file and returns the data as a Smile dataframe.
 * @param path Path to the CSV file
 * @return Smile dataframe
 */
suspend fun readCSVAsSmileDFAsync(path: String): smile.data.DataFrame {
    return withContext(Dispatchers.IO) {
        readCSVAsSmileDF(path)
    }
}


/**
 * Deletes all folders in a directory that are older than 2 days.
 * @param folderPath: Path to the directory that contains the folders to be deleted
 */
fun deleteFolder(folderPath: File) {

    val folders = folderPath.listFiles() ?: emptyArray()

    for(folder in folders) {
        val lastModified = Date(folder.lastModified())
        val currentDateMinus2Days = Date(System.currentTimeMillis() - 2 * 24 * 60 * 60 * 1000)
        if (currentDateMinus2Days > lastModified) {
            folder.deleteRecursively()
            ProjectLogger.logger.info("Deleted results folder: $folder")
        }
    }
}
