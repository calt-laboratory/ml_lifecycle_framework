package util

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.io.readCSV

/**
 * Reads CSV file and returns a dataframe.
 * @param path Path to the CSV file
 * @return Unprocessed data frame
 */
fun readDataFrameCSV(path: String): DataFrame<*> {
    return DataFrame.readCSV(path)
}
