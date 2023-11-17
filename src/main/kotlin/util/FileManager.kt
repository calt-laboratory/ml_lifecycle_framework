package util

import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import org.jetbrains.kotlinx.dataframe.io.readCSV
import org.jetbrains.kotlinx.dataframe.io.writeCSV

/**
 * Reads CSV file and returns a dataframe.
 * @param path Path to the CSV file
 * @return Unprocessed data frame
 */
fun readCSVAsKotlinDF(path: String): DataFrame<*> {
    return DataFrame.readCSV(path)
}

fun storeKotlinDFAsCSV(df: DataFrame<*>, path: String) {
    df.writeCSV(path = path)
}

fun storeTargetAsCSV(target: DataColumn<*>, path: String) {
    val df = target.toDataFrame()
    df.writeCSV(path = path)
}

fun readCSVAsSmileDF(path: String): smile.data.DataFrame  {
    return smile.read.csv(path)
}
