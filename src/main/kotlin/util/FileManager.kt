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
fun readDataFrameAsCSV(path: String): DataFrame<*> {
    return DataFrame.readCSV(path)
}


fun storeDataFrameAsCSV(df: DataFrame<*>, path: String) {
    df.writeCSV(path = path)
}

fun storeTargetAsCSV(target: DataColumn<*>, path: String) {
    val df = target.toDataFrame()
    df.writeCSV(path = path)
}


fun readCSVWithSmile(path: String): smile.data.DataFrame  {
    val df = smile.read.csv(path)
    println(df.structure())
    return df
}
