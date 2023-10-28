package util

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.io.readCSV

fun readDataFrameCSV(path: String): DataFrame<*> {
    return DataFrame.readCSV(path)
}
