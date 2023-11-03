package util

import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.DataRow
import org.jetbrains.kotlinx.dataframe.api.rows


/**
 * Extension function to convert Kotlin DataFrame DataRow type a DoubleArray.
 */
fun <T> DataRow<T>.columnsAsDoubleArray(): DoubleArray {
    return this.values().map { it.toString().toDouble() }.toDoubleArray()
}

/**
 * Extension function to convert Kotlin DataFrame type to a 2D array of doubles.
 */
fun <T> DataFrame<T>.toDoubleArray(): Array<DoubleArray> {
    return this.rows()
        .map { it.columnsAsDoubleArray() }
        .toTypedArray()
}

/**
 * Extension function to convert Kotlin DataFrame DataColumn to IntArray.
 */
fun <T> DataColumn<T>.toIntArray(): IntArray {
    return values().map { it.toString().toInt() }.toIntArray()
}


fun convertKotlinDFToSmileDF(df: DataFrame<Any?>): smile.data.DataFrame {
    val numCols = df.columnsCount()
    val numRows = df.rows().count()
//    val attributes = Array(numCols) { NominalScale("Column $it") }

    val data = Array(numCols) { DoubleArray(numRows) }

    for (col in 0 until numCols) {
        for (row in 0 until numRows) {
            val value = df[row, col] as Double
            data[col][row] = value
        }
    }

    val smileDf = smile.data.DataFrame.of(data)
    return smileDf
}
