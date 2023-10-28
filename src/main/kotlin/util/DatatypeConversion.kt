package util

import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.DataRow
import org.jetbrains.kotlinx.dataframe.api.rows

fun <T> DataRow<T>.columnsAsDoubleArray(): DoubleArray {
    /*
    Converts the type of the DataRow from the Kotlin DataFrame library to a double array.
     */
    return this.values().map { it.toString().toDouble() }.toDoubleArray()
}

fun <Type> DataFrame<Type>.toDoubleArray(): Array<DoubleArray> {
    /*
    * Converts the type of Kotlin Dataframe to a 2D array of doubles.
    * Replaces the convertDataFrameToDoubleArrayArray() function.
     */
    return this.rows()
        .map { it.columnsAsDoubleArray() }
        .toTypedArray()
}


fun <T> DataColumn<T>.toIntArray(): IntArray {
    return values().map { it.toString().toInt() }.toIntArray()
}

/*
fun convertDataFrameToDoubleArrayArray(dataFrame: DataFrame<*>): Array<DoubleArray> {
    val numRows = dataFrame.rowsCount()
    val numCols = dataFrame.columnsCount()
    val result = Array(numRows) { DoubleArray(numCols) }

    for (i in 0 until numRows) {
        val row = dataFrame[i]
        for (j in 0 until numCols) {
            result[i][j] = row[j].toString().toDouble()
        }
    }
    return result
}
*/


/*
fun convertDataColumnToIntArray(dataColumn: DataColumn<*>): IntArray {
    val result = IntArray(dataColumn.size())
    for (i in 0 until dataColumn.size()) {
        result[i] = dataColumn[i].toString().toInt()
    }
    return result
}
*/

