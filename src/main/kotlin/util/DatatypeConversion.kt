package util

import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.DataRow
import org.jetbrains.kotlinx.dataframe.api.rows


/**
 * Extension function to convert Kotlin DataFrame DataRow type to DoubleArray.
 */
fun <T> DataRow<T>.columnsAsDoubleArray(): DoubleArray {
    return this.values().map { it.toString().toDouble() }.toDoubleArray()
}

/**
 * Extension function to convert Kotlin DataFrame type to a 2D array of doubles.
 */
fun <T> DataFrame<T>.to2DDoubleArray(): Array<DoubleArray> {
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

fun DataFrame<*>.to2DFloatArray(): Array<FloatArray> {
    return this.rows()
        .map { it.columnsToFloatArray() }
        .toTypedArray()
}

fun DataColumn<*>.toFloatArray(): FloatArray {
    val floatList = this.toList().map { it.toString().toFloat() }
    return floatList.toFloatArray()
}

fun <T> DataRow<T>.columnsToFloatArray(): FloatArray {
    return this.values().map { it.toString().toFloat() }.toFloatArray()
}



