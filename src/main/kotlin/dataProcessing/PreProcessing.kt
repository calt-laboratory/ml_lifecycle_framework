package dataProcessing

import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.count
import org.jetbrains.kotlinx.dataframe.api.map
import org.jetbrains.kotlinx.dataframe.api.remove
import org.jetbrains.kotlinx.dataframe.api.replace
import org.jetbrains.kotlinx.dataframe.api.with
import kotlin.random.Random

data class PreProcessedDataset(val df: DataFrame<Any?>, val xData: DataFrame<Any?>, val yData: DataColumn<*>)

/**
 *
 */
fun dataPreProcessing(df: DataFrame<*>): PreProcessedDataset {
    // Remove id column
    val dfWithoutID = df.remove("id")
    // Map B (= benign) to 0 and M (= malignant) to 1
    val updatedDiagnosis = dfWithoutID["diagnosis"].map { if (it == "B") 0 else 1 }
    val updatedDataFrame = dfWithoutID.replace { it["diagnosis"] }.with { updatedDiagnosis }
    val yData = updatedDataFrame["diagnosis"]
    val xData = updatedDataFrame.remove("diagnosis")
    return PreProcessedDataset(updatedDataFrame, xData, yData)
}

data class SplitData(
    val xTrain: DataFrame<Any?>,
    val xTest: DataFrame<Any?>,
    val yTrain: DataColumn<Any?>,
    val yTest: DataColumn<Any?>
)

fun trainTestSplit(xData: DataFrame<Any?>, yData: DataColumn<*>, testSize: Double, randomState: Int): SplitData {
    val random = Random(seed = randomState)
    // Create random indices (= integers) aligned w/ the number of rows in the dataframe
    val indices = (0 until xData.count()).shuffled(random)
    // Index where to split the dataframe
    val splitIndex = (xData.count() * (1 - testSize)).toInt()

    // Create indices for train and test set
    val trainIndices = indices.subList(0, splitIndex)
    val testIndices = indices.subList(splitIndex, xData.count())

    // Create train and test sets for features and target
    val xTrain = xData[trainIndices]
    val xTest = xData[testIndices]
    val yTrain = yData[trainIndices]
    val yTest = yData[testIndices]

    return SplitData(xTrain, xTest, yTrain, yTest)
}
