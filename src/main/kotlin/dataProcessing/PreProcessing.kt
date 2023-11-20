package dataProcessing

import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.count
import org.jetbrains.kotlinx.dataframe.api.map
import org.jetbrains.kotlinx.dataframe.api.remove
import org.jetbrains.kotlinx.dataframe.api.replace
import org.jetbrains.kotlinx.dataframe.api.with
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import util.to2DFloatArray
import util.toFloatArray
import kotlin.random.Random

data class PreProcessedDataset(val df: DataFrame<Any?>, val xData: DataFrame<Any?>, val yData: DataColumn<*>)

/**
 * Removes the id column and maps the diagnosis column to 0 and 1.
 * @param df Original dataframe
 * @return PreProcessedDataset Contains the dataframe, the features and the target
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

/**
 * Splits the data into train and test sets.
 * @param xData Features
 * @param yData Target
 * @param testSize Size of the test set
 * @param randomState Random state used to create the random indices
 * @return SplitData Contains the train and test set
 */
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

data class SplitDataForSmile(val trainData: DataFrame<Any?>, val testData: DataFrame<Any?>, val xTestData: DataFrame<Any?>, val yTestData: DataColumn<*>)

/**
 * Splits a Kotlin dataframe into train set, test sets, x-test data and y-test to make it usable for Smile.
 * @param data Kotlin Dataframe
 * @param testSize Size of the test set
 * @param randomState Random state used to create the random indices
 * @return Contains the train set, test set, x-test and y-test data
 */
fun trainTestSplitForSmile(data: DataFrame<*>, testSize: Double, randomState: Int): SplitDataForSmile {
    val random = Random(seed = randomState)
    val indices = (0 until data.rowsCount()).shuffled(random)
    val splitIndex = (data.rowsCount() * (1 - testSize)).toInt()

    val trainIndices = indices.subList(0, splitIndex)
    val testIndices = indices.subList(splitIndex, data.rowsCount())

    val trainData = data[trainIndices]
    val testData = data[testIndices]
    val xTestData = testData.remove("diagnosis")
    val yTestData = testData["diagnosis"]

    return SplitDataForSmile(trainData, testData, xTestData, yTestData)
}

/**
 * Converts data into KotlinDL specific format and splits it into train and test sets.
 * @param xData Features
 * @param yData Target
 * @param trainSize Size of the train set
 * @return Contains the train and test
 */
fun trainTestSplitForKotlinDL(xData: DataFrame<*>, yData: DataColumn<*>, trainSize: Double) : Pair<OnHeapDataset, OnHeapDataset> {
    val dataset = OnHeapDataset.create(features = xData.to2DFloatArray(), labels = yData.toFloatArray())
    val (train, test) = dataset.split(splitRatio = trainSize)
    return Pair(train, test)
}
