package training

import dataProcessing.dataPreProcessing
import dataProcessing.trainTestSplit
import smile.validation.metric.Accuracy
import util.PATH_TO_DATASET
import util.SEED
import util.TEST_SIZE
import util.readDataFrameCSV
import util.toDoubleArray
import util.toIntArray

/**
 * Comprises all preprocessing steps and the training of the model.
 */
fun trainingPipeline() {
    val data = readDataFrameCSV(path = PATH_TO_DATASET)

    val (_, xData, yData) = dataPreProcessing(df = data)

    val (xTrain, xTest, yTrain, yTest) = trainTestSplit(xData = xData, yData = yData, testSize = TEST_SIZE, randomState = SEED)

    // Convert training dataframes of type Kotlin DataFrame to make them compatible with Smile
    val xTrainDoubleArray = xTrain.toDoubleArray()
    val xTestDoubleArray = xTest.toDoubleArray()

    // Convert test columns of type Kotlin DataColumn to make them compatible with Smile
    val yTrainIntArray = yTrain.toIntArray()
    val yTestIntArray = yTest.toIntArray()

    // Train the model
    val logisticRegression = LogisticRegressionModel()
    logisticRegression.fit(xTrain = xTrainDoubleArray, yTrain = yTrainIntArray)

    // Calculate y-predictions based on the x-test set
    val predictions = logisticRegression.predictModel(xTest = xTestDoubleArray)

    // Calculate accuracy of y-predictions compared to y-test set
    val accuracy = Accuracy.of(yTestIntArray, predictions)
    println("Accuracy: $accuracy")
}