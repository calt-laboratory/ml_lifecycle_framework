package training

import dataProcessing.dataPreProcessing
import dataProcessing.trainTestSplit
import smile.classification.LogisticRegression
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

    val fittedLogisticRegressionModel = LogisticRegression.fit(xTrainDoubleArray, yTrainIntArray)
    val predictions = fittedLogisticRegressionModel.predict(xTestDoubleArray)
    val accuracy = Accuracy.of(yTestIntArray, predictions)
    println("Accuracy: $accuracy")
}