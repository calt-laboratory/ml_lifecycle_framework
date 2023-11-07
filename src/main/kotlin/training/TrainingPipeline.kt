package training

//import smile.validation.metric.Accuracy
import dataProcessing.dataPreProcessing
import dataProcessing.trainTestSplit
import util.PATH_TO_DATASET
import util.PATH_TO_PREPROCESSED_DATASET
import util.PATH_TO_PREPROCESSED_X_DATASET
import util.PATH_TO_PREPROCESSED_Y_DATASET
import util.SEED
import util.TEST_SIZE
import util.readCSVWithSmile
import util.readDataFrameAsCSV
import util.storeDataFrameAsCSV
import util.storeTargetAsCSV
import util.toDoubleArray
import util.toIntArray

/**
 * Comprises all preprocessing steps and the training of the model.
 */
fun trainingPipeline() {
    val data = readDataFrameAsCSV(path = PATH_TO_DATASET)
    val (preProcessedDF, xData, yData) = dataPreProcessing(df = data)
    storeDataFrameAsCSV(df = preProcessedDF, path = PATH_TO_PREPROCESSED_DATASET)
    Thread.sleep(3000)
    // TODO: Consider to implement an extra function which splits the preprocessed data set into xData and yData
    val prePreProcessedData = readDataFrameAsCSV(path = PATH_TO_PREPROCESSED_DATASET)

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
    val accuracy = calculateAccuracy(y_true = yTestIntArray, y_pred = predictions)
    println("Accuracy: $accuracy")
}

fun trainingPipelineWithSmile() {

    val data = readDataFrameAsCSV(path = PATH_TO_DATASET)
    val (preProcessedDF, xData, yData) = dataPreProcessing(df = data)
    storeDataFrameAsCSV(df = preProcessedDF, path = PATH_TO_PREPROCESSED_DATASET)
    storeDataFrameAsCSV(df = xData, path = PATH_TO_PREPROCESSED_X_DATASET)
    storeTargetAsCSV(target = yData, path = PATH_TO_PREPROCESSED_Y_DATASET)
    Thread.sleep(3000)
    val preProcessedXData = readCSVWithSmile(path = PATH_TO_PREPROCESSED_X_DATASET)

//    val model = cart(Formula.lhs("diagnosis"), preProcessedXData, SplitRule.GINI, 20, 0, 5)
//    println(model)
//
//    val prediction = model.predict(smileTestData)
//    println(prediction)
}