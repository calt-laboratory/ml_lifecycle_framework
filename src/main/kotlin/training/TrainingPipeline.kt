package training

import azure.downloadFileFromBlob
import azure.getBlobClientConnection
import config.readYamlConfig
import dataProcessing.dataPreProcessing
import dataProcessing.trainTestSplit
import dataProcessing.trainTestSplitForSmile
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import util.PATH_TO_DATASET
import util.PATH_TO_PREPROCESSED_DATASET
import util.PATH_TO_PREPROCESSED_SMILE_Y_TEST_DATA
import util.PATH_TO_PREPROCESSED_TEST_DATASET
import util.PATH_TO_PREPROCESSED_TRAIN_DATASET
import util.PATH_TO_PREPROCESSED_X_DATA
import util.PATH_TO_PREPROCESSED_Y_DATA
import util.PATH_TO_YAML_CONFIG
import util.RAW_DATA_BLOB_CONTAINER_NAME
import util.RAW_FILE_NAME
import util.STORAGE_CONNECTION_STRING
import util.readCSVWithSmile
import util.readDataFrameAsCSV
import util.storeDataFrameAsCSV
import util.toDoubleArray
import util.toIntArray

/**
 * Comprises all preprocessing steps and the training of the model.
 */
fun trainingPipeline() {
    // Read config yaml file
    val cfg = readYamlConfig(filePath = PATH_TO_YAML_CONFIG)

    val blobClient = getBlobClientConnection(
        storageConnectionString = STORAGE_CONNECTION_STRING,
        blobContainerName = RAW_DATA_BLOB_CONTAINER_NAME,
        fileName = RAW_FILE_NAME,
    )
    downloadFileFromBlob(blobClient = blobClient, filePath = PATH_TO_DATASET)

    val data = readDataFrameAsCSV(path = PATH_TO_DATASET)
    val (preProcessedDF, xData, yData) = dataPreProcessing(df = data)

    // TODO: Implement connection to Blob to store preprocessed data there

    storeDataFrameAsCSV(df = preProcessedDF, path = PATH_TO_PREPROCESSED_DATASET)
    storeDataFrameAsCSV(df = xData, path = PATH_TO_PREPROCESSED_X_DATA)
    storeDataFrameAsCSV(df = yData.toDataFrame(), path = PATH_TO_PREPROCESSED_Y_DATA)
    Thread.sleep(3000)

    val prePreProcessedXData = readDataFrameAsCSV(path = PATH_TO_PREPROCESSED_X_DATA)
    val prePreProcessedYData = readDataFrameAsCSV(path = PATH_TO_PREPROCESSED_Y_DATA)

    val (xTrain, xTest, yTrain, yTest) = trainTestSplit(
        xData = prePreProcessedXData,
        yData = prePreProcessedYData["diagnosis"],
        testSize = cfg.preProcessing.testSize,
        randomState = cfg.preProcessing.seed,
    )

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

fun trainingPipelineForEnsembleClassifiers() {
    // Read config yaml file
    val cfg = readYamlConfig(filePath = PATH_TO_YAML_CONFIG)

    val blobClient = getBlobClientConnection(
        storageConnectionString = STORAGE_CONNECTION_STRING,
        blobContainerName = RAW_DATA_BLOB_CONTAINER_NAME,
        fileName = RAW_FILE_NAME,
    )
    downloadFileFromBlob(blobClient = blobClient, filePath = PATH_TO_DATASET)

    val data = readDataFrameAsCSV(path = PATH_TO_DATASET)
    val (preProcessedDF, _, _) = dataPreProcessing(df = data)

    val (trainData, testData, _, yTestData) = trainTestSplitForSmile(
        data = preProcessedDF,
        testSize = cfg.preProcessing.testSize,
        randomState = cfg.preProcessing.seed,
    )

    // TODO: Implement connection to Blob to store preprocessed data there

    storeDataFrameAsCSV(df = preProcessedDF, path = PATH_TO_PREPROCESSED_DATASET)
    storeDataFrameAsCSV(df = trainData, path = PATH_TO_PREPROCESSED_TRAIN_DATASET)
    storeDataFrameAsCSV(df = testData, path = PATH_TO_PREPROCESSED_TEST_DATASET)
    storeDataFrameAsCSV(df = yTestData.toDataFrame(), path = PATH_TO_PREPROCESSED_SMILE_Y_TEST_DATA)
    Thread.sleep(3000)

    val preProcessedTrainData = readCSVWithSmile(path = PATH_TO_PREPROCESSED_TRAIN_DATASET)
    val preProcessedTestData = readCSVWithSmile(path = PATH_TO_PREPROCESSED_TEST_DATASET)
    val preProcessedYTestData = readDataFrameAsCSV(path = PATH_TO_PREPROCESSED_SMILE_Y_TEST_DATA)

    var predictions = intArrayOf()

    // TODO: Implement logger to replace print statements
    // TODO: Replace if w/ when

    if (cfg.train.algorithm == "decisionTree") {
        val model = DecisionTreeClassifier(cfg = cfg)
        model.fit(trainDF = preProcessedTrainData)
        predictions = model.predict(testDF = preProcessedTestData)
        println("Decision Tree")
    } else if (cfg.train.algorithm  == "randomForest") {
        val model = RandomForestClassifier(cfg = cfg)
        model.fit(trainDF = preProcessedTrainData)
        predictions = model.predict(testDF = preProcessedTestData)
        println("Random Forest")
    } else if (cfg.train.algorithm  == "adaBoost") {
        val model = AdaBoostClassifier(cfg = cfg)
        model.fit(trainDF = preProcessedTrainData)
        predictions = model.predict(testDF = preProcessedTestData)
        println("AdaBoost")
    }
    val acc = calculateAccuracy(y_true = preProcessedYTestData["diagnosis"].toIntArray(), y_pred = predictions)
    println("Accuracy: $acc")
}
