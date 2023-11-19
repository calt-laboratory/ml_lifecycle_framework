package training

import azure.downloadFileFromBlob
import azure.getBlobClientConnection
import azure.uploadFileToBlob
import config.Config
import config.readYamlConfig
import constants.PATH_TO_DATASET
import constants.PATH_TO_PREPROCESSED_DATASET
import constants.PATH_TO_PREPROCESSED_SMILE_Y_TEST_DATA
import constants.PATH_TO_PREPROCESSED_TEST_DATASET
import constants.PATH_TO_PREPROCESSED_TRAIN_DATASET
import constants.PATH_TO_PREPROCESSED_X_DATA
import constants.PATH_TO_PREPROCESSED_Y_DATA
import constants.PATH_TO_YAML_CONFIG
import constants.PREPROCESSED_FILE_NAME
import constants.PREPROCESSED_SMILE_Y_TEST_DATASET_FILE_NAME
import constants.PREPROCESSED_TEST_DATASET_FILE_NAME
import constants.PREPROCESSED_TRAIN_DATASET_FILE_NAME
import constants.PROCESSED_DATA_BLOB_CONTAINER_NAME
import constants.RAW_DATA_BLOB_CONTAINER_NAME
import constants.RAW_FILE_NAME
import dataProcessing.dataPreProcessing
import dataProcessing.trainTestSplit
import dataProcessing.trainTestSplitForKotlinDL
import dataProcessing.trainTestSplitForSmile
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.slf4j.LoggerFactory
import util.readCSVAsKotlinDF
import util.readCSVAsKotlinDFAsync
import util.readCSVAsSmileDFAsync
import util.storeKotlinDFAsCSV
import util.storeKotlinDFAsCSVAsync
import util.to2DDoubleArray
import util.toIntArray


private val logger = LoggerFactory.getLogger("TrainingPipeline")

/**
 * Provides various training pipelines (e.g. for ensemble classifiers or logistic regression).
 */
fun trainingPipeline() {
    val cfg = readYamlConfig(filePath = PATH_TO_YAML_CONFIG)

    if (cfg.train.algorithm in listOf("decisionTree", "randomForest", "adaBoost", "gradientBoosting")) {
        ensembleTrainingPipeline(cfg = cfg)
    } else if (cfg.train.algorithm == "logisticRegression") {
        logisticRegressionTrainingPipeline(cfg = cfg)
    } else if (cfg.train.algorithm == "deepLearningClassifier") {
        deepLearningTrainingPipeline(cfg = cfg)
    } else {
        println("No valid algorithm specified in config file.")
    }
}


/**
 * Comprises all preprocessing steps and the training/prediction for an ensemble classifier.
 */
fun ensembleTrainingPipeline(cfg: Config) = runBlocking {
    logger.info("Starting ensemble training pipeline...")
    val storageConnectionString = System.getenv("STORAGE_CONNECTION_STRING")

    val blobClient = getBlobClientConnection(
        storageConnectionString = storageConnectionString,
        blobContainerName = RAW_DATA_BLOB_CONTAINER_NAME,
        fileName = RAW_FILE_NAME,
    )
    downloadFileFromBlob(blobClient = blobClient, filePath = PATH_TO_DATASET)

    val data = readCSVAsKotlinDF(path = PATH_TO_DATASET)
    val (preProcessedDF, _, _) = dataPreProcessing(df = data)

    val (trainData, testData, _, yTestData) = trainTestSplitForSmile(
        data = preProcessedDF,
        testSize = cfg.preProcessing.testSize,
        randomState = cfg.preProcessing.seed,
    )

    // Store Kotlin DataFrame locally
    val dataframesAndPaths = listOf(
        preProcessedDF to PATH_TO_PREPROCESSED_DATASET,
        trainData to PATH_TO_PREPROCESSED_TRAIN_DATASET,
        testData to PATH_TO_PREPROCESSED_TEST_DATASET,
        yTestData.toDataFrame() to PATH_TO_PREPROCESSED_SMILE_Y_TEST_DATA
    )
    val preProcessedKotlinDFToStore = dataframesAndPaths.map { (df, path) ->
        async { storeKotlinDFAsCSVAsync(df, path) }
    }
    preProcessedKotlinDFToStore.awaitAll()

    // Upload preprocessed data to Blob
    val filesToUpload = listOf(
        Pair(PREPROCESSED_FILE_NAME, PATH_TO_PREPROCESSED_DATASET),
        Pair(PREPROCESSED_TRAIN_DATASET_FILE_NAME, PATH_TO_PREPROCESSED_TRAIN_DATASET),
        Pair(PREPROCESSED_TEST_DATASET_FILE_NAME, PATH_TO_PREPROCESSED_TEST_DATASET),
        Pair(PREPROCESSED_SMILE_Y_TEST_DATASET_FILE_NAME, PATH_TO_PREPROCESSED_SMILE_Y_TEST_DATA),
    )

    val deferredUploads = filesToUpload.map { (fileName, localFilePath) ->
        async {
            val blobClientPreProcessedData = getBlobClientConnection(
                storageConnectionString = storageConnectionString,
                blobContainerName = PROCESSED_DATA_BLOB_CONTAINER_NAME,
                fileName = fileName
            )
            uploadFileToBlob(blobClient = blobClientPreProcessedData, filePath = localFilePath)
        }
    }
    deferredUploads.awaitAll()

    // Read in preprocessed data
    val preProcessedTrainData = async { readCSVAsSmileDFAsync(PATH_TO_PREPROCESSED_TRAIN_DATASET) }.await()
    val preProcessedTestData = async { readCSVAsSmileDFAsync(PATH_TO_PREPROCESSED_TEST_DATASET) }.await()
    val preProcessedYTestData = async { readCSVAsKotlinDFAsync(PATH_TO_PREPROCESSED_SMILE_Y_TEST_DATA) }.await()

    var predictions = intArrayOf()

    // TODO: Implement logger to replace print statements

    when(cfg.train.algorithm) {
        "decisionTree" -> {
            val model = DecisionTreeClassifier(cfg = cfg)
            model.fit(trainDF = preProcessedTrainData)
            predictions = model.predict(testDF = preProcessedTestData)
            logger.info("Decision Tree training started.")
            println("Decision Tree")
        }
        "randomForest" -> {
            val model = RandomForestClassifier(cfg = cfg)
            model.fit(trainDF = preProcessedTrainData)
            predictions = model.predict(testDF = preProcessedTestData)
            logger.info("Random Forest training started.")
            println("Random Forest")
        }
        "adaBoost" -> {
            logger.info("AdaBoost training started.")
            val model = AdaBoostClassifier(cfg = cfg)
            model.fit(trainDF = preProcessedTrainData)
            predictions = model.predict(testDF = preProcessedTestData)
            println("AdaBoost")
        }
        "gradientBoosting" -> {
            logger.info("Gradient Boosting training started.")
            val model = GradientBoostingClassifier(cfg = cfg)
            model.fit(trainDF = preProcessedTrainData)
            predictions = model.predict(testDF = preProcessedTestData)
            println("Gradient Boosting")
        }
    }

    val acc = calculateAccuracy(yTrue = preProcessedYTestData["diagnosis"].toIntArray(), yPred = predictions)
    println("Accuracy: $acc")
}


/**
 * Comprises all preprocessing steps and the training/prediction for Logistic Regression.
 */
fun logisticRegressionTrainingPipeline(cfg: Config) {

    val storageConnectionString = System.getenv("STORAGE_CONNECTION_STRING")

    val blobClient = getBlobClientConnection(
        storageConnectionString = storageConnectionString,
        blobContainerName = RAW_DATA_BLOB_CONTAINER_NAME,
        fileName = RAW_FILE_NAME,
    )
    downloadFileFromBlob(blobClient = blobClient, filePath = PATH_TO_DATASET)

    val data = readCSVAsKotlinDF(path = PATH_TO_DATASET)
    val (preProcessedDF, xData, yData) = dataPreProcessing(df = data)

    // TODO: Implement connection to Blob to store preprocessed data there
    // TODO: Integrate coroutines like in ensembleTrainingPipeline()

    storeKotlinDFAsCSV(df = preProcessedDF, path = PATH_TO_PREPROCESSED_DATASET)
    storeKotlinDFAsCSV(df = xData, path = PATH_TO_PREPROCESSED_X_DATA)
    storeKotlinDFAsCSV(df = yData.toDataFrame(), path = PATH_TO_PREPROCESSED_Y_DATA)
    Thread.sleep(3000)

    val prePreProcessedXData = readCSVAsKotlinDF(path = PATH_TO_PREPROCESSED_X_DATA)
    val prePreProcessedYData = readCSVAsKotlinDF(path = PATH_TO_PREPROCESSED_Y_DATA)

    val (xTrain, xTest, yTrain, yTest) = trainTestSplit(
        xData = prePreProcessedXData,
        yData = prePreProcessedYData["diagnosis"],
        testSize = cfg.preProcessing.testSize,
        randomState = cfg.preProcessing.seed,
    )

    // Convert training dataframes of type Kotlin DataFrame to make them compatible with Smile
    val xTrainDoubleArray = xTrain.to2DDoubleArray()
    val xTestDoubleArray = xTest.to2DDoubleArray()

    // Convert test columns of type Kotlin DataColumn to make them compatible with Smile
    val yTrainIntArray = yTrain.toIntArray()
    val yTestIntArray = yTest.toIntArray()

    // Train the model
    val logisticRegression = LogisticRegressionModel(cfg = cfg)
    logisticRegression.fit(xTrain = xTrainDoubleArray, yTrain = yTrainIntArray)
    // Calculate y-predictions based on the x-test set
    val predictions = logisticRegression.predict(xTest = xTestDoubleArray)
    println("Logistic Regression")

    // Calculate accuracy of y-predictions compared to y-test set
    val accuracy = calculateAccuracy(yTrue = yTestIntArray, yPred = predictions)
    println("Accuracy: $accuracy")
}


fun deepLearningTrainingPipeline(cfg: Config) {

    val storageConnectionString = System.getenv("STORAGE_CONNECTION_STRING")

    val blobClient = getBlobClientConnection(
        storageConnectionString = storageConnectionString,
        blobContainerName = RAW_DATA_BLOB_CONTAINER_NAME,
        fileName = RAW_FILE_NAME,
    )
    downloadFileFromBlob(blobClient = blobClient, filePath = PATH_TO_DATASET)

    val data = readCSVAsKotlinDF(path = PATH_TO_DATASET)
    val (_, xData, yData) = dataPreProcessing(df = data)

    val (train, test) = trainTestSplitForKotlinDL(xData = xData, yData = yData)

    val deepLearningClassifier = DeepLearningClassifier(cfg = cfg)
    val predictions = deepLearningClassifier.fitAndPredict(xData = train, yData = test)
    val accuracy = predictions.metrics[Metrics.ACCURACY]
    println("Accuracy: $accuracy")
}
