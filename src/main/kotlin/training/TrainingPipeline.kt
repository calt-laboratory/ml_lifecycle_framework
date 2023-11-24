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
import constants.PREPROCESSED_X_DATA_FILE_NAME
import constants.PREPROCESSED_Y_DATA_FILE_NAME
import constants.PROCESSED_DATA_BLOB_CONTAINER_NAME
import constants.RAW_DATA_BLOB_CONTAINER_NAME
import constants.RAW_FILE_NAME
import constants.TRAINING_RESULT_DB_URL
import dataProcessing.dataPreProcessing
import dataProcessing.trainTestSplit
import dataProcessing.trainTestSplitForKotlinDL
import dataProcessing.trainTestSplitForSmile
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import mlflow.logMlflowInfos
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.slf4j.LoggerFactory
import postgres.TrainingResults
import postgres.connectToDB
import postgres.createTable
import postgres.insertTrainingResults
import util.readCSVAsKotlinDF
import util.readCSVAsKotlinDFAsync
import util.readCSVAsSmileDFAsync
import util.round
import util.storeKotlinDFAsCSVAsync
import util.to2DDoubleArray
import util.toIntArray
import java.io.File
import kotlin.time.measureTime


private val logger = LoggerFactory.getLogger("TrainingPipeline")

/**
 * Provides various training pipelines (e.g. for ensemble classifiers or logistic regression).
 */
fun trainingPipeline() {
    val cfg = readYamlConfig(filePath = PATH_TO_YAML_CONFIG)

    // TODO: Replace if with when

    if (cfg.train.algorithm in listOf("decisionTree", "randomForest", "adaBoost", "gradientBoosting")) {
        val duration = measureTime { ensembleTrainingPipeline(cfg = cfg) }
        println("Ensemble training pipeline duration: ${duration.inWholeSeconds} seconds")
    } else if (cfg.train.algorithm == "logisticRegression") {
        val duration = measureTime { logisticRegressionTrainingPipeline(cfg = cfg) }
        println("Logistic Regression training pipeline duration: ${duration.inWholeSeconds} seconds")
    } else if (cfg.train.algorithm == "deepLearningClassifier") {
        val duration = measureTime { deepLearningTrainingPipeline(cfg = cfg) }
        println("Deep Learning training pipeline duration: ${duration.inWholeSeconds} seconds")
    } else {
        println("No valid algorithm specified in config file.")
    }
}


/**
 * Comprises all preprocessing steps and the training/prediction for an ensemble classifier.
 */
fun ensembleTrainingPipeline(cfg: Config) = runBlocking {
    println("Starting ensemble training pipeline...")
    val storageConnectionString = System.getenv("STORAGE_CONNECTION_STRING")

    if (!File(PATH_TO_DATASET).exists()) {
        println("Downloading original dataset from Blob...")
        val blobClient = getBlobClientConnection(
            storageConnectionString = storageConnectionString,
            blobContainerName = RAW_DATA_BLOB_CONTAINER_NAME,
            fileName = RAW_FILE_NAME,
        )
        downloadFileFromBlob(blobClient = blobClient, filePath = PATH_TO_DATASET)
    }

    val data = readCSVAsKotlinDF(path = PATH_TO_DATASET)
    val (preProcessedDF, _, _) = dataPreProcessing(df = data)

    val (trainData, testData, _, yTestData) = trainTestSplitForSmile(
        data = preProcessedDF,
        testSize = cfg.preProcessing.testSize,
        randomState = cfg.preProcessing.seed,
    )

    // Store Kotlin DFs locally
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

    // Store training results in Postgres DB
    connectToDB(dbURL = TRAINING_RESULT_DB_URL)
    createTable(table = TrainingResults)
    insertTrainingResults(algorithmName = cfg.train.algorithm, accuracy = acc)

    // Log MLflow infos
    logMlflowInfos(
        metricKey = "accuracy",
        metricValue = acc,
        paramKey = "algorithm",
        paramValue = cfg.train.algorithm,
        tagKey = "dataset",
        tagValue = "breast_cancer",
        experimentName = "breast_cancer",
    )
}


/**
 * Comprises all preprocessing steps and the training/prediction for Logistic Regression.
 */
fun logisticRegressionTrainingPipeline(cfg: Config) = runBlocking {
    println("Starting the Logistic Regression pipeline...")
    val storageConnectionString = System.getenv("STORAGE_CONNECTION_STRING")

    if(!File(PATH_TO_DATASET).exists()) {
        println("Downloading original dataset from Blob...")
        val blobClient = getBlobClientConnection(
            storageConnectionString = storageConnectionString,
            blobContainerName = RAW_DATA_BLOB_CONTAINER_NAME,
            fileName = RAW_FILE_NAME,
        )
        downloadFileFromBlob(blobClient = blobClient, filePath = PATH_TO_DATASET)
    }

    val data = readCSVAsKotlinDF(path = PATH_TO_DATASET)
    val (preProcessedDF, xData, yData) = dataPreProcessing(df = data)

    // Store Kotlin DF's locally
    val kotlinDFsAndPaths = listOf(
        preProcessedDF to PATH_TO_PREPROCESSED_DATASET,
        xData to PATH_TO_PREPROCESSED_X_DATA,
        yData.toDataFrame() to PATH_TO_PREPROCESSED_Y_DATA,
    )
    val preProcessedKotlinDFsToStore = kotlinDFsAndPaths.map { (df, path) ->
        async { storeKotlinDFAsCSVAsync(df, path) }
    }
    preProcessedKotlinDFsToStore.awaitAll()

    // Upload preprocessed data to Blob
    val filesToUpload = listOf(
        Pair(PREPROCESSED_FILE_NAME, PATH_TO_PREPROCESSED_DATASET),
        Pair(PREPROCESSED_X_DATA_FILE_NAME, PATH_TO_PREPROCESSED_X_DATA),
        Pair(PREPROCESSED_Y_DATA_FILE_NAME, PATH_TO_PREPROCESSED_Y_DATA),
    )

    val deferredUploads = filesToUpload.map {
        async {
            val blobClientPreProcessedData = getBlobClientConnection(
                storageConnectionString = storageConnectionString,
                blobContainerName = PROCESSED_DATA_BLOB_CONTAINER_NAME,
                fileName = it.first
            )
            uploadFileToBlob(blobClient = blobClientPreProcessedData, filePath = it.second)
        }
    }
    deferredUploads.awaitAll()

    val prePreProcessedXData = async { readCSVAsKotlinDF(path = PATH_TO_PREPROCESSED_X_DATA) }.await()
    val prePreProcessedYData = async {readCSVAsKotlinDF(path = PATH_TO_PREPROCESSED_Y_DATA) }.await()

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

    // Store training results in Postgres DB
    connectToDB(dbURL = TRAINING_RESULT_DB_URL)
    createTable(table = TrainingResults)
    insertTrainingResults(algorithmName = cfg.train.algorithm, accuracy = accuracy)
}


/**
 * Comprises all preprocessing steps and the training/prediction for a Deep Learning Classifier.
 */
fun deepLearningTrainingPipeline(cfg: Config) = runBlocking {
    println("Starting the Deep Learning Classifier pipeline...")
    val storageConnectionString = System.getenv("STORAGE_CONNECTION_STRING")

    if (!File(PATH_TO_DATASET).exists()) {
        println("Downloading original dataset from Blob...")
        val blobClient = getBlobClientConnection(
            storageConnectionString = storageConnectionString,
            blobContainerName = RAW_DATA_BLOB_CONTAINER_NAME,
            fileName = RAW_FILE_NAME,
        )
        downloadFileFromBlob(blobClient = blobClient, filePath = PATH_TO_DATASET)
    }

    val data = readCSVAsKotlinDF(path = PATH_TO_DATASET)
    val (_, xData, yData) = dataPreProcessing(df = data)

    // Store Kotlin DF locally
    val kotlinDFsAndPaths = listOf(
        xData to PATH_TO_PREPROCESSED_X_DATA,
        yData.toDataFrame() to PATH_TO_PREPROCESSED_Y_DATA,
    )
    val preProcessedKotlinDFsToStore = kotlinDFsAndPaths.map { (df, path) ->
        async { storeKotlinDFAsCSVAsync(df, path) }
    }
    preProcessedKotlinDFsToStore.awaitAll()

    // Upload preprocessed data to Blob
    val filesToUpload = listOf(
        Pair(PREPROCESSED_X_DATA_FILE_NAME, PATH_TO_PREPROCESSED_X_DATA),
        Pair(PREPROCESSED_Y_DATA_FILE_NAME, PATH_TO_PREPROCESSED_Y_DATA),
    )
    val deferredUploads = filesToUpload.map {
        async {
            val blobClientPreProcessedData = getBlobClientConnection(
                storageConnectionString = storageConnectionString,
                blobContainerName = PROCESSED_DATA_BLOB_CONTAINER_NAME,
                fileName = it.first
            )
            uploadFileToBlob(blobClient = blobClientPreProcessedData, filePath = it.second)
        }
    }
    deferredUploads.awaitAll()

    val (train, test) = trainTestSplitForKotlinDL(xData = xData, yData = yData, trainSize = cfg.preProcessingDL.trainSize)

    val deepLearningClassifier = DeepLearningClassifier(cfg = cfg)
    val predictions = deepLearningClassifier.fitAndPredict(xData = train, yData = test)
    val accuracy = predictions.metrics[Metrics.ACCURACY]

    accuracy?.let { nonNullAccuracy ->
        println("Accuracy: ${round(value = nonNullAccuracy, places = 4)}")
    }

    // Store training results in Postgres DB
    connectToDB(dbURL = TRAINING_RESULT_DB_URL)
    createTable(table = TrainingResults)
    accuracy?.let { nonNullAccuracy ->
        insertTrainingResults(algorithmName = cfg.train.algorithm, accuracy = round(value = nonNullAccuracy, places = 4))
    }
}
