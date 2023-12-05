package training

import config.Config
import config.DecisionTreeConfig
import config.RandomForestConfig
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import smile.classification.AdaBoost
import smile.classification.DecisionTree
import smile.classification.GradientTreeBoost
import smile.classification.LogisticRegression
import smile.classification.RandomForest
import smile.classification.adaboost
import smile.classification.cart
import smile.classification.logit
import smile.classification.randomForest
import smile.data.formula.Formula


abstract class EnsembleClassifier {

    abstract fun fit(trainDF: smile.data.DataFrame)

    abstract fun predict(testDF: smile.data.DataFrame): IntArray
}


class DecisionTreeClassifier(
    private val decisionTreeConfig: DecisionTreeConfig,
) : EnsembleClassifier() {

    private lateinit var model: DecisionTree

    override fun fit(trainDF: smile.data.DataFrame) {
        model = cart(
            formula = Formula.lhs("diagnosis"),
            data = trainDF,
            splitRule = decisionTreeConfig.splitRule,
            maxDepth = decisionTreeConfig.maxDepth,
            maxNodes = decisionTreeConfig.maxNodes,
            nodeSize = decisionTreeConfig.nodeSize,
        )
    }

    override fun predict(testDF: smile.data.DataFrame): IntArray {
        return model.predict(testDF)
    }
}


class RandomForestClassifier(private val randomForestConfig: RandomForestConfig) : EnsembleClassifier() {

    private lateinit var model: RandomForest

    override fun fit(trainDF: smile.data.DataFrame) {
        model = randomForest(
            formula = Formula.lhs("diagnosis"),
            data = trainDF,
            ntrees = randomForestConfig.nTrees,
            mtry = randomForestConfig.mtry,
            splitRule = randomForestConfig.splitRule,
            maxDepth = randomForestConfig.maxDepth,
            maxNodes = randomForestConfig.maxNodes,
            nodeSize = randomForestConfig.nodeSize,
            subsample = randomForestConfig.subsample,
            classWeight = randomForestConfig.classWeight,
            seeds = randomForestConfig.seeds,
        )
    }

    override fun predict(testDF: smile.data.DataFrame): IntArray {
        return model.predict(testDF)
    }
}


class AdaBoostClassifier(private val cfg: Config) : EnsembleClassifier() {

    private var model: AdaBoost? = null

    override fun fit(trainDF: smile.data.DataFrame) {
        model = adaboost(
            formula = Formula.lhs("diagnosis"),
            data = trainDF,
            ntrees = cfg.train.adaBoost.nTrees,
            maxDepth = cfg.train.adaBoost.maxDepth,
            maxNodes = cfg.train.adaBoost.maxNodes,
            nodeSize = cfg.train.adaBoost.nodeSize,
        )
    }

    override fun predict(testDF: smile.data.DataFrame): IntArray {
        val model = requireNotNull(model) { "Model is not fitted yet." }
        return model.predict(testDF)
    }
}


class GradientBoostingClassifier(private val cfg: Config) : EnsembleClassifier() {

    private var model: GradientTreeBoost? = null

    override fun fit(trainDF: smile.data.DataFrame) {
        model = smile.classification.gbm(
            formula = Formula.lhs("diagnosis"),
            data = trainDF,
            ntrees = cfg.train.gradientBoosting.nTrees,
            maxDepth = cfg.train.gradientBoosting.maxDepth,
            maxNodes = cfg.train.gradientBoosting.maxNodes,
            nodeSize = cfg.train.gradientBoosting.nodeSize,
            shrinkage = cfg.train.gradientBoosting.shrinkage,
            subsample = cfg.train.gradientBoosting.subsample,
        )
    }

    override fun predict(testDF: smile.data.DataFrame): IntArray {
        val model = requireNotNull(model) { "Model is not fitted yet." }
        return model.predict(testDF)
    }
}


class LogisticRegressionModel(private val cfg: Config) {

    private var model: LogisticRegression? = null

    fun fit(xTrain: Array<DoubleArray>, yTrain: IntArray) {
        model = logit(
            x = xTrain,
            y = yTrain,
            lambda = cfg.train.logisticRegression.lambda,
            tol = cfg.train.logisticRegression.tol,
            maxIter = cfg.train.logisticRegression.maxIter,
        )
    }

    fun predict(xTest: Array<DoubleArray>): IntArray {
        val model = requireNotNull(model) { "Model is not fitted yet." }
        return model.predict(xTest)
    }
}


class DeepLearningClassifier(private val cfg: Config) {

    private var model = Sequential.of(
        Input(30),
        Dense(
            outputSize = 300,
            activation = Activations.Relu,
            kernelInitializer = HeNormal(cfg.train.deepLearningClassifier.kernelInitializerSeed),
            biasInitializer = Zeros(),
        ),
        Dense(
            outputSize = 2,
            activation = Activations.Linear,
            kernelInitializer = HeNormal(cfg.train.deepLearningClassifier.kernelInitializerSeed),
            biasInitializer = Zeros(),
        ),
    )

    fun fitAndPredict(trainData: OnHeapDataset, testData: OnHeapDataset): Pair<Sequential, Double?> {

        model.compile(optimizer = SGD(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Metrics.ACCURACY)
        model.fit(
            dataset = trainData,
            epochs = cfg.train.deepLearningClassifier.epochs,
            batchSize = cfg.train.deepLearningClassifier.trainBatchSize,
        )
        val accuracy = model.evaluate(
            dataset = testData,
            batchSize = cfg.train.deepLearningClassifier.testBatchSize,
        ).metrics[Metrics.ACCURACY]
        return Pair(model, accuracy)
    }
}
