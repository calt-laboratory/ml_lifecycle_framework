package training

import config.Config
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.EvaluationResult
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.impl.summary.logSummary
import smile.base.cart.SplitRule
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


abstract class EnsembleClassifier(val cfg: Config) {

    abstract fun fit(trainDF: smile.data.DataFrame)

    abstract fun predict(testDF: smile.data.DataFrame): IntArray
}


// TODO: Add SplitRule as yaml config param in DecisionTreeClassifier and RandomForestClassifier
class DecisionTreeClassifier(cfg: Config) : EnsembleClassifier(cfg) {

    private lateinit var model: DecisionTree

    override fun fit(trainDF: smile.data.DataFrame) {
        model = cart(
            formula = Formula.lhs("diagnosis"),
            data = trainDF,
            splitRule = SplitRule.GINI,
            maxDepth = cfg.train.decisionTree.maxDepth,
            maxNodes = cfg.train.decisionTree.maxNodes,
            nodeSize = cfg.train.decisionTree.nodeSize,
        )
    }

    override fun predict(testDF: smile.data.DataFrame): IntArray {
        val predictions = model.predict(testDF)
        return predictions
    }
}


class RandomForestClassifier(cfg: Config) : EnsembleClassifier(cfg) {

    private lateinit var model: RandomForest

    override fun fit(trainDF: smile.data.DataFrame) {
        model = randomForest(
            formula = Formula.lhs("diagnosis"),
            data = trainDF,
            ntrees = cfg.train.randomForest.nTrees,
            mtry = cfg.train.randomForest.mtry,
            splitRule = SplitRule.GINI,
            maxDepth = cfg.train.randomForest.maxDepth,
            maxNodes = cfg.train.randomForest.maxNodes,
            nodeSize = cfg.train.randomForest.nodeSize,
            subsample = cfg.train.randomForest.subsample,
            classWeight = cfg.train.randomForest.classWeight,
            seeds = cfg.train.randomForest.seeds,
            )
    }

    override fun predict(testDF: smile.data.DataFrame) : IntArray {
        val predictions = model.predict(testDF)
        return predictions
    }
}


class AdaBoostClassifier(cfg: Config) : EnsembleClassifier(cfg) {

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

    override fun predict(testDF: smile.data.DataFrame) : IntArray {
        val model = requireNotNull(model) { "Model is not fitted yet." }
        val predictions = model.predict(testDF)
        return predictions
    }
}


class GradientBoostingClassifier(cfg: Config) : EnsembleClassifier(cfg) {

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

    override fun predict(testDF: smile.data.DataFrame) : IntArray {
        val model = requireNotNull(model) { "Model is not fitted yet." }
        val predictions = model.predict(testDF)
        return predictions
    }
}


class LogisticRegressionModel(val cfg: Config) {

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

    fun predict(xTest: Array<DoubleArray>) : IntArray {
        val model = requireNotNull(model) { "Model is not fitted yet." }
        val predictions = model.predict(xTest)
        return predictions
    }
}


class DeepLearningClassifier() {

    val SEED = 12L
    val TEST_BATCH_SIZE = 5
    val EPOCHS = 20
    val TRAINING_BATCH_SIZE = 5

    private var model = Sequential.of(
        Input(30),
        Dense(outputSize = 300, activation = Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
        Dense(outputSize = 2, activation =  Activations.Linear, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    )

    fun fitAndPredict(xData: OnHeapDataset, yData: OnHeapDataset) : EvaluationResult {
        model.use {
            it.compile(optimizer = SGD(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Metrics.ACCURACY)
            it.logSummary()
            it.fit(dataset = xData, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)
            return model.evaluate(dataset = yData, batchSize = TEST_BATCH_SIZE)
        }
    }
}