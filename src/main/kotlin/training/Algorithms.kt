package training

import config.Config
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


// TODO: Implement Deep Learning Net using KotlinDL
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
            ntrees = 500,
            maxDepth = 20,
            maxNodes = 6,
            nodeSize = 1,
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
