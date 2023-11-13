package training

import config.Config
import smile.base.cart.SplitRule
import smile.classification.AdaBoost
import smile.classification.DecisionTree
import smile.classification.LogisticRegression
import smile.classification.RandomForest
import smile.classification.adaboost
import smile.classification.cart
import smile.classification.randomForest
import smile.data.formula.Formula

abstract class ClassificationAlgorithm {

    abstract fun fit(xTrain: Array<DoubleArray>, yTrain: IntArray)

    abstract fun predictModel(xTest: Array<DoubleArray>): IntArray
}


class LogisticRegressionModel : ClassificationAlgorithm() {

    private var model: LogisticRegression? = null

    override fun fit(xTrain: Array<DoubleArray>, yTrain: IntArray) {
        model = LogisticRegression.fit(xTrain, yTrain)
    }

    override fun predictModel(xTest: Array<DoubleArray>) : IntArray {
        val model = requireNotNull(model) { "Model is not fitted yet." }
        val predictions = model.predict(xTest)
        return predictions
    }
}

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
