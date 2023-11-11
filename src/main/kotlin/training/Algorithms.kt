package training

import smile.base.cart.SplitRule
import smile.classification.DecisionTree
import smile.classification.LogisticRegression
import smile.classification.RandomForest
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

abstract class EnsembleClassifier {

    abstract fun fit(trainDF: smile.data.DataFrame)

    abstract fun predict(testDF: smile.data.DataFrame): IntArray
}

class DecisionTreeClassifier : EnsembleClassifier() {

    private lateinit var model: DecisionTree

    override fun fit(trainDF: smile.data.DataFrame) {
        model = cart(
            formula = Formula.lhs("diagnosis"),
            data = trainDF,
            splitRule = SplitRule.GINI,
            maxDepth = 20,
            maxNodes = 0,
            nodeSize = 5,
        )
    }
    override fun predict(testDF: smile.data.DataFrame): IntArray {
        val predictions = model.predict(testDF)
        return predictions
    }
}

class RandomForestClassifier : EnsembleClassifier() {

    private lateinit var model: RandomForest
    override fun fit(trainDF: smile.data.DataFrame) {
        model = randomForest(
            formula = Formula.lhs("diagnosis"),
            data = trainDF,
            ntrees = 500,
            splitRule = SplitRule.GINI,
            )
    }

    override fun predict(testDF: smile.data.DataFrame) : IntArray {
        val predictions = model.predict(testDF)
        return predictions
    }
}