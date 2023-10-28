package training

import smile.classification.LogisticRegression

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