package training

import util.round

/**
 * Calculates the rounded accuracy of a model.
 * @param yTrue: True values of y-test set
 * @param yPred: Predicted values of y-test set
 * @return accuracy: Accuracy of a model
 */
fun calculateAccuracy(yTrue: IntArray, yPred: IntArray) : Double {
    require(value = yTrue.size == yPred.size) { "Arrays must have the same size" }
    val accuracy = yTrue.zip(yPred).count { (a, b) -> a == b }.toDouble() / yTrue.size
    return round(value = accuracy, places = 4)
}


fun precision(yTrue: IntArray, yPred: IntArray) : Double {
    require(value = yTrue.size == yPred.size) { "Arrays must have the same size" }
    val truePositives = yTrue.zip(yPred).count { (a, b) -> a == b && a == 1 }
    val falsePositives = yTrue.zip(yPred).count { (a, b) -> a != b && a == 0 }
    return round(value = truePositives.toDouble() / (truePositives + falsePositives), places = 4)
}
