package training

/**
 * Calculates the accuracy of a model.
 * @param yTrue: True values of y-test set
 * @param yPred: Predicted values of y-test set
 * @return accuracy: Accuracy of a model
 */
fun calculateAccuracy(yTrue: IntArray, yPred: IntArray) : Double {
    require(value = yTrue.size == yPred.size) { "Arrays must have the same size" }
    val accuracy = yTrue.zip(yPred).count { (a, b) -> a == b }.toDouble() / yTrue.size
    return accuracy
}
