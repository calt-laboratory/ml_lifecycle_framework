package training

/**
 * Calculates the accuracy of a model.
 * @param y_true: True values of y-test set
 * @param y_pred: Predicted values of y-test set
 * @return accuracy: Accuracy of a model
 */
fun calculateAccuracy(y_true: IntArray , y_pred: IntArray) : Double {
    require(value = y_true.size == y_pred.size) { "Arrays must have the same size" }
    val accuracy = y_true.zip(y_pred).count { (a, b) -> a == b }.toDouble() / y_true.size
    return accuracy
}
