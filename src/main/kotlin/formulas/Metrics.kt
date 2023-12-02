package formulas


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

fun recall(yTrue: IntArray, yPred: IntArray) : Double {
    require(value = yTrue.size == yPred.size) { "Arrays must have the same size" }
    val truePositives = yTrue.zip(yPred).count { (a, b) -> a == b && a == 1 }
    val falseNegatives = yTrue.zip(yPred).count { (a, b) -> a != b && a == 1 }
    return round(value = truePositives.toDouble() / (truePositives + falseNegatives), places = 4)
}

fun f1Score(yTrue: IntArray, yPred: IntArray) : Double {
    require(value = yTrue.size == yPred.size) { "Arrays must have the same size" }
    val precision = precision(yTrue, yPred)
    val recall = recall(yTrue, yPred)
    return round(value = 2 * (precision * recall) / (precision + recall), places = 4)
}
