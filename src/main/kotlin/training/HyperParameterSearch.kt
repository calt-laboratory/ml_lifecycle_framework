package training


fun <T> cartesianProduct(parameters: Map<String, List<T>>): Set<Map<String, T>> {
    val keys = parameters.keys.toList()
    val values = parameters.values.toList()

    val combinations = mutableListOf<Map<String, T>>()
    generateCombinations(keys, values, mutableMapOf(), combinations)

    return combinations.toSet()
}


fun <T> generateCombinations(
    keys: List<String>,
    values: List<List<T>>,
    current: MutableMap<String, T>,
    combinations: MutableList<Map<String, T>>
) {
    if (keys.isEmpty()) {
        combinations.add(current.toMap())
        return
    }

    val currentKey = keys.first()
    val currentValues = values.first()

    for (value in currentValues) {
        current[currentKey] = value
        generateCombinations(keys.drop(1), values.drop(1), current, combinations)
    }
}
