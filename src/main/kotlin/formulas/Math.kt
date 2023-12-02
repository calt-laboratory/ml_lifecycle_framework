package formulas


/**
 * Rounds a double value to a specified number of decimal places.
 * @param value The value to round
 * @param places The number of decimal places
 * @return The rounded value
 */
fun round(value: Double, places: Int) : Double {
    return "%.${places}f".format(java.util.Locale.US, value).toDouble()
}
