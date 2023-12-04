package datetime

import kotlinx.datetime.Clock
import kotlinx.datetime.TimeZone
import kotlinx.datetime.toLocalDateTime


/**
 * Creates a timestamp in the format "yyyy-MM-ddTHH_mm_ss".
 */
fun createTimeStamp(): String {
    val currentDatetime = Clock.System.now()
        .toLocalDateTime(TimeZone.of("Europe/Berlin"))
        .toString()
    return currentDatetime
        .substringBeforeLast(".")  // remove nanoseconds
        .replace(oldValue = ":", newValue = "_")
}
