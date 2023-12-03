package datetime

import kotlinx.datetime.Clock
import kotlinx.datetime.TimeZone
import kotlinx.datetime.toLocalDateTime


fun createTimeStamp(): String {
    val currentDatetime = Clock.System.now().toLocalDateTime(TimeZone.of("Europe/Berlin")).toString()
    return currentDatetime.substringBeforeLast(".")
}
