package postgres

import org.jetbrains.exposed.dao.id.IntIdTable
import org.jetbrains.exposed.sql.Database
import org.jetbrains.exposed.sql.SchemaUtils
import org.jetbrains.exposed.sql.insertAndGetId
import org.jetbrains.exposed.sql.kotlin.datetime.CurrentDateTime
import org.jetbrains.exposed.sql.kotlin.datetime.datetime
import org.jetbrains.exposed.sql.transactions.transaction

fun connectToDB () {
    Database.connect(
        url = "jdbc:postgresql://localhost:5432/training_results",
        driver = "org.postgresql.Driver",
        user = "postgres",
        password = System.getenv("POSTGRES_PW")
    )
}

object TrainingResults : IntIdTable() {
    val date = datetime(name = "date").defaultExpression(CurrentDateTime)
    val algorithmName = varchar(name = "algorithm_name", length = 100)
    val accuracy = double(name = "accuracy")
}

fun createTable () {
    transaction {
        SchemaUtils.create(TrainingResults)
    }
}

fun insertTrainingResults (algorithmName: String, accuracy: Double) {
    transaction {
        TrainingResults.insertAndGetId {
            it[TrainingResults.algorithmName] = algorithmName
            it[TrainingResults.accuracy] = accuracy
        }
    }
}
