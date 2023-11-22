package postgres

import org.jetbrains.exposed.dao.id.IntIdTable
import org.jetbrains.exposed.sql.Database
import org.jetbrains.exposed.sql.SchemaUtils
import org.jetbrains.exposed.sql.insertAndGetId
import org.jetbrains.exposed.sql.kotlin.datetime.CurrentDateTime
import org.jetbrains.exposed.sql.kotlin.datetime.datetime
import org.jetbrains.exposed.sql.transactions.transaction

object TrainingResults : IntIdTable() {
    val date = datetime(name = "date").defaultExpression(CurrentDateTime)
    val algorithmName = varchar(name = "algorithm_name", length = 100)
    val accuracy = double(name = "accuracy")
}

/**
 * Connects to a PostgreSQL database.
 * @param dbURL URL of the database consumed from a constant
 */
fun connectToDB (dbURL: String) {
    Database.connect(
        url = dbURL,
        driver = "org.postgresql.Driver",
        user = "postgres",
        password = System.getenv("POSTGRES_PW")
    )
}

/**
 * Creates InIdTable in a database.
 * @param table The table to create
 */
fun createTable (table: IntIdTable) {
    transaction {
        SchemaUtils.create(table)
    }
}

/**
 * Inserts training results into a database.
 * @param algorithmName The name of the algorithm
 * @param accuracy The accuracy of the model prediction
 */
fun insertTrainingResults (algorithmName: String, accuracy: Double) {
    transaction {
        TrainingResults.insertAndGetId {
            it[TrainingResults.algorithmName] = algorithmName
            it[TrainingResults.accuracy] = accuracy
        }
    }
}
