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
    val accuracy = double(name = "accuracy").nullable()
    val precision = double(name = "precision").nullable()
    val recall = double(name = "recall").nullable()
    val f1Score = double(name = "f1_score").nullable()
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
        password = "password"
        // password = System.getenv("POSTGRES_PW"),
    )
}


/**
 * Creates IntIdTable in a database.
 * @param table The table to create
 */
fun createTable (table: IntIdTable) {
    transaction {
        SchemaUtils.create(table)
    }
}


fun updateTableStructure (table: IntIdTable) {
    transaction {
        SchemaUtils.createMissingTablesAndColumns(table)
    }
}


/**
 * Inserts training results into a database.
 * @param algorithmName The name of the algorithm
 * @param metrics All model prediction classification metrics
 */
fun insertTrainingResults (algorithmName: String, metrics: Map<String, Double?>) {
    transaction {
        TrainingResults.insertAndGetId {
            it[TrainingResults.algorithmName] = algorithmName
            it[accuracy] = metrics.getValue("accuracy")
            it[precision] = metrics.getValue("precision")
            it[recall] = metrics.getValue("recall")
            it[f1Score] = metrics.getValue("f1Score")
        }
    }
}
