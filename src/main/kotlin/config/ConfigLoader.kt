package config

import com.charleskorn.kaml.Yaml
import kotlinx.serialization.Serializable
import java.nio.file.Path
import java.util.stream.LongStream

@Serializable
data class TrainConfig(
    val algorithm: String,
    val decisionTree: DecisionTreeConfig,
    val randomForest: RandomForestConfig,
    val adaBoost: AdaBoostConfig,
    val logisticRegression: LogisticRegressionConfig,
)

@Serializable
data class DecisionTreeConfig(
    val maxDepth: Int,
    val maxNodes: Int,
    val nodeSize: Int,
)

@Serializable
data class RandomForestConfig(
    val nTrees: Int,
    val mtry: Int,
    val maxDepth: Int,
    val maxNodes: Int,
    val nodeSize: Int,
    val subsample: Double,
    val classWeight: IntArray?,
    val seeds: LongStream?,
)

@Serializable
data class AdaBoostConfig(
    val nTrees: Int,
    val maxDepth: Int,
    val maxNodes: Int,
    val nodeSize: Int,
)

@Serializable
data class LogisticRegressionConfig(
    val lambda: Double,
    val tol: Double,
    val maxIter: Int,
)

@Serializable
data class PreProcessingConfig(
    val seed: Int,
    val testSize: Double,
)

@Serializable
data class Config(
    val train: TrainConfig,
    val preProcessing: PreProcessingConfig,
)

fun readYamlConfig(filePath: String): Config {
    return Yaml.default.decodeFromString(
        Config.serializer(),
        Path.of(filePath).toFile().readText()
    )
}
