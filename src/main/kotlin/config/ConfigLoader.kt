package config

import com.charleskorn.kaml.Yaml
import kotlinx.serialization.Serializable
import smile.base.cart.SplitRule
import java.nio.file.Path
import java.util.stream.LongStream

@Serializable
data class TrainConfig(
    val algorithm: String,
    val decisionTree: DecisionTreeConfig,
    val randomForest: RandomForestConfig,
    val adaBoost: AdaBoostConfig,
    val gradientBoosting: GradientBoostingConfig,
    val logisticRegression: LogisticRegressionConfig,
    val deepLearningClassifier: DeepLearningClassifierConfig,
)

@Serializable
data class DecisionTreeConfig(
    val splitRule: SplitRule,
    val maxDepth: Int,
    val maxNodes: Int,
    val nodeSize: Int,
)

@Serializable
data class RandomForestConfig(
    val nTrees: Int,
    val mtry: Int,
    val splitRule: SplitRule,
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
data class GradientBoostingConfig(
    val nTrees: Int,
    val maxDepth: Int,
    val maxNodes: Int,
    val nodeSize: Int,
    val shrinkage: Double,
    val subsample: Double,
)

@Serializable
data class LogisticRegressionConfig(
    val lambda: Double,
    val tol: Double,
    val maxIter: Int,
)

@Serializable
data class DeepLearningClassifierConfig(
    val kernelInitializerSeed: Long,
    val epochs: Int,
    val trainBatchSize: Int,
    val testBatchSize: Int,
)

@Serializable
data class PreProcessingConfig(
    val seed: Int,
    val testSize: Double,
)

@Serializable
data class PreProcessingDeepLearningConfig(
    val trainSize: Double,
)

@Serializable
data class Config(
    val train: TrainConfig,
    val preProcessing: PreProcessingConfig,
    val preProcessingDL: PreProcessingDeepLearningConfig,
)

fun readYamlConfig(filePath: String): Config {
    return Yaml.default.decodeFromString(
        Config.serializer(),
        Path.of(filePath).toFile().readText()
    )
}
