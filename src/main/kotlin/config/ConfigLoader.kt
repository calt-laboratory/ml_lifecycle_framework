package config

import com.charleskorn.kaml.Yaml
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import smile.base.cart.SplitRule
import java.util.stream.LongStream

enum class Algorithm {
    @SerialName("decisionTree")
    DECISION_TREE,

    @SerialName("randomForest")
    RANDOM_FOREST,

    @SerialName("adaBoost")
    ADA_BOOST,

    @SerialName("gradientBoosting")
    GRADIENT_BOOSTING,

    @SerialName("logisticRegression")
    LOGISTIC_REGRESSION,

    @SerialName("deepLearningClassifier")
    DEEP_LEARNING_CLASSIFIER,
}

val ensembleAlgorithms = listOf(
    Algorithm.DECISION_TREE,
    Algorithm.RANDOM_FOREST,
    Algorithm.ADA_BOOST,
    Algorithm.GRADIENT_BOOSTING,
)

@Serializable
data class TrainConfig(
    val algorithms: List<Algorithm>,
    val decisionTree: DecisionTreeConfig,
    val decisionTreeGridSearch: DecisionTreeGridSearchConfig,
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
data class DecisionTreeGridSearchConfig(
    val splitRule: List<SplitRule>,
    val maxDepth: List<Int>,
    val maxNodes: List<Int>,
    val nodeSize: List<Int>,
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
data class CloudProvider(
    val aws: Boolean,
    val azure: Boolean)

@Serializable
data class Config(
    val train: TrainConfig,
    val preProcessing: PreProcessingConfig,
    val preProcessingDL: PreProcessingDeepLearningConfig,
    val cloudProvider: CloudProvider,
) {
    companion object {
        private const val PATH_TO_YAML_CONFIG = "config.yml"

        fun fromYaml(path: String = PATH_TO_YAML_CONFIG): Config {
            val contents = object {}.javaClass.classLoader.getResource(path)?.readText()
                ?: throw Exception("Could not read config file $path")
            return Yaml.default.decodeFromString(contents)
        }
    }
}
