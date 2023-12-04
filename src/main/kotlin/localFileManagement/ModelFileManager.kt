package localFileManagement

import constants.PATH_TO_TRAINED_MODELS
import datetime.createTimeStamp
import org.jetbrains.kotlinx.dl.api.core.SavingFormat
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import java.io.File

fun saveDLClassifierModel(model: Sequential) {
    val resultFolderName = createTimeStamp() + "_deepLearningClassifier/"
    val pathToResults = File(PATH_TO_TRAINED_MODELS + resultFolderName)
    model.save(
        modelDirectory = pathToResults,
        savingFormat = SavingFormat.TF_GRAPH_CUSTOM_VARIABLES,
        writingMode = WritingMode.OVERRIDE,
    )
}
