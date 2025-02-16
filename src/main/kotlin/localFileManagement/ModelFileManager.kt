package localFileManagement

import constants.PATH_TO_TRAINED_MODELS
import datetime.createTimeStamp
import logging.ProjectLogger.logger
import org.jetbrains.kotlinx.dl.api.core.SavingFormat
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import training.EnsembleClassifier
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.ObjectOutputStream


fun saveDLClassifierModel(model: Sequential) {
    val resultFolderName = createTimeStamp() + "_deepLearningClassifier/"
    val pathToResults = File(PATH_TO_TRAINED_MODELS + resultFolderName)
    model.save(
        modelDirectory = pathToResults,
        savingFormat = SavingFormat.TF_GRAPH_CUSTOM_VARIABLES,
        writingMode = WritingMode.OVERRIDE,
    )
}


fun storeSmileClassifierModel(model: EnsembleClassifier, path: File) {
    try {
        FileOutputStream(path).use { fileOut ->
            ObjectOutputStream(fileOut).use { out ->
                out.writeObject(model)
            }
        }
        logger.info("Serialized data is saved in decision_tree_model.ser")
    } catch (i: IOException) {
        i.printStackTrace()
    }
}
