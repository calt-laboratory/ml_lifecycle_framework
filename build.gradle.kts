plugins {
    kotlin("jvm") version "1.9.0"
    kotlin("plugin.serialization") version "1.9.0"
    application
    id("org.jlleitschuh.gradle.ktlint") version "11.5.1"
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    jvmToolchain(17)
}

application {
    mainClass.set("MainKt")
}

dependencies {
    implementation(kotlin("reflect"))
    implementation("org.jetbrains.kotlinx:dataframe:0.11.1")
    implementation("com.github.haifengl:smile-kotlin:3.0.1")
    implementation("com.azure:azure-storage-blob:12.18.0")
    implementation("org.yaml:snakeyaml:2.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.0")
    implementation("com.charleskorn.kaml:kaml:0.55.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    // implementation("io.github.oshai:kotlin-logging-jvm:5.1.0")
    implementation("ch.qos.logback:logback-classic:1.2.9")
    implementation("org.jetbrains.kotlinx:kotlin-deeplearning-tensorflow:[0.5.2]")
    implementation("org.jetbrains.kotlinx:kotlin-deeplearning-onnx:[0.5.2]")
    implementation("org.jetbrains.kotlinx:kotlin-deeplearning-visualization:[0.5.2]")
    implementation("org.jetbrains.exposed:exposed-core:0.44.1")
    implementation("org.jetbrains.exposed:exposed-dao:0.44.1")
    implementation("org.jetbrains.exposed:exposed-kotlin-datetime:0.44.1")
    implementation("org.postgresql:postgresql:42.2.27")

    testImplementation(kotlin("test"))
}
