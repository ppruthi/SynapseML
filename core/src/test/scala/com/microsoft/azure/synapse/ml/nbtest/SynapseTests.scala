// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.nbtest

import com.microsoft.azure.synapse.ml.core.test.base.TestBase
import com.microsoft.azure.synapse.ml.nbtest.SynapseUtilities.{createSparkPools, deleteSparkPools, exec}

import java.io.File
import java.nio.file.{Path, Paths}
import java.util.concurrent.TimeUnit
import scala.concurrent.Await
import scala.concurrent.duration.Duration
import scala.language.existentials
import scala.sys.process.Process

/** Tests to validate fuzzing of modules. */
class SynapseTests extends TestBase {
  val os: String = sys.props("os.name").toLowerCase
  os match {
    case x if x contains "windows" =>
      exec("conda activate synapseml " +
        "&& jupyter nbconvert --to script .\\notebooks\\features\\**\\*.ipynb")
    case _ =>
      Process(s"conda init bash; conda activate synapseml; " +
        "jupyter nbconvert --to script ./notebooks/features/**/*.ipynb")
  }

  SynapseUtilities.listPythonFiles().map(f => {
    val newPath = f
      .replace(" ", "")
      .replace("-", "")
    new File(f).renameTo(new File(newPath))
  })

  val workspaceName = "mmlsparkppe"
  val resourceGroupName = "marhamil-mmlspark"
  val subscriptionId = "e342c2c0-f844-4b18-9208-52c8c234c30e"
  val expectedPoolCount = 3

  println(s"SynapseTests E2E Test Suite starting. Creating $expectedPoolCount Spark Pools...")
  var (sparkPools, actualPoolCount) = {
    createSparkPools(expectedPoolCount, subscriptionId, resourceGroupName, workspaceName)
  }
  sparkPools = sparkPools.slice(0, actualPoolCount)
  println("Spark Pools Created: [" + sparkPools.mkString(", ") + "]")

  if (actualPoolCount == 0) {
    println("Failed to create any Spark Pool for running the tests. Exiting...")
    System.exit(1)
  }

  if (actualPoolCount != expectedPoolCount) {
    println(s"WARNING: Expecting $expectedPoolCount pools; only $actualPoolCount created." +
      "This may impact test runtime performance")
  }

  SynapseUtilities.listPythonJobFiles()
    .filterNot(_.contains(" "))
    .filterNot(_.contains("-"))
    .foreach(file => {
      val poolName = SynapseUtilities.monitorPool(workspaceName, sparkPools)
      val livyUrl = "https://" +
        workspaceName +
        ".dev.azuresynapse-dogfood.net/livyApi/versions/2019-11-01-preview/sparkPools/" +
        poolName +
        "/batches"
      val (livyBatch: LivyBatch, jobName: String) = SynapseUtilities.uploadAndSubmitNotebook(livyUrl, file)
      val path: Path = Paths.get(file)
      val fileName: String = path.getFileName.toString

      println(s"submitted livy job: ${livyBatch.id} for file $fileName to sparkPool: $poolName")

      val livyBatchJob: LivyBatchJob = LivyBatchJob(livyBatch, poolName, livyUrl)

      test(fileName) {
        try {
          val result = Await.ready(
            livyBatchJob.monitor(),
            Duration(SynapseUtilities.TimeoutInMillis.toLong, TimeUnit.MILLISECONDS)).value.get

          val jobUrl = "https://web-staging.azuresynapse.net/en-us/monitoring/sparkapplication/" +
            jobName +
            s"?workspace=%2Fsubscriptions%2F${subscriptionId}" +
            s"%2FresourceGroups%2F${resourceGroupName}" +
            s"%2Fproviders%2FMicrosoft.Synapse%2Fworkspaces%2F${workspaceName}" +
            s"&sparkPoolName=${poolName}&livyId=${livyBatch.id}"
          assert(result.isSuccess, s"Job failed see ${jobUrl} for details")
        } catch {
          case t: Throwable =>
            println(s"Cancelling job ${livyBatchJob.livyBatch.id} for file $fileName")
            SynapseUtilities.cancelRun(livyBatchJob.livyUrl, livyBatchJob.livyBatch.id)

            throw t
        }
      }
    })

  protected override def afterAll(): Unit = {
    println("SynapseTests E2E Test Suite finished. Deleting Spark Pools...")
    val deletionSuccess = deleteSparkPools(sparkPools, subscriptionId, resourceGroupName, workspaceName)
    println(s"Deletion: [$deletionSuccess]")
    super.afterAll()
  }
}
