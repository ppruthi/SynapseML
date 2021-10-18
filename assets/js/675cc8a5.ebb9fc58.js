(self.webpackChunksynapseml=self.webpackChunksynapseml||[]).push([[2859],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return c},kt:function(){return m}});var a=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function s(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function r(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?s(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):s(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,a,i=function(e,t){if(null==e)return{};var n,a,i={},s=Object.keys(e);for(a=0;a<s.length;a++)n=s[a],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(e);for(a=0;a<s.length;a++)n=s[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var l=a.createContext({}),p=function(e){var t=a.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):r(r({},t),e)),n},c=function(e){var t=p(e.components);return a.createElement(l.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},f=a.forwardRef((function(e,t){var n=e.components,i=e.mdxType,s=e.originalType,l=e.parentName,c=o(e,["components","mdxType","originalType","parentName"]),f=p(n),m=i,d=f["".concat(l,".").concat(m)]||f[m]||u[m]||s;return n?a.createElement(d,r(r({ref:t},c),{},{components:n})):a.createElement(d,r({ref:t},c))}));function m(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var s=n.length,r=new Array(s);r[0]=f;var o={};for(var l in t)hasOwnProperty.call(t,l)&&(o[l]=t[l]);o.originalType=e,o.mdxType="string"==typeof e?e:i,r[1]=o;for(var p=2;p<s;p++)r[p]=n[p];return a.createElement.apply(null,r)}return a.createElement.apply(null,n)}f.displayName="MDXCreateElement"},206:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return o},contentTitle:function(){return l},metadata:function(){return p},toc:function(){return c},default:function(){return f}});var a=n(2122),i=n(9756),s=(n(7294),n(3905)),r=["components"],o={title:"Classification - Adult Census with Vowpal Wabbit",hide_title:!0,status:"stable",name:"Classification - Adult Census with Vowpal Wabbit"},l="Classification - Adult Census using Vowpal Wabbit in SynapseML",p={unversionedId:"examples/classification/Classification - Adult Census with Vowpal Wabbit",id:"version-0.9.1/examples/classification/Classification - Adult Census with Vowpal Wabbit",isDocsHomePage:!1,title:"Classification - Adult Census with Vowpal Wabbit",description:"In this example, we predict incomes from the Adult Census dataset using Vowpal Wabbit (VW) classifier in SynapseML.",source:"@site/versioned_docs/version-0.9.1/examples/classification/Classification - Adult Census with Vowpal Wabbit.md",sourceDirName:"examples/classification",slug:"/examples/classification/Classification - Adult Census with Vowpal Wabbit",permalink:"/docs/examples/classification/Classification - Adult Census with Vowpal Wabbit",version:"0.9.1",frontMatter:{title:"Classification - Adult Census with Vowpal Wabbit",hide_title:!0,status:"stable",name:"Classification - Adult Census with Vowpal Wabbit"},sidebar:"version-0.9.1/docs",previous:{title:"OpenCV - Pipeline Image Transformations",permalink:"/docs/examples/OpenCV - Pipeline Image Transformations"},next:{title:"Classification - Adult Census",permalink:"/docs/examples/classification/Classification - Adult Census"}},c=[],u={toc:c};function f(e){var t=e.components,n=(0,i.Z)(e,r);return(0,s.kt)("wrapper",(0,a.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,s.kt)("h1",{id:"classification---adult-census-using-vowpal-wabbit-in-synapseml"},"Classification - Adult Census using Vowpal Wabbit in SynapseML"),(0,s.kt)("p",null,"In this example, we predict incomes from the ",(0,s.kt)("em",{parentName:"p"},"Adult Census")," dataset using Vowpal Wabbit (VW) classifier in SynapseML.\nFirst, we read the data and split it into train and test sets as in this ",(0,s.kt)("a",{parentName:"p",href:"https://github.com/Microsoft/SynapseML/blob/master/notebooks/Classification%20-%20Adult%20Census.ipynb"},"example"),"."),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},'import os\nif os.environ.get("AZURE_SERVICE", None) == "Microsoft.ProjectArcadia":\n    from pyspark.sql import SparkSession\n    spark = SparkSession.builder.getOrCreate()\n')),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},'data = spark.read.parquet("wasbs://publicwasb@mmlspark.blob.core.windows.net/AdultCensusIncome.parquet")\ndata = data.select(["education", "marital-status", "hours-per-week", "income"])\ntrain, test = data.randomSplit([0.75, 0.25], seed=123)\ntrain.limit(10).toPandas()\n')),(0,s.kt)("p",null,"Next, we define a pipeline that includes feature engineering and training of a VW classifier. We use a featurizer provided by VW that hashes the feature names.\nNote that VW expects classification labels being -1 or 1. Thus, the income category is mapped to this space before feeding training data into the pipeline."),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},'from pyspark.sql.functions import when, col\nfrom pyspark.ml import Pipeline\nfrom synapse.ml.vw import VowpalWabbitFeaturizer, VowpalWabbitClassifier\n\n# Define classification label\ntrain = train.withColumn("label", when(col("income").contains("<"), 0.0).otherwise(1.0)).repartition(1).cache()\nprint(train.count())\n\n# Specify featurizer\nvw_featurizer = VowpalWabbitFeaturizer(inputCols=["education", "marital-status", "hours-per-week"],\n                                       outputCol="features")\n\n# Define VW classification model\nargs = "--loss_function=logistic --quiet --holdout_off"\nvw_model = VowpalWabbitClassifier(featuresCol="features",\n                                  labelCol="label",\n                                  args=args,\n                                  numPasses=10)\n\n# Create a pipeline\nvw_pipeline = Pipeline(stages=[vw_featurizer, vw_model])\n')),(0,s.kt)("p",null,"Then, we are ready to train the model by fitting the pipeline with the training data."),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"# Train the model\nvw_trained = vw_pipeline.fit(train)\n")),(0,s.kt)("p",null,"After the model is trained, we apply it to predict the income of each sample in the test set."),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},'# Making predictions\ntest = test.withColumn("label", when(col("income").contains("<"), 0.0).otherwise(1.0))\nprediction = vw_trained.transform(test)\nprediction.limit(10).toPandas()\n')),(0,s.kt)("p",null,"Finally, we evaluate the model performance using ",(0,s.kt)("inlineCode",{parentName:"p"},"ComputeModelStatistics")," function which will compute confusion matrix, accuracy, precision, recall, and AUC by default for classificaiton models."),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},'from synapse.ml.train import ComputeModelStatistics\nmetrics = ComputeModelStatistics(evaluationMetric="classification", \n                                 labelCol="label", \n                                 scoredLabelsCol="prediction").transform(prediction)\nmetrics.toPandas()\n')))}f.isMDXComponent=!0}}]);