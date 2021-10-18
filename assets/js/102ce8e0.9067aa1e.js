(self.webpackChunksynapseml=self.webpackChunksynapseml||[]).push([[4065],{3905:function(e,n,t){"use strict";t.d(n,{Zo:function(){return m},kt:function(){return d}});var r=t(7294);function o(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function a(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function i(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?a(Object(t),!0).forEach((function(n){o(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):a(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function s(e,n){if(null==e)return{};var t,r,o=function(e,n){if(null==e)return{};var t,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||(o[t]=e[t]);return o}(e,n);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(o[t]=e[t])}return o}var l=r.createContext({}),p=function(e){var n=r.useContext(l),t=n;return e&&(t="function"==typeof e?e(n):i(i({},n),e)),t},m=function(e){var n=p(e.components);return r.createElement(l.Provider,{value:n},e.children)},c={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},u=r.forwardRef((function(e,n){var t=e.components,o=e.mdxType,a=e.originalType,l=e.parentName,m=s(e,["components","mdxType","originalType","parentName"]),u=p(t),d=o,f=u["".concat(l,".").concat(d)]||u[d]||c[d]||a;return t?r.createElement(f,i(i({ref:n},m),{},{components:t})):r.createElement(f,i({ref:n},m))}));function d(e,n){var t=arguments,o=n&&n.mdxType;if("string"==typeof e||o){var a=t.length,i=new Array(a);i[0]=u;var s={};for(var l in n)hasOwnProperty.call(n,l)&&(s[l]=n[l]);s.originalType=e,s.mdxType="string"==typeof e?e:o,i[1]=s;for(var p=2;p<a;p++)i[p]=t[p];return r.createElement.apply(null,i)}return r.createElement.apply(null,t)}u.displayName="MDXCreateElement"},595:function(e,n,t){"use strict";t.r(n),t.d(n,{frontMatter:function(){return s},contentTitle:function(){return l},metadata:function(){return p},toc:function(){return m},default:function(){return u}});var r=t(2122),o=t(9756),a=(t(7294),t(3905)),i=["components"],s={title:"ModelInterpretability - Snow Leopard Detection",hide_title:!0,status:"stable",url_path:"features/model_interpretability/ModelInterpretability - Snow Leopard Detection",name:"ModelInterpretability - Snow Leopard Detection"},l=void 0,p={unversionedId:"features/model_interpretability/ModelInterpretability - Snow Leopard Detection",id:"features/model_interpretability/ModelInterpretability - Snow Leopard Detection",isDocsHomePage:!1,title:"ModelInterpretability - Snow Leopard Detection",description:"Automated Snow Leopard Detection with Microsoft ML for Apache Spark",source:"@site/docs/features/model_interpretability/ModelInterpretability - Snow Leopard Detection.md",sourceDirName:"features/model_interpretability",slug:"/features/model_interpretability/ModelInterpretability - Snow Leopard Detection",permalink:"/docs/next/features/model_interpretability/ModelInterpretability - Snow Leopard Detection",version:"current",frontMatter:{title:"ModelInterpretability - Snow Leopard Detection",hide_title:!0,status:"stable",url_path:"features/model_interpretability/ModelInterpretability - Snow Leopard Detection",name:"ModelInterpretability - Snow Leopard Detection"},sidebar:"docs",previous:{title:"About",permalink:"/docs/next/features/model_interpretability/about"},next:{title:"About",permalink:"/docs/next/features/onnx/about"}},m=[{value:"Automated Snow Leopard Detection with Microsoft ML for Apache Spark",id:"automated-snow-leopard-detection-with-microsoft-ml-for-apache-spark",children:[{value:"Your results will look like:",id:"your-results-will-look-like",children:[]}]}],c={toc:m};function u(e){var n=e.components,t=(0,o.Z)(e,i);return(0,a.kt)("wrapper",(0,r.Z)({},c,t,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("h2",{id:"automated-snow-leopard-detection-with-microsoft-ml-for-apache-spark"},"Automated Snow Leopard Detection with Microsoft ML for Apache Spark"),(0,a.kt)("img",{src:"https://mmlspark.blob.core.windows.net/graphics/SnowLeopardAD/SLTrust.PNG",width:"900"}),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'import os\nif os.environ.get("AZURE_SERVICE", None) == "Microsoft.ProjectArcadia":\n    from pyspark.sql import SparkSession\n    spark = SparkSession.builder.getOrCreate()\n    from notebookutils.mssparkutils.credentials import getSecret\n    os.environ["BING_IMAGE_SEARCH_KEY"] = getSecret("mmlspark-keys", "bing-image-search-key")\n\n# WARNING this notebook requires alot of memory.\n# If you get a heap space error, try dropping the number of images bing returns\n# or by writing out the images to parquet first\n\n# Replace the following with a line like: BING_IMAGE_SEARCH_KEY =  "hdwo2oyd3o928s....."\nBING_IMAGE_SEARCH_KEY = os.environ["BING_IMAGE_SEARCH_KEY"]\n')),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'from synapse.ml.cognitive import *\nfrom synapse.ml.core.spark import FluentAPI\nfrom pyspark.sql.functions import lit\n\ndef bingPhotoSearch(name, queries, pages):\n  offsets = [offset*10 for offset in range(0, pages)] \n  parameters = [(query, offset) for offset in offsets for query in queries]\n  \n  return spark.createDataFrame(parameters, ("queries","offsets")) \\\n    .mlTransform(\n      BingImageSearch()                             # Apply Bing Image Search\n        .setSubscriptionKey(BING_IMAGE_SEARCH_KEY)  # Set the API Key\n        .setOffsetCol("offsets")                    # Specify a column containing the offsets\n        .setQueryCol("queries")                     # Specify a column containing the query words\n        .setCount(10)                               # Specify the number of images to return per offset\n        .setImageType("photo")                      # Specify a filter to ensure we get photos\n        .setOutputCol("images")) \\\n    .mlTransform(BingImageSearch.getUrlTransformer("images", "urls")) \\\n    .withColumn("labels", lit(name)) \\\n    .limit(400)\n\n')),(0,a.kt)("img",{src:"https://mmlspark.blob.core.windows.net/graphics/SparkSummit2/cog_services.png",width:"900"}),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'def displayDF(df, n=5, image_cols = set(["urls"])):\n  rows = df.take(n)\n  cols = df.columns\n  header = "".join(["<th>" + c  + "</th>" for c in cols])\n  \n  style = """\n<!DOCTYPE html>\n<html>\n<head>\n\n</head>"""\n  \n  table = []\n  for row in rows:\n    table.append("<tr>")\n    for col in cols:\n      if col in image_cols:\n        rep = \'<img src="{}",  width="100">\'.format(row[col])\n      else:\n        rep = row[col]\n      table.append("<td>{}</td>".format(rep))\n    table.append("</tr>")\n  tableHTML = "".join(table)\n  \n  body = """\n<body>\n<table>\n  <tr>\n    {} \n  </tr>\n  {}\n</table>\n</body>\n</html>\n  """.format(header, tableHTML)\n  try:\n    displayHTML(style + body)\n  except:\n    pass\n')),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'snowLeopardQueries = ["snow leopard"]\nsnowLeopardUrls = bingPhotoSearch("snow leopard", snowLeopardQueries, pages=100)\ndisplayDF(snowLeopardUrls)\n')),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'randomWords = spark.read.parquet("wasbs://publicwasb@mmlspark.blob.core.windows.net/random_words.parquet").cache()\nrandomWords.show()\n')),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'randomLinks = randomWords \\\n  .mlTransform(BingImageSearch()\n    .setSubscriptionKey(BING_IMAGE_SEARCH_KEY)\n    .setCount(10)\n    .setQueryCol("words")\n    .setOutputCol("images")) \\\n  .mlTransform(BingImageSearch.getUrlTransformer("images", "urls")) \\\n  .withColumn("label", lit("other")) \\\n  .limit(400)\n  \ndisplayDF(randomLinks)\n')),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'images = snowLeopardUrls.union(randomLinks).distinct().repartition(100)\\\n  .mlTransform(BingImageSearch.downloadFromUrls("urls", "image", concurrency=5, timeout=5000))\\\n  .dropna()\n\ntrain, test = images.randomSplit([.7,.3], seed=1)\n')),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'from pyspark.ml import Pipeline\n\nfrom pyspark.ml.feature import StringIndexer\n\nfrom pyspark.ml.classification import LogisticRegression\n\nfrom pyspark.sql.functions import udf\n\nfrom synapse.ml.downloader import ModelDownloader\n\nfrom synapse.ml.cntk import ImageFeaturizer\n\nfrom synapse.ml.stages import UDFTransformer\n\nfrom pyspark.sql.types import *\n\n\n\ndef getIndex(row):\n\n  return float(row[1])\n\n\n\nif os.environ.get("AZURE_SERVICE", None) == "Microsoft.ProjectArcadia":\n\n  network = ModelDownloader(spark, "abfss://synapse@mmlsparkeuap.dfs.core.windows.net/models/").downloadByName("ResNet50")\n\nelse:\n\n  network = ModelDownloader(spark, "dbfs:/Models/").downloadByName("ResNet50")\n\n\n\nmodel = Pipeline(stages=[\n\n  StringIndexer(inputCol = "labels", outputCol="index"),\n\n  ImageFeaturizer(inputCol="image", outputCol="features", cutOutputLayers=1).setModel(network),\n\n  LogisticRegression(maxIter=5, labelCol="index", regParam=10.0),\n\n  UDFTransformer()\\\n\n      .setUDF(udf(getIndex, DoubleType()))\\\n\n      .setInputCol("probability")\\\n\n      .setOutputCol("leopard_prob")\n\n])\n\n\n\nfitModel = model.fit(train)\n')),(0,a.kt)("img",{src:"https://mmlspark.blob.core.windows.net/graphics/SnowLeopardAD/SLPipeline.PNG",width:"900"}),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'def plotConfusionMatrix(df, label, prediction, classLabels):\n\n  from synapse.ml.plot import confusionMatrix\n\n  import matplotlib.pyplot as plt\n\n  fig = plt.figure(figsize=(4.5, 4.5))\n\n  confusionMatrix(df, label, prediction, classLabels)\n\n  display(fig)\n\n\n\nif os.environ.get("AZURE_SERVICE", None) != "Microsoft.ProjectArcadia":\n\n  plotConfusionMatrix(fitModel.transform(test), "index", "prediction", fitModel.stages[0].labels)\n')),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'import urllib.request\n\nfrom synapse.ml.lime import ImageLIME\n\n\n\ntest_image_url = "https://mmlspark.blob.core.windows.net/graphics/SnowLeopardAD/snow_leopard1.jpg"\n\nwith urllib.request.urlopen(test_image_url) as url:\n\n    barr = url.read()\n\ntest_subsample = spark.createDataFrame([(bytearray(barr),)], ["image"])\n\n\n\nlime = ImageLIME()\\\n\n  .setModel(fitModel)\\\n\n  .setPredictionCol("leopard_prob")\\\n\n  .setOutputCol("weights")\\\n\n  .setInputCol("image")\\\n\n  .setCellSize(100.0)\\\n\n  .setModifier(50.0)\\\n\n  .setNSamples(300)\n\n\n\nresult = lime.transform(test_subsample)\n')),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"import matplotlib.pyplot as plt\nimport PIL, io, numpy as np\n\ndef plot_superpixels(row):\n    image_bytes = row['image']\n    superpixels = row['superpixels']['clusters']\n    weights = list(row['weights'])\n    mean_weight = np.percentile(weights,90)\n    img = (PIL.Image.open(io.BytesIO(image_bytes))).convert('RGBA')\n    image_array = np.asarray(img).copy()\n    for (sp, w) in zip(superpixels, weights):\n        if w > mean_weight:\n            for (x, y) in sp:\n                image_array[y, x, 1] = 255\n                image_array[y, x, 3] = 200\n    plt.clf()\n    plt.imshow(image_array)\n    display()\n\n# Gets first row from the LIME-transformed data frame\nif os.environ.get(\"AZURE_SERVICE\", None) != \"Microsoft.ProjectArcadia\":\n    plot_superpixels(result.take(1)[0])\n")),(0,a.kt)("h3",{id:"your-results-will-look-like"},"Your results will look like:"),(0,a.kt)("img",{src:"https://mmlspark.blob.core.windows.net/graphics/SnowLeopardAD/lime_results.png",width:"900"}))}u.isMDXComponent=!0}}]);