(self.webpackChunksynapseml=self.webpackChunksynapseml||[]).push([[1631],{3905:function(e,n,t){"use strict";t.d(n,{Zo:function(){return m},kt:function(){return y}});var a=t(7294);function o(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function r(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,a)}return t}function l(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?r(Object(t),!0).forEach((function(n){o(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):r(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function s(e,n){if(null==e)return{};var t,a,o=function(e,n){if(null==e)return{};var t,a,o={},r=Object.keys(e);for(a=0;a<r.length;a++)t=r[a],n.indexOf(t)>=0||(o[t]=e[t]);return o}(e,n);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(a=0;a<r.length;a++)t=r[a],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(o[t]=e[t])}return o}var i=a.createContext({}),c=function(e){var n=a.useContext(i),t=n;return e&&(t="function"==typeof e?e(n):l(l({},n),e)),t},m=function(e){var n=c(e.components);return a.createElement(i.Provider,{value:n},e.children)},u={inlineCode:"code",wrapper:function(e){var n=e.children;return a.createElement(a.Fragment,{},n)}},p=a.forwardRef((function(e,n){var t=e.components,o=e.mdxType,r=e.originalType,i=e.parentName,m=s(e,["components","mdxType","originalType","parentName"]),p=c(t),y=o,d=p["".concat(i,".").concat(y)]||p[y]||u[y]||r;return t?a.createElement(d,l(l({ref:n},m),{},{components:t})):a.createElement(d,l({ref:n},m))}));function y(e,n){var t=arguments,o=n&&n.mdxType;if("string"==typeof e||o){var r=t.length,l=new Array(r);l[0]=p;var s={};for(var i in n)hasOwnProperty.call(n,i)&&(s[i]=n[i]);s.originalType=e,s.mdxType="string"==typeof e?e:o,l[1]=s;for(var c=2;c<r;c++)l[c]=t[c];return a.createElement.apply(null,l)}return a.createElement.apply(null,t)}p.displayName="MDXCreateElement"},1332:function(e,n,t){"use strict";var a=t(7294);n.Z=function(e){var n=e.children,t=e.hidden,o=e.className;return a.createElement("div",{role:"tabpanel",hidden:t,className:o},n)}},5386:function(e,n,t){"use strict";t.d(n,{Z:function(){return u}});var a=t(7294),o=t(8578);var r=function(){var e=(0,a.useContext)(o.Z);if(null==e)throw new Error('"useUserPreferencesContext" is used outside of "Layout" component.');return e},l=t(6010),s="tabItem_2kG2",i="tabItemActive_3NDg";var c=37,m=39;var u=function(e){var n=e.lazy,t=e.block,o=e.defaultValue,u=e.values,p=e.groupId,y=e.className,d=r(),f=d.tabGroupChoices,T=d.setTabGroupChoices,v=(0,a.useState)(o),Z=v[0],g=v[1],b=a.Children.toArray(e.children),h=[];if(null!=p){var k=f[p];null!=k&&k!==Z&&u.some((function(e){return e.value===k}))&&g(k)}var A=function(e){var n=e.currentTarget,t=h.indexOf(n),a=u[t].value;g(a),null!=p&&(T(p,a),setTimeout((function(){var e,t,a,o,r,l,s,c;(e=n.getBoundingClientRect(),t=e.top,a=e.left,o=e.bottom,r=e.right,l=window,s=l.innerHeight,c=l.innerWidth,t>=0&&r<=c&&o<=s&&a>=0)||(n.scrollIntoView({block:"center",behavior:"smooth"}),n.classList.add(i),setTimeout((function(){return n.classList.remove(i)}),2e3))}),150))},D=function(e){var n,t;switch(e.keyCode){case m:var a=h.indexOf(e.target)+1;t=h[a]||h[0];break;case c:var o=h.indexOf(e.target)-1;t=h[o]||h[h.length-1]}null==(n=t)||n.focus()};return a.createElement("div",{className:"tabs-container"},a.createElement("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,l.Z)("tabs",{"tabs--block":t},y)},u.map((function(e){var n=e.value,t=e.label;return a.createElement("li",{role:"tab",tabIndex:Z===n?0:-1,"aria-selected":Z===n,className:(0,l.Z)("tabs__item",s,{"tabs__item--active":Z===n}),key:n,ref:function(e){return h.push(e)},onKeyDown:D,onFocus:A,onClick:A},t)}))),n?(0,a.cloneElement)(b.filter((function(e){return e.props.value===Z}))[0],{className:"margin-vert--md"}):a.createElement("div",{className:"margin-vert--md"},b.map((function(e,n){return(0,a.cloneElement)(e,{key:n,hidden:e.props.value!==Z})}))))}},8578:function(e,n,t){"use strict";var a=(0,t(7294).createContext)(void 0);n.Z=a},1989:function(e,n,t){"use strict";var a=t(7294),o=t(2263);n.Z=function(e){var n=e.className,t=e.py,r=e.scala,l=e.sourceLink,s=(0,o.Z)().siteConfig.customFields.version,i="https://mmlspark.blob.core.windows.net/docs/"+s+"/pyspark/"+t,c="https://mmlspark.blob.core.windows.net/docs/"+s+"/scala/"+r;return a.createElement("table",null,a.createElement("tbody",null,a.createElement("tr",null,a.createElement("td",null,a.createElement("strong",null,"Python API: "),a.createElement("a",{href:i},n)),a.createElement("td",null,a.createElement("strong",null,"Scala API: "),a.createElement("a",{href:c},n)),a.createElement("td",null,a.createElement("strong",null,"Source: "),a.createElement("a",{href:l},n)))))}},7452:function(e,n,t){"use strict";t.r(n),t.d(n,{frontMatter:function(){return m},contentTitle:function(){return u},metadata:function(){return p},toc:function(){return y},default:function(){return f}});var a=t(2122),o=t(9756),r=(t(7294),t(3905)),l=t(5386),s=t(1332),i=t(1989),c=["components"],m={},u=void 0,p={unversionedId:"documentation/transformers/cognitive/_AnomalyDetection",id:"version-0.9.1/documentation/transformers/cognitive/_AnomalyDetection",isDocsHomePage:!1,title:"_AnomalyDetection",description:"\x3c!--",source:"@site/versioned_docs/version-0.9.1/documentation/transformers/cognitive/_AnomalyDetection.md",sourceDirName:"documentation/transformers/cognitive",slug:"/documentation/transformers/cognitive/_AnomalyDetection",permalink:"/docs/documentation/transformers/cognitive/_AnomalyDetection",version:"0.9.1",frontMatter:{}},y=[{value:"Anomaly Detection",id:"anomaly-detection",children:[{value:"DetectLastAnomaly",id:"detectlastanomaly",children:[]},{value:"DetectAnomalies",id:"detectanomalies",children:[]},{value:"SimpleDetectAnomalies",id:"simpledetectanomalies",children:[]}]}],d={toc:y};function f(e){var n=e.components,t=(0,o.Z)(e,c);return(0,r.kt)("wrapper",(0,a.Z)({},d,t,{components:n,mdxType:"MDXLayout"}),(0,r.kt)("h2",{id:"anomaly-detection"},"Anomaly Detection"),(0,r.kt)("h3",{id:"detectlastanomaly"},"DetectLastAnomaly"),(0,r.kt)(l.Z,{defaultValue:"py",values:[{label:"Python",value:"py"},{label:"Scala",value:"scala"}],mdxType:"Tabs"},(0,r.kt)(s.Z,{value:"py",mdxType:"TabItem"},(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'from synapse.ml.cognitive import *\n\nanomalyKey = os.environ.get("ANOMALY_API_KEY", getSecret("anomaly-api-key"))\ndf = (spark.createDataFrame([\n    ("1972-01-01T00:00:00Z", 826.0),\n    ("1972-02-01T00:00:00Z", 799.0),\n    ("1972-03-01T00:00:00Z", 890.0),\n    ("1972-04-01T00:00:00Z", 900.0),\n    ("1972-05-01T00:00:00Z", 766.0),\n    ("1972-06-01T00:00:00Z", 805.0),\n    ("1972-07-01T00:00:00Z", 821.0),\n    ("1972-08-01T00:00:00Z", 20000.0),\n    ("1972-09-01T00:00:00Z", 883.0),\n    ("1972-10-01T00:00:00Z", 898.0),\n    ("1972-11-01T00:00:00Z", 957.0),\n    ("1972-12-01T00:00:00Z", 924.0),\n    ("1973-01-01T00:00:00Z", 881.0),\n    ("1973-02-01T00:00:00Z", 837.0),\n    ("1973-03-01T00:00:00Z", 90000.0)\n], ["timestamp", "value"])\n      .withColumn("group", lit(1))\n      .withColumn("inputs", struct(col("timestamp"), col("value")))\n      .groupBy(col("group"))\n      .agg(sort_array(collect_list(col("inputs"))).alias("inputs")))\n\ndla = (DetectLastAnomaly()\n      .setSubscriptionKey(anomalyKey)\n      .setLocation("westus2")\n      .setOutputCol("anomalies")\n      .setSeriesCol("inputs")\n      .setGranularity("monthly")\n      .setErrorCol("errors"))\n\ndisplay(dla.transform(df))\n'))),(0,r.kt)(s.Z,{value:"scala",mdxType:"TabItem"},(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-scala"},'import com.microsoft.azure.synapse.ml.cognitive._\nimport spark.implicits._\nimport org.apache.spark.sql.functions.{col, collect_list, lit, sort_array, struct}\n\nval anomalyKey = sys.env.getOrElse("ANOMALY_API_KEY", None)\nval df = (Seq(\n    ("1972-01-01T00:00:00Z", 826.0),\n    ("1972-02-01T00:00:00Z", 799.0),\n    ("1972-03-01T00:00:00Z", 890.0),\n    ("1972-04-01T00:00:00Z", 900.0),\n    ("1972-05-01T00:00:00Z", 766.0),\n    ("1972-06-01T00:00:00Z", 805.0),\n    ("1972-07-01T00:00:00Z", 821.0),\n    ("1972-08-01T00:00:00Z", 20000.0),\n    ("1972-09-01T00:00:00Z", 883.0),\n    ("1972-10-01T00:00:00Z", 898.0),\n    ("1972-11-01T00:00:00Z", 957.0),\n    ("1972-12-01T00:00:00Z", 924.0),\n    ("1973-01-01T00:00:00Z", 881.0),\n    ("1973-02-01T00:00:00Z", 837.0),\n    ("1973-03-01T00:00:00Z", 90000.0)\n  ).toDF("timestamp","value")\n    .withColumn("group", lit(1))\n    .withColumn("inputs", struct(col("timestamp"), col("value")))\n    .groupBy(col("group"))\n    .agg(sort_array(collect_list(col("inputs"))).alias("inputs")))\n\nval dla = (new DetectLastAnomaly()\n            .setSubscriptionKey(anomalyKey)\n            .setLocation("westus2")\n            .setOutputCol("anomalies")\n            .setSeriesCol("inputs")\n            .setGranularity("monthly")\n            .setErrorCol("errors"))\n\ndisplay(dla.transform(df))\n')))),(0,r.kt)(i.Z,{className:"DetectLastAnomaly",py:"synapse.ml.cognitive.html#module-synapse.ml.cognitive.DetectLastAnomaly",scala:"com/microsoft/azure/synapse/ml/cognitive/DetectLastAnomaly.html",sourceLink:"https://github.com/microsoft/SynapseML/blob/master/cognitive/src/main/scala/com/microsoft/azure/synapse/ml/cognitive/AnomalyDetection.scala",mdxType:"DocTable"}),(0,r.kt)("h3",{id:"detectanomalies"},"DetectAnomalies"),(0,r.kt)(l.Z,{defaultValue:"py",values:[{label:"Python",value:"py"},{label:"Scala",value:"scala"}],mdxType:"Tabs"},(0,r.kt)(s.Z,{value:"py",mdxType:"TabItem"},(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'from synapse.ml.cognitive import *\n\nanomalyKey = os.environ.get("ANOMALY_API_KEY", getSecret("anomaly-api-key"))\ndf = (spark.createDataFrame([\n    ("1972-01-01T00:00:00Z", 826.0),\n    ("1972-02-01T00:00:00Z", 799.0),\n    ("1972-03-01T00:00:00Z", 890.0),\n    ("1972-04-01T00:00:00Z", 900.0),\n    ("1972-05-01T00:00:00Z", 766.0),\n    ("1972-06-01T00:00:00Z", 805.0),\n    ("1972-07-01T00:00:00Z", 821.0),\n    ("1972-08-01T00:00:00Z", 20000.0),\n    ("1972-09-01T00:00:00Z", 883.0),\n    ("1972-10-01T00:00:00Z", 898.0),\n    ("1972-11-01T00:00:00Z", 957.0),\n    ("1972-12-01T00:00:00Z", 924.0),\n    ("1973-01-01T00:00:00Z", 881.0),\n    ("1973-02-01T00:00:00Z", 837.0),\n    ("1973-03-01T00:00:00Z", 90000.0)\n], ["timestamp", "value"])\n      .withColumn("group", lit(1))\n      .withColumn("inputs", struct(col("timestamp"), col("value")))\n      .groupBy(col("group"))\n      .agg(sort_array(collect_list(col("inputs"))).alias("inputs")))\n\nda = (DetectAnomalies()\n      .setSubscriptionKey(anomalyKey)\n      .setLocation("westus2")\n      .setOutputCol("anomalies")\n      .setSeriesCol("inputs")\n      .setGranularity("monthly"))\n\ndisplay(da.transform(df))\n'))),(0,r.kt)(s.Z,{value:"scala",mdxType:"TabItem"},(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-scala"},'import com.microsoft.azure.synapse.ml.cognitive._\nimport spark.implicits._\n\nval anomalyKey = sys.env.getOrElse("ANOMALY_API_KEY", None)\nval df = (Seq(\n    ("1972-01-01T00:00:00Z", 826.0),\n    ("1972-02-01T00:00:00Z", 799.0),\n    ("1972-03-01T00:00:00Z", 890.0),\n    ("1972-04-01T00:00:00Z", 900.0),\n    ("1972-05-01T00:00:00Z", 766.0),\n    ("1972-06-01T00:00:00Z", 805.0),\n    ("1972-07-01T00:00:00Z", 821.0),\n    ("1972-08-01T00:00:00Z", 20000.0),\n    ("1972-09-01T00:00:00Z", 883.0),\n    ("1972-10-01T00:00:00Z", 898.0),\n    ("1972-11-01T00:00:00Z", 957.0),\n    ("1972-12-01T00:00:00Z", 924.0),\n    ("1973-01-01T00:00:00Z", 881.0),\n    ("1973-02-01T00:00:00Z", 837.0),\n    ("1973-03-01T00:00:00Z", 90000.0)\n  ).toDF("timestamp","value")\n    .withColumn("group", lit(1))\n    .withColumn("inputs", struct(col("timestamp"), col("value")))\n    .groupBy(col("group"))\n    .agg(sort_array(collect_list(col("inputs"))).alias("inputs")))\n\nval da = (new DetectAnomalies()\n            .setSubscriptionKey(anomalyKey)\n            .setLocation("westus2")\n            .setOutputCol("anomalies")\n            .setSeriesCol("inputs")\n            .setGranularity("monthly"))\n\ndisplay(da.transform(df))\n')))),(0,r.kt)(i.Z,{className:"DetectAnomalies",py:"mmlspark.cognitive.html#module-mmlspark.cognitive.DetectAnomalies",scala:"com/microsoft/azure/synapse/ml/cognitive/DetectAnomalies.html",sourceLink:"https://github.com/microsoft/SynapseML/blob/master/cognitive/src/main/scala/com/microsoft/azure/synapse/ml/cognitive/AnomalyDetection.scala",mdxType:"DocTable"}),(0,r.kt)("h3",{id:"simpledetectanomalies"},"SimpleDetectAnomalies"),(0,r.kt)(l.Z,{defaultValue:"py",values:[{label:"Python",value:"py"},{label:"Scala",value:"scala"}],mdxType:"Tabs"},(0,r.kt)(s.Z,{value:"py",mdxType:"TabItem"},(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'from synapse.ml.cognitive import *\n\nanomalyKey = os.environ.get("ANOMALY_API_KEY", getSecret("anomaly-api-key"))\ndf = (spark.createDataFrame([\n    ("1972-01-01T00:00:00Z", 826.0, 1.0),\n    ("1972-02-01T00:00:00Z", 799.0, 1.0),\n    ("1972-03-01T00:00:00Z", 890.0, 1.0),\n    ("1972-04-01T00:00:00Z", 900.0, 1.0),\n    ("1972-05-01T00:00:00Z", 766.0, 1.0),\n    ("1972-06-01T00:00:00Z", 805.0, 1.0),\n    ("1972-07-01T00:00:00Z", 821.0, 1.0),\n    ("1972-08-01T00:00:00Z", 20000.0, 1.0),\n    ("1972-09-01T00:00:00Z", 883.0, 1.0),\n    ("1972-10-01T00:00:00Z", 898.0, 1.0),\n    ("1972-11-01T00:00:00Z", 957.0, 1.0),\n    ("1972-12-01T00:00:00Z", 924.0, 1.0),\n    ("1973-01-01T00:00:00Z", 881.0, 1.0),\n    ("1973-02-01T00:00:00Z", 837.0, 1.0),\n    ("1973-03-01T00:00:00Z", 90000.0, 1.0),\n    ("1972-01-01T00:00:00Z", 826.0, 2.0),\n    ("1972-02-01T00:00:00Z", 799.0, 2.0),\n    ("1972-03-01T00:00:00Z", 890.0, 2.0),\n    ("1972-04-01T00:00:00Z", 900.0, 2.0),\n    ("1972-05-01T00:00:00Z", 766.0, 2.0),\n    ("1972-06-01T00:00:00Z", 805.0, 2.0),\n    ("1972-07-01T00:00:00Z", 821.0, 2.0),\n    ("1972-08-01T00:00:00Z", 20000.0, 2.0),\n    ("1972-09-01T00:00:00Z", 883.0, 2.0),\n    ("1972-10-01T00:00:00Z", 898.0, 2.0),\n    ("1972-11-01T00:00:00Z", 957.0, 2.0),\n    ("1972-12-01T00:00:00Z", 924.0, 2.0),\n    ("1973-01-01T00:00:00Z", 881.0, 2.0),\n    ("1973-02-01T00:00:00Z", 837.0, 2.0),\n    ("1973-03-01T00:00:00Z", 90000.0, 2.0)\n], ["timestamp", "value", "group"])\n\nsda = (SimpleDetectAnomalies()\n      .setSubscriptionKey(anomalyKey)\n      .setLocation("westus2")\n      .setOutputCol("anomalies")\n      .setSeriesCol("inputs")\n      .setGranularity("monthly"))\n\ndisplay(sda.transform(df))\n'))),(0,r.kt)(s.Z,{value:"scala",mdxType:"TabItem"},(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-scala"},'import com.microsoft.azure.synapse.ml.cognitive._\nimport spark.implicits._\n\nval anomalyKey = sys.env.getOrElse("ANOMALY_API_KEY", None)\nval baseSeq = Seq(\n    ("1972-01-01T00:00:00Z", 826.0),\n    ("1972-02-01T00:00:00Z", 799.0),\n    ("1972-03-01T00:00:00Z", 890.0),\n    ("1972-04-01T00:00:00Z", 900.0),\n    ("1972-05-01T00:00:00Z", 766.0),\n    ("1972-06-01T00:00:00Z", 805.0),\n    ("1972-07-01T00:00:00Z", 821.0),\n    ("1972-08-01T00:00:00Z", 20000.0),\n    ("1972-09-01T00:00:00Z", 883.0),\n    ("1972-10-01T00:00:00Z", 898.0),\n    ("1972-11-01T00:00:00Z", 957.0),\n    ("1972-12-01T00:00:00Z", 924.0),\n    ("1973-01-01T00:00:00Z", 881.0),\n    ("1973-02-01T00:00:00Z", 837.0),\n    ("1973-03-01T00:00:00Z", 9000.0)\n  )\nval df = (baseSeq.map(p => (p._1,p._2,1.0))\n    .++(baseSeq.map(p => (p._1,p._2,2.0)))\n    .toDF("timestamp","value","group"))\n\nval sda = (new SimpleDetectAnomalies()\n            .setSubscriptionKey(anomalyKey)\n            .setLocation("westus2")\n            .setOutputCol("anomalies")\n            .setGroupbyCol("group")\n            .setGranularity("monthly"))\n\ndisplay(sda.transform(df))\n')))),(0,r.kt)(i.Z,{className:"SimpleDetectAnomalies",py:"mmlspark.cognitive.html#module-mmlspark.cognitive.SimpleDetectAnomalies",scala:"com/microsoft/azure/synapse/ml/cognitive/SimpleDetectAnomalies.html",sourceLink:"https://github.com/microsoft/SynapseML/blob/master/cognitive/src/main/scala/com/microsoft/azure/synapse/ml/cognitive/AnomalyDetection.scala",mdxType:"DocTable"}))}f.isMDXComponent=!0},6010:function(e,n,t){"use strict";function a(e){var n,t,o="";if("string"==typeof e||"number"==typeof e)o+=e;else if("object"==typeof e)if(Array.isArray(e))for(n=0;n<e.length;n++)e[n]&&(t=a(e[n]))&&(o&&(o+=" "),o+=t);else for(n in e)e[n]&&(o&&(o+=" "),o+=n);return o}function o(){for(var e,n,t=0,o="";t<arguments.length;)(e=arguments[t++])&&(n=a(e))&&(o&&(o+=" "),o+=n);return o}t.d(n,{Z:function(){return o}})}}]);