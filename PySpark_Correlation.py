from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.stat import Statistics

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

#Create our dataframe from CSV file
df = spark.read.load('/filepath/Icecream.csv', format="csv", sep=",", inferSchema="true", header="true")
df.printSchema()
df.show(2,truncate= True)

#Vectorize features
assembler = VectorAssembler(inputCols=["cons", "temp"],outputCol="variables")
output = assembler.transform(df)
print(type(output))

#Select variables vector and calculate the correlation
output = output.select("variables")
corre = Correlation.corr(output, "variables").head()
print("Pearson correlation matrix:\n" + str(corre[0]))
