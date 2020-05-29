// Databricks notebook source
//SparkSession
spark

// COMMAND ----------

//This range of numbers represents a distributed collection. When
//run on a cluster, each part of this range of numbers exists on a different executor. This is a Spark
//DataFrame.
val myRange = spark.range(1000).toDF("number")

// COMMAND ----------

//Datasets
case class Flight(DEST_COUNTRY_NAME: String, ORIGIN_COUNTRY_NAME: String, count: BigInt)

// COMMAND ----------

//https://github.com/databricks/Spark-The-Definitive-Guide/tree/master/data
val flightsDF = spark.read.parquet("/FileStore/tables/part_r_00000_1a9822ba_b8fb_4d8e_844a_ea30d0801b9e_gz-11168.parquet")
val flights = flightsDF.as[Flight]

// COMMAND ----------

flights.show(2)

// COMMAND ----------

flights.first.DEST_COUNTRY_NAME

// COMMAND ----------

def originIsDestination(flight_row: Flight): Boolean = {
return flight_row.ORIGIN_COUNTRY_NAME == flight_row.DEST_COUNTRY_NAME
}

// COMMAND ----------

flights.filter(flight_row => originIsDestination(flight_row)).first()

// COMMAND ----------

//this dataset is small enough for us to collect to the driver (as an Array of Flights)
//on which we can operate and perform the exact same filtering operation
flights.collect().filter(flight_row => originIsDestination(flight_row))

// COMMAND ----------

//Extraction using maps
val destinations = flights.map(f => f.DEST_COUNTRY_NAME)
//Execution happens now
val localDestinations = destinations.take(5)

// COMMAND ----------

//Advanced join
case class FlightMetadata(count: BigInt, randomData: BigInt)

val flightsMeta = spark.range(500).map(x => (x, scala.util.Random.nextLong))
.withColumnRenamed("_1", "count").withColumnRenamed("_2", "randomData")
.as[FlightMetadata]

// COMMAND ----------

//Notice that we end up with a Dataset of a sort of key-value pair, in which each row represents a Flight and the Flight Metadata.
val flights2 = flights
.joinWith(flightsMeta, flights.col("count") === flightsMeta.col("count"))

// COMMAND ----------

flights2.selectExpr("_1.DEST_COUNTRY_NAME").first()

// COMMAND ----------

flights2.take(2)

// COMMAND ----------

flights2.selectExpr("_2.randomData").first()

// COMMAND ----------

//Normal non JVM join
val flights2 = flights.join(flightsMeta, Seq("count"))
//It’s also important to note that there are no problems joining a DataFrame and a Dataset—we end up with the same result:

// COMMAND ----------

//Agg
flights.groupBy("DEST_COUNTRY_NAME").count().first()

// COMMAND ----------

// can use a function fot group by field
flights.groupByKey(x => x.DEST_COUNTRY_NAME).count()

// COMMAND ----------

//Function aggregations
//After we perform a grouping with a key on a Dataset, we can operate on the Key Value Dataset with functions that will manipulate the groupings as raw objects:
def grpSum(countryName:String, values: Iterator[Flight]) = {
values.dropWhile(_.count < 5).map(x => (countryName, x))
}
flights.groupByKey(x => x.DEST_COUNTRY_NAME).flatMapGroups(grpSum).show(5)

// COMMAND ----------

//We can even create new manipulations and define how groups should be reduced:
def sum2(left:Flight, right:Flight) = {
Flight(left.DEST_COUNTRY_NAME, null, left.count + right.count)
}

flights.groupByKey(x => x.DEST_COUNTRY_NAME).reduceGroups((l, r) => sum2(l, r))
.take(5)

// COMMAND ----------


