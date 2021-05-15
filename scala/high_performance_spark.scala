// Databricks notebook source
// This is a practice workbook as I read the High Performance Spark book by Holden Karau, Rachel Warren
// Website link : https://www.oreilly.com/library/view/high-performance-spark/9781491943199/

// COMMAND ----------

// Spark Shell comes with SparkSession called spark to accompany the SparkContext call sc

// COMMAND ----------

// spark sql imports
import org.apache.spark.sql.{Dataset,DataFrame,SparkSession,Row}
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType


// creating a spark session
val session = SparkSession.builder()
    .enableHiveSupport()
    .getOrCreate() // if an existing session exists, config values may be ignored

import session.implicits._

// COMMAND ----------

// Loading data
// Sales
// https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data?select=sales_train.csv
// Items
// https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data?select=items.csv
// Shops
// https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data?select=shops.csv

case class Sales(date: String, date_block_num: Double, shop_id: Double, item_id: Double, item_price: Double, item_cnt_day: Double)
case class Shops(shop_name: String, shop_id: Double)
case class Items(item_name: String, item_id: Double, item_category_id: Double)

val sales_schema = ScalaReflection.schemaFor[Sales].dataType.asInstanceOf[StructType]

val sales_ds = spark.read
  .option("header", "true")
  .schema(sales_schema)  // passing schema 
  .csv("dbfs:/FileStore/shared_uploads/royarchi31@gmail.com/sales_train.csv")// csv path
  .as[Sales] // convert to DS

val shop_schema = ScalaReflection.schemaFor[Shops].dataType.asInstanceOf[StructType]

val shops_ds = spark.read
  .option("header", "true")
  .schema(shop_schema)  // passing schema 
  .csv("dbfs:/FileStore/shared_uploads/royarchi31@gmail.com/shops.csv")// csv path
  .as[Shops] // convert to DS

val item_schema = ScalaReflection.schemaFor[Items].dataType.asInstanceOf[StructType]

val items_ds = spark.read
  .option("header", "true")
  .schema(item_schema)  // passing schema 
  .csv("dbfs:/FileStore/shared_uploads/royarchi31@gmail.com/items.csv")// csv path
  .as[Items] // convert to DS

// COMMAND ----------

// show sample data
sales_ds.show()

// COMMAND ----------

// show sample data
shops_ds.show()

// COMMAND ----------

// show sample data
items_ds.show()

// COMMAND ----------


