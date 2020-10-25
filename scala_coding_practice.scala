// Dependency addition in sbt file
// https://mvnrepository.com/artifact/org.apache.spark/spark-core
// Ensure scala version required for spark is consistent with the Scala version in use

// Source Github link
// https://github.com/MarkCLewis/BigDataAnalyticswithSpark/tree/master/src/main/scala


// Code start

// Reading data
package standardScala

// Add dependency in sbt file 
// The one below is only required for versions after Scala 2.13
// libraryDependencies += "org.scala-lang.modules" %% "scala-parallel-collections" % "0.2.0"
// Reference stackoverflow link
// https://stackoverflow.com/questions/56542568/missing-import-scala-collection-parallel-in-scala-2-13

import scala.collection.parallel.CollectionConverters._

case class TempData(day: Int, doy: Int, month: Int, year: Int,
                    precip: Double, snow: Double, tave:Double, tmax: Double, tmin: Double)

object TempData {
 def toDoubleOrNeg(s: String): Double = {
  try {
   s.toDouble
  } catch {
   case _:NumberFormatException => -1
  }
 }

 def main(args: Array[String]): Unit = {
  val source = scala.io.Source.fromFile("MN212142_9392.csv") // Place in working directory
  val lines = source.getLines().drop(1)
  val data = lines.flatMap { line =>
   val p = line.split(",")
   if (p(7) == "." || p(8) == "." || p(9) == ".") Seq.empty else
    Seq(TempData(p(0).toInt, p(1).toInt, p(2).toInt, p(4).toInt,
     toDoubleOrNeg(p(5)), toDoubleOrNeg(p(6)), p(7).toDouble, p(8).toDouble,
     p(9).toDouble))
  }.toArray
  source.close()
  println(data.length)

  val maxTemp = data.map(_.tmax).max
  val hotDays = data.filter(_.tmax == maxTemp)
  println(s"Hot days are ${hotDays.mkString(", ")}")

  val hotDay = data.maxBy(_.tmax)
  println(s"Hot day 1 is $hotDay")

  val hotDay2 = data.reduceLeft((d1, d2) => if(d1.tmax >= d2.tmax) d1 else d2)
  println(s"Hot day 2 is $hotDay2")

  val (rainySum, rainyCount2) = data.foldLeft((0.0, 0))({ case ((sum, cnt), td) =>
   if(td.precip < 1.0) (sum, cnt) else (sum+td.tmax, cnt+1)
  }
  )
  println(s"Average Rainy temp is ${rainySum/rainyCount2}")

  val rainyTemps = data.flatMap(td => if(td.precip < 1.0) Seq.empty else Seq(td.tmax))
  println(s"Average Rainy temp is ${rainyTemps.sum/rainyTemps.length}")
   
  val monthGroups = data.groupBy(_.month)
  val monthlyTemp = monthGroups.map { case (m, days) =>
   m -> days.foldLeft(0.0)((sum, td) => sum+td.tmax)/days.length
  }
  monthlyTemp.toSeq.sortBy(_._1) foreach println
   
  // This is the parellel version of FoldLeft. The output is same as obtained from FoldLeft
  // Parellel operations work only on associative functions like addition
  // Parellel operations fail on operation like subtraction since subtraction is not associative
  // aggregate takes in 2 ops: sequential ops and combination ops
  // sequential ops is the same operation of FoldLeft
  // Output of sequential operation is combined in combination operation
  // Combination ops fails to produce right output if the sequential ops is not associative
   
  val (rainySum_par, rainyCount2_par) = data.par.aggregate(0.0 -> 0)({ case ((sum, cnt), td) =>
   if(td.precip < 1.0) (sum, cnt) else (sum+td.tmax, cnt+1)
  }, { case ((s1, c1), (s2, c2)) =>
   (s1+s2, c1+c2)
  })
  println(s"Average Rainy temp is ${rainySum_par/rainyCount2_par}") 
   
   
 }
}

// Code end

// Code start

package standardScala

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object TempDataRDD {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Temp Data").setMaster("local[*]")
    val sc = new SparkContext(conf)

    sc.setLogLevel("WARN")

    val lines = sc.textFile("MN212142_9392.csv").filter(!_.contains("Day"))

    val data = lines.flatMap { line =>
      val p = line.split(",")
      if (p(7) == "." || p(8) == "." || p(9) == ".") Seq.empty else
        Seq(TempData(p(0).toInt, p(1).toInt, p(2).toInt, p(4).toInt,
          TempData.toDoubleOrNeg(p(5)), TempData.toDoubleOrNeg(p(6)), p(7).toDouble, p(8).toDouble,
          p(9).toDouble))
    }.cache()

    val maxTemp = data.map(_.tmax).max
    val hotDays = data.filter(_.tmax == maxTemp)
    println(s"Hot days are ${hotDays.collect().mkString(", ")}")

    println(data.max()(Ordering.by(_.tmax)))

    println(data.reduce((td1, td2) => if (td1.tmax >= td2.tmax) td1 else td2))

    val rainyCount = data.filter(_.precip >= 1.0).count()
    println(s"There are $rainyCount rainy days. There is ${rainyCount * 100.0 / data.count()} percent.")

    val (rainySum, rainyCount2) = data.aggregate(0.0 -> 0)({
      case ((sum, cnt), td) =>
        if (td.precip < 1.0) (sum, cnt) else (sum + td.tmax, cnt + 1)
    }, {
      case ((s1, c1), (s2, c2)) =>
        (s1 + s2, c1 + c2)
    })
    println(s"Average Rainy temp is ${rainySum / rainyCount2}")

    val rainyTemps = data.flatMap(td => if (td.precip < 1.0) Seq.empty else Seq(td.tmax))
    println(s"Average Rainy temp is ${rainyTemps.sum / rainyTemps.count}")

    val monthGroups = data.groupBy(_.month)
    val monthlyHighTemp = monthGroups.map {
      case (m, days) =>
        m -> days.foldLeft(0.0)((sum, td) => sum + td.tmax) / days.size
    }
    val monthlyLowTemp = monthGroups.map {
      case (m, days) =>
        m -> days.foldLeft(0.0)((sum, td) => sum + td.tmin) / days.size
    }

    monthlyHighTemp.collect.sortBy(_._1) foreach println

    println("Stdev of highs: " + data.map(_.tmax).stdev())
    println("Stdev of lows: " + data.map(_.tmin).stdev())
    println("Stdev of averages: " + data.map(_.tave).stdev())
    
    // Takes a RDD and returns a (key, value) tuple for 'ByKey' functions
    val keyedByYear = data.map(td => td.year -> td) // Takes a RDD and returns a (key, value) tuple to
    val averageTempsByYear = keyedByYear.aggregateByKey((0.0, 0))({ case ((sum, cnt), td) =>
      (sum+td.tmax, cnt+1)
    }, { case ((s1, c1), (s2, c2)) => (s1+s2, c1+c2) })

    averageTempsByYear.collect.sortBy(_._1) foreach println

    // print yearly average temp
    val averageTempbyYearComputed = averageTempsByYear.map({ case (s, (m, n)) =>  (s, m/n) })
    averageTempbyYearComputed.collect.sortBy(_._1) foreach println
  
  }
}

// Code end


// Code start

// This code snippet was not run locally since I could not find the raw txt files. 

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

case class Area(code: String, text: String)
case class Series(id: String, area: String, measure: String, title: String)
case class LAData(id: String, year: Int, period: Int, value: Double)

object RDDUnemployment {
  val conf = new SparkConf().setAppName("Temp Data").setMaster("local[*]")
  val sc = new SparkContext(conf)

  sc.setLogLevel("WARN")

  val areas = sc.textFile("data/la.area").filter(!_.contains("area_type")).map { line =>
    val p = line.split("\t").map(_.trim)
    Area(p(1), p(2))
  }.cache()
  areas.take(5) foreach println

  val series = sc.textFile("data/la.series").filter(!_.contains("area_code")).map { line =>
    val p = line.split("\t").map(_.trim)
    Series(p(0), p(2), p(3), p(6))
  }.cache()
  series.take(5) foreach println
  
  val data = sc.textFile("data/la.data.30.Minnesota").filter(!_.contains("year")).map { line =>
    val p = line.split("\t").map(_.trim)
    LAData(p(0), p(1).toInt, p(2).drop(1).toInt, p(3).toDouble)
  }.cache()
  data.take(5) foreach println

  sc.stop()
}

// Code end
