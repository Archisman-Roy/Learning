// Code start

// Reading data
// Source Github link
// https://github.com/MarkCLewis/BigDataAnalyticswithSpark/blob/master/src/main/scala/standardscala/TempData.scala

package standardScala

// Add dependency in sbt file 
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
// Code end
