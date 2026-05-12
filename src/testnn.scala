import scala.python.*
import mlx.utils.*
import mlx.nn.*

class MyMlp(val inputDim: Int, val outputDim: Int, hiddenDim: Int = 32) extends Component:
  override def toString(): String = s"MLP($inputDim --$hiddenDim--> $outputDim)"
  def asPy: PyDynamic = this.asInstanceOf

@main def runTestNN(): Unit =
  println("Greetings")
  val model = MyMlp(128, 4)
  println(model)
  println(model.asPy.toString)
  println(model.asPy.parameters())
