import scala.python.*
import mlx.core.*

@main def runMain(): Unit =
  println("--- construction ---")
  val z: Tensor[Float32]  = Tensor.zeros((2, 3))
  val o: Tensor[Float32]  = Tensor.ones((2, 3))
  val f: Tensor[Int32]    = Tensor.full((2, 3), 7)
  val r: Tensor[Int32]    = Tensor.arange[Int32](6).reshape((2, 3))
  val l: Tensor[Float32]  = Tensor.linspace[Float32](0.0, 1.0, 5)
  val i: Tensor[Float32]  = Tensor.eye[Float32](3)
  val a: Tensor[Int32]    = Tensor.array[Int32]((1, 2, 3, 4))
  println(z); println(o); println(f); println(r); println(l); println(i); println(a)

  println("--- operators ---")
  val s = (r + 1) * 2 - 3
  println(s)
  println(-s)
  println(r ** 2)

  println("--- elementwise unary ---")
  println(Tensor.exp(l))
  println(Tensor.sigmoid(l))
  println(Tensor.sqrt(Tensor.full[Float32, (Int, Int), Double]((2, 2), 9.0)))

  println("--- comparisons ---")
  val mask: Tensor[Bool] = r > Tensor.full[Int32, (Int, Int), Int]((2, 3), 2)
  println(mask)
  println(Tensor.where(mask, r, -r))

  println("--- reductions ---")
  println(r.sum())
  println(r.sum(0))
  println(r.sum(1))
  println(Tensor.mean(l))
  println(Tensor.argmax(r, 1))

  println("--- shape ---")
  val rt = r.transpose()
  println(rt)
  println(r.reshape((3, 2)))
  println(Tensor.expandDims(r, 0))
  println(Tensor.broadcastTo(a, (3, 4)))

  println("--- combining ---")
  println(Tensor.concatenate(r, r, 0))
  println(Tensor.stack(a, a, a, 0))

  println("--- matmul ---")
  val m1 = Tensor.arange[Float32](6).reshape((2, 3))
  val m2 = Tensor.arange[Float32](6).reshape((3, 2))
  println(m1 @@ m2)

  println("--- scalar item ---")
  val si: Int    = r.sum().item
  val sd: Double = Tensor.mean(l).item
  println(s"sum(r)=$si  mean(l)=$sd")
