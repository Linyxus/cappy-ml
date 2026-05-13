import scala.python.*
import mlx.utils.*
import mlx.core.*

@main def runMain(): Unit =

  // ====================================================================
  // 1. Construction (companion factories)
  // ====================================================================
  println("=== construction ===")
  val z:  Tensor[Float32] = Tensor.zeros((2, 3))
  val o:  Tensor[Float32] = Tensor.ones((2, 3))
  val f:  Tensor[Int32]   = Tensor.full((2, 3), 7)
  val r:  Tensor[Int32]   = Tensor.arange[Int32](6).reshape((2, 3))
  val l:  Tensor[Float32] = Tensor.linspace[Float32](0.0, 1.0, 5)
  val i:  Tensor[Float32] = Tensor.eye[Float32](3)
  val ek: Tensor[Float32] = Tensor.eye[Float32](3, 4, 1)
  val id: Tensor[Float32] = Tensor.identity[Float32](3)
  val a:  Tensor[Int32]   = Tensor.array[Int32]((1, 2, 3, 4))
  val zl: Tensor[Float32] = Tensor.zerosLike(z)
  val ol: Tensor[Float32] = Tensor.onesLike(o)
  val ad: Tensor[Float32] = Tensor.aranged[Float32](0.0, 1.0, 0.25)
  println(s"zeros        = $z")
  println(s"ones         = $o")
  println(s"full(7)      = $f")
  println(s"arange.resh  = $r")
  println(s"linspace     = $l")
  println(s"eye(3)       = $i")
  println(s"eye(3,4,k=1) = $ek")
  println(s"identity(3)  = $id")
  println(s"array(1..4)  = $a")
  println(s"zerosLike    = $zl")
  println(s"onesLike     = $ol")
  println(s"aranged 1/4  = $ad")

  // ====================================================================
  // 2. Native body — val properties (ndim, size, shape, T) and def
  //    methods (flatten, swapaxes, moveaxis). These are the new
  //    facade-direct surface — no allocation, no `data` field.
  // ====================================================================
  println("\n=== native body: vals + 0/1-arity defs ===")
  val t3 = Tensor.arange[Int32](24).reshape((2, 3, 4))
  println(s"ndim                  = ${t3.ndim}")
  println(s"size                  = ${t3.size}")
  println(s"shape                 = ${t3.shape}")
  println(s"T.shape (val native)  = ${t3.T.shape}")
  println(s"flatten().shape       = ${t3.flatten().shape}")
  println(s"swapaxes(0,2).shape   = ${t3.swapaxes(0, 2).shape}")
  println(s"moveaxis(0,2).shape   = ${t3.moveaxis(0, 2).shape}")

  // ====================================================================
  // 3. toPy escape hatch — verify the no-op cast really yields the
  //    underlying mx.array Python class.
  // ====================================================================
  println("\n=== toPy escape hatch ===")
  println(s"toPy class    = ${pyBuiltins.`type`(t3.toPy).__name__}")
  println(s"toPy.ndim     = ${t3.toPy.ndim}")
  println(s"toPy is mx.array dispatched: ${t3.toPy.size}")

  // ====================================================================
  // 4. Dtype conversion + scalar extraction (extension methods that
  //    depend on `DataTypeInfo[DT]`).
  // ====================================================================
  println("\n=== astype / toDataType / item ===")
  val ints     = Tensor.array[Int32]((1, 2, 3, 4))
  val asFloat  = ints.astype[Float32]
  val asLong   = ints.toDataType[Int64]
  println(s"ints                  = $ints")
  println(s"astype[Float32]       = $asFloat")
  println(s"toDataType[Int64]     = $asLong")
  val sInt:   Int    = ints.sum().item
  val sDbl:   Double = l.mean().item
  println(s"item (Int)            = $sInt")
  println(s"item (Double)         = $sDbl")

  // ====================================================================
  // 5. Tensor⊕Tensor operators (extension overloads).
  // ====================================================================
  println("\n=== operators (tensor / tensor) ===")
  val v1 = Tensor.array[Float32]((1.0, 2.0, 3.0))
  val v2 = Tensor.array[Float32]((4.0, 5.0, 6.0))
  println(s"v1 + v2  = ${v1 + v2}")
  println(s"v1 - v2  = ${v1 - v2}")
  println(s"v1 * v2  = ${v1 * v2}")
  println(s"v1 / v2  = ${v1 / v2}")
  println(s"v1 ** v2 = ${v1 ** v2}")
  println(s"-v1      = ${-v1}")

  // ====================================================================
  // 6. Tensor⊕scalar operators (second extension block with `using`).
  // ====================================================================
  println("\n=== operators (tensor / scalar) ===")
  val k = Tensor.array[Float32]((1.0, 2.0, 3.0, 4.0))
  println(s"k + 10.0 = ${k + 10.0}")
  println(s"k - 1.0  = ${k - 1.0}")
  println(s"k * 2.0  = ${k * 2.0}")
  println(s"k / 2.0  = ${k / 2.0}")
  println(s"k ** 2.0 = ${k ** 2.0}")
  // Int scalar against Int32 tensor — exercises the V<:<ItemType evidence.
  val ki = Tensor.array[Int32]((1, 2, 3, 4))
  println(s"ki + 10  = ${ki + 10}")
  println(s"ki ** 2  = ${ki ** 2}")

  // ====================================================================
  // 7. Comparison operators (extension `===`, `!==`, `<`, `<=`, `>`, `>=`).
  // ====================================================================
  println("\n=== comparison operators ===")
  val c1 = Tensor.array[Int32]((1, 2, 3, 4))
  val c2 = Tensor.array[Int32]((2, 2, 2, 4))
  println(s"c1 === c2 = ${c1 === c2}")
  println(s"c1 !== c2 = ${c1 !== c2}")
  println(s"c1 <  c2  = ${c1 <  c2}")
  println(s"c1 <= c2  = ${c1 <= c2}")
  println(s"c1 >  c2  = ${c1 >  c2}")
  println(s"c1 >= c2  = ${c1 >= c2}")

  // ====================================================================
  // 8. Elementwise unary (free functions in companion).
  // ====================================================================
  println("\n=== elementwise unary (companion) ===")
  val u = Tensor.array[Float32]((0.0, 0.5, 1.0, 2.0))
  println(s"exp        = ${Tensor.exp(u)}")
  println(s"expm1      = ${Tensor.expm1(u)}")
  println(s"log(u+1)   = ${Tensor.log(u + 1.0)}")
  println(s"log2(u+1)  = ${Tensor.log2(u + 1.0)}")
  println(s"log10(u+1) = ${Tensor.log10(u + 1.0)}")
  println(s"log1p      = ${Tensor.log1p(u)}")
  println(s"sqrt       = ${Tensor.sqrt(u)}")
  println(s"rsqrt(u+1) = ${Tensor.rsqrt(u + 1.0)}")
  println(s"square     = ${Tensor.square(u)}")
  println(s"recip(u+1) = ${Tensor.reciprocal(u + 1.0)}")
  println(s"abs(-u)    = ${Tensor.abs(-u)}")
  println(s"sign(-u)   = ${Tensor.sign(-u)}")
  println(s"negative   = ${Tensor.negative(u)}")
  println(s"ceil(u+.3) = ${Tensor.ceil(u + 0.3)}")
  println(s"floor(u+.7)= ${Tensor.floor(u + 0.7)}")
  println(s"round(u+.5)= ${Tensor.round(u + 0.5)}")
  println(s"sin        = ${Tensor.sin(u)}")
  println(s"cos        = ${Tensor.cos(u)}")
  println(s"tan        = ${Tensor.tan(u)}")
  println(s"sinh       = ${Tensor.sinh(u)}")
  println(s"cosh       = ${Tensor.cosh(u)}")
  println(s"tanh       = ${Tensor.tanh(u)}")
  println(s"arcsin(u/2)= ${Tensor.arcsin(u / 2.0)}")
  println(s"arccos(u/2)= ${Tensor.arccos(u / 2.0)}")
  println(s"arctan     = ${Tensor.arctan(u)}")
  println(s"sigmoid    = ${Tensor.sigmoid(u)}")
  println(s"erf        = ${Tensor.erf(u)}")
  println(s"erfinv(u/3)= ${Tensor.erfinv(u / 3.0)}")

  // ====================================================================
  // 9. softmax (1- and 2-arg overload).
  // ====================================================================
  println("\n=== softmax (overload) ===")
  val sm = Tensor.array[Float32]((1.0, 2.0, 3.0))
  println(s"softmax(sm)    = ${Tensor.softmax(sm)}")
  println(s"softmax(sm, 0) = ${Tensor.softmax(sm, 0)}")

  // ====================================================================
  // 10. isnan / isinf / isfinite — needs synthetic ±inf / nan values
  //     (produced by dividing through Tensor.array).
  // ====================================================================
  println("\n=== isnan / isinf / isfinite ===")
  val one = Tensor.array[Float32](Tuple1(1.0))
  val zer = Tensor.array[Float32](Tuple1(0.0))
  val inf = one / zer       // → array([inf])
  val nan = zer / zer       // → array([nan])
  println(s"isnan(0/0)    = ${Tensor.isnan(nan)}")
  println(s"isinf(1/0)    = ${Tensor.isinf(inf)}")
  println(s"isfinite(u)   = ${Tensor.isfinite(u)}")

  // ====================================================================
  // 11. Logical ops.
  // ====================================================================
  println("\n=== logical ops ===")
  val ba = Tensor.array[Bool]((true,  true,  false, false))
  val bb = Tensor.array[Bool]((true,  false, true,  false))
  println(s"and(ba, bb)  = ${Tensor.logicalAnd(ba, bb)}")
  println(s"or (ba, bb)  = ${Tensor.logicalOr (ba, bb)}")
  println(s"not(ba)      = ${Tensor.logicalNot(ba)}")

  // ====================================================================
  // 12. Instance reductions — all three arities for sum, both for the
  //     rest. Also argmax / argmin.
  // ====================================================================
  println("\n=== reductions: instance ===")
  val mat = Tensor.arange[Int32](6).reshape((2, 3))
  println(s"mat:                     $mat")
  println(s"sum()                  = ${mat.sum()}")
  println(s"sum(0)                 = ${mat.sum(0)}")
  println(s"sum(1, keepdims=true)  = ${mat.sum(1, true)}")
  println(s"mean()  (cast f32)     = ${mat.astype[Float32].mean()}")
  println(s"mean(0) (cast f32)     = ${mat.astype[Float32].mean(0)}")
  println(s"max()                  = ${mat.max()}")
  println(s"max(1)                 = ${mat.max(1)}")
  println(s"min()                  = ${mat.min()}")
  println(s"min(0)                 = ${mat.min(0)}")
  println(s"(mat+1).prod()         = ${(mat + 1).prod()}")
  println(s"(mat+1).prod(0)        = ${(mat + 1).prod(0)}")
  println(s"argmax()               = ${mat.argmax()}")
  println(s"argmax(1)              = ${mat.argmax(1)}")
  println(s"argmin()               = ${mat.argmin()}")
  println(s"argmin(0)              = ${mat.argmin(0)}")

  // ====================================================================
  // 13. Free-function reductions.
  // ====================================================================
  println("\n=== reductions: companion ===")
  println(s"Tensor.sum (single arg)    = ${Tensor.sum(mat)}")
  println(s"Tensor.sum  (axis, keep)   = ${Tensor.sum(mat, 0, true)}")
  println(s"Tensor.mean(cast)          = ${Tensor.mean(mat.astype[Float32])}")
  println(s"Tensor.std(linspace)       = ${Tensor.std(l)}")
  println(s"Tensor.all(true, true)     = ${Tensor.all(Tensor.array[Bool]((true, true)))}")
  println(s"Tensor.any(false, true)    = ${Tensor.any(Tensor.array[Bool]((false, true)))}")
  println(s"Tensor.argmax(mat, 1)      = ${Tensor.argmax(mat, 1)}")
  println(s"Tensor.argmin(mat)         = ${Tensor.argmin(mat)}")

  // ====================================================================
  // 14. Free-function shape ops (parallel to instance overloads).
  // ====================================================================
  println("\n=== shape ops: companion ===")
  val m3 = Tensor.arange[Int32](24).reshape((2, 3, 4))
  println(s"reshape((4,6))          .shape = ${Tensor.reshape(m3, (4, 6)).shape}")
  println(s"transpose()             .shape = ${Tensor.transpose(m3).shape}")
  println(s"transpose((2,0,1))      .shape = ${Tensor.transpose(m3, (2, 0, 1)).shape}")
  println(s"expandDims(m3, 1)       .shape = ${Tensor.expandDims(m3, 1).shape}")
  println(s"squeeze(expanded)       .shape = ${Tensor.squeeze(Tensor.expandDims(m3, 1)).shape}")
  println(s"squeeze(expanded, 1)    .shape = ${Tensor.squeeze(Tensor.expandDims(m3, 1), 1).shape}")
  println(s"flatten(m3)             .shape = ${Tensor.flatten(m3).shape}")
  println(s"broadcastTo(a, (3,4))   .shape = ${Tensor.broadcastTo(a, (3, 4)).shape}")
  println(s"swapaxes(m3, 0, 2)      .shape = ${Tensor.swapaxes(m3, 0, 2).shape}")
  println(s"moveaxis(m3, 0, 2)      .shape = ${Tensor.moveaxis(m3, 0, 2).shape}")

  // ====================================================================
  // 15. Combining — all overload arities for concatenate / stack;
  //     tile and repeat.
  // ====================================================================
  println("\n=== combining ===")
  val rr = Tensor.arange[Int32](6).reshape((2, 3))
  println(s"concat(rr, rr).shape          = ${Tensor.concatenate(rr, rr, 0).shape}")
  println(s"concat(rr, rr, rr).shape      = ${Tensor.concatenate(rr, rr, rr, 0).shape}")
  println(s"concat(rr, rr, rr, rr).shape  = ${Tensor.concatenate(rr, rr, rr, rr, 0).shape}")
  val sv = Tensor.array[Int32]((1, 2, 3))
  println(s"stack(sv, sv, 0).shape        = ${Tensor.stack(sv, sv, 0).shape}")
  println(s"stack(sv, sv, sv, 0).shape    = ${Tensor.stack(sv, sv, sv, 0).shape}")
  println(s"stack(sv x4, 0).shape         = ${Tensor.stack(sv, sv, sv, sv, 0).shape}")
  println(s"tile(sv, (2, 3)).shape        = ${Tensor.tile(sv, (2, 3)).shape}")
  println(s"repeat(sv, 3, 0).shape        = ${Tensor.repeat(sv, 3, 0).shape}")

  // ====================================================================
  // 16. Linear algebra.
  // ====================================================================
  println("\n=== linear algebra ===")
  val A = Tensor.arange[Float32](6).reshape((2, 3))
  val B = Tensor.arange[Float32](6).reshape((3, 2))
  println(s"A @@ B            = ${A @@ B}")
  println(s"matmul(A, B)      = ${Tensor.matmul(A, B)}")
  println(s"inner(v1, v2)     = ${Tensor.inner(v1, v2)}")
  println(s"outer(v1, v2)     = ${Tensor.outer(v1, v2)}")
  println(s"tensordot(A,B,1)  = ${Tensor.tensordot(A, B, 1)}")

  // ====================================================================
  // 17. Selection — where, clip (tensor and scalar overloads).
  // ====================================================================
  println("\n=== selection ===")
  val maskFull = Tensor.full[Int32, (Int, Int), Int]((2, 3), 2)
  val mask = r > maskFull
  println(s"where(mask, r, -r)    = ${Tensor.where(mask, r, -r)}")
  val lo = Tensor.full[Int32, (Int, Int), Int]((2, 3), 1)
  val hi = Tensor.full[Int32, (Int, Int), Int]((2, 3), 4)
  println(s"clip(r, lo, hi)       = ${Tensor.clip(r, lo, hi)}")
  println(s"clip(r, 1, 4) scalar  = ${Tensor.clip(r, 1, 4)}")

  // ====================================================================
  // 18. eval — force materialisation. After eval, accessing the array
  //     should give the computed values (no error).
  // ====================================================================
  println("\n=== eval ===")
  val deferred = Tensor.arange[Int32](6).reshape((2, 3)) * 2 + 1
  Tensor.eval(deferred)
  println(s"after eval: $deferred")

  // ====================================================================
  // 19. Dtype propagation sanity — verify various ops keep the static
  //     type as expected (no wrapper, no `data` field — direct view).
  // ====================================================================
  println("\n=== dtype propagation ===")
  val fi: Tensor[Int32]   = mat.flatten()
  val ti: Tensor[Int32]   = mat.T
  val si: Tensor[Int32]   = mat.swapaxes(0, 1)
  val mi: Tensor[Int32]   = mat.moveaxis(0, 1)
  val cmp: Tensor[Bool]   = mat === mat
  val am: Tensor[UInt32]  = mat.argmax(0)
  println(s"flatten dtype=${fi.toPy.dtype} T dtype=${ti.toPy.dtype}")
  println(s"swap dtype=${si.toPy.dtype}    move dtype=${mi.toPy.dtype}")
  println(s"=== cmp dtype=${cmp.toPy.dtype}  argmax dtype=${am.toPy.dtype}")

  println("\nall Tensor facade members exercised.")
