package mlx.core

import scala.python.*
import mlx.utils.{pyBuiltins, IsSizeType}

@extern("mlx", "core")
object _mx extends PyDynamic

// ---------------------------------------------------------------------------
// Tensor — direct facade for `mlx.core.array`.
//
// At runtime a `Tensor[DT]` IS the Python `mlx.core.array` instance. The
// type parameter is erased — purely a compile-time hand-rail. The class
// body carries only `native` declarations (Python attributes + single-
// signature methods). Every overload, every implicit-using method, and
// every operator lives in the extension block, where it inlines to a
// direct `Dynamic` dispatch. Free functions / factories sit in the
// companion.
// ---------------------------------------------------------------------------

@extern("mlx.core", "array")
class Tensor[DT <: DataType]:

  // --- Python @properties / attributes --------------------------------
  val ndim: Int = native
  val size: Int = native
  /** Shape as a Python tuple (use `ndim`, `size` for primitives). */
  val shape: PyDynamic = native
  /** Default transpose — reverse all axes. (Python `@property`). */
  val T: Tensor[DT] = native

  // --- single-signature instance methods ------------------------------
  def flatten(): Tensor[DT] = native
  def swapaxes(a: Int, b: Int): Tensor[DT] = native
  def moveaxis(src: Int, dst: Int): Tensor[DT] = native

// ---------------------------------------------------------------------------
// Instance surface — overloads, implicit-requiring methods, operators.
//
// Every member here is `inline`, so `t.toPy.foo(...)` collapses to a single
// `Dynamic` dispatch at the call site — no wrapper allocation, no static-
// method indirection. `asInstanceOf[Tensor[DT]]` is a no-op cast: the
// returned Python `mlx.core.array` is the same object, just retyped.
// ---------------------------------------------------------------------------

extension [DT <: DataType](t: Tensor[DT])

  /** Lift `t` back to a raw `PyDynamic` for un-typed Python calls. */
  inline def toPy: PyDynamic = t.asInstanceOf[PyDynamic]

  /** Scalar value (rank-0 only). Result type derives from the dtype. */
  inline def item(using ti: DataTypeInfo[DT]): ti.ItemType =
    t.toPy.item().asInstanceOf[ti.ItemType]

  inline def toDataType[U <: DataType](using ti: DataTypeInfo[U]): Tensor[U] =
    t.toPy.astype(ti.dtype).asInstanceOf[Tensor[U]]
  inline def astype[U <: DataType](using DataTypeInfo[U]): Tensor[U] =
    toDataType[U]

  // --- shape transformations ------------------------------------------
  inline def reshape[ST: IsSizeType](shape: ST): Tensor[DT] =
    t.toPy.reshape(shape).asInstanceOf[Tensor[DT]]
  inline def transpose(): Tensor[DT] = t.toPy.transpose().asInstanceOf[Tensor[DT]]
  inline def transpose[ST: IsSizeType](axes: ST): Tensor[DT] =
    _mx.transpose(t.toPy, axes).asInstanceOf[Tensor[DT]]
  inline def squeeze(): Tensor[DT] = t.toPy.squeeze().asInstanceOf[Tensor[DT]]
  inline def squeeze(axis: Int): Tensor[DT] = t.toPy.squeeze(axis).asInstanceOf[Tensor[DT]]

  // --- reductions (instance) ------------------------------------------
  inline def sum(): Tensor[DT] = t.toPy.sum().asInstanceOf[Tensor[DT]]
  inline def sum(axis: Int): Tensor[DT] = t.toPy.sum(axis).asInstanceOf[Tensor[DT]]
  inline def sum(axis: Int, keepdims: Boolean): Tensor[DT] =
    t.toPy.sum(axis, keepdims).asInstanceOf[Tensor[DT]]
  inline def mean(): Tensor[DT] = t.toPy.mean().asInstanceOf[Tensor[DT]]
  inline def mean(axis: Int): Tensor[DT] = t.toPy.mean(axis).asInstanceOf[Tensor[DT]]
  inline def max(): Tensor[DT] = t.toPy.max().asInstanceOf[Tensor[DT]]
  inline def max(axis: Int): Tensor[DT] = t.toPy.max(axis).asInstanceOf[Tensor[DT]]
  inline def min(): Tensor[DT] = t.toPy.min().asInstanceOf[Tensor[DT]]
  inline def min(axis: Int): Tensor[DT] = t.toPy.min(axis).asInstanceOf[Tensor[DT]]
  inline def prod(): Tensor[DT] = t.toPy.prod().asInstanceOf[Tensor[DT]]
  inline def prod(axis: Int): Tensor[DT] = t.toPy.prod(axis).asInstanceOf[Tensor[DT]]
  inline def argmax(): Tensor[UInt32] = t.toPy.argmax().asInstanceOf[Tensor[UInt32]]
  inline def argmax(axis: Int): Tensor[UInt32] = t.toPy.argmax(axis).asInstanceOf[Tensor[UInt32]]
  inline def argmin(): Tensor[UInt32] = t.toPy.argmin().asInstanceOf[Tensor[UInt32]]
  inline def argmin(axis: Int): Tensor[UInt32] = t.toPy.argmin(axis).asInstanceOf[Tensor[UInt32]]

  // --- operator syntax (tensor / tensor) ------------------------------
  inline def + (that: Tensor[DT]): Tensor[DT] = Tensor.add(t, that)
  inline def - (that: Tensor[DT]): Tensor[DT] = Tensor.subtract(t, that)
  inline def * (that: Tensor[DT]): Tensor[DT] = Tensor.multiply(t, that)
  inline def / (that: Tensor[DT]): Tensor[DT] = Tensor.divide(t, that)
  inline def ** (that: Tensor[DT]): Tensor[DT] = Tensor.power(t, that)
  inline def unary_- : Tensor[DT] = Tensor.negative(t)

  inline def === (that: Tensor[DT]): Tensor[Bool] = Tensor.equal(t, that)
  inline def !== (that: Tensor[DT]): Tensor[Bool] = Tensor.notEqual(t, that)
  inline def <   (that: Tensor[DT]): Tensor[Bool] = Tensor.less(t, that)
  inline def <=  (that: Tensor[DT]): Tensor[Bool] = Tensor.lessEqual(t, that)
  inline def >   (that: Tensor[DT]): Tensor[Bool] = Tensor.greater(t, that)
  inline def >=  (that: Tensor[DT]): Tensor[Bool] = Tensor.greaterEqual(t, that)

  /** Matrix multiplication, infix form. */
  inline def @@ (that: Tensor[DT]): Tensor[DT] = Tensor.matmul(t, that)

// --- operator syntax (tensor / scalar) --------------------------------

extension [DT <: DataType, V](t: Tensor[DT])
  inline def + (v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] = Tensor.add(t, v)
  inline def - (v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] = Tensor.subtract(t, v)
  inline def * (v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] = Tensor.multiply(t, v)
  inline def / (v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] = Tensor.divide(t, v)
  inline def ** (v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] = Tensor.power(t, v)

// ---------------------------------------------------------------------------
// Companion — factories + free functions. Every helper inlines to a single
// `_mx.foo(...)` dispatch with an `asInstanceOf` re-typing (no-op at runtime).
// ---------------------------------------------------------------------------

object Tensor:

  /** Zero-cost cast: retype a `PyDynamic` from `_mx.*` as a `Tensor[DT]`.
   *  Used internally by every factory and free-function below. */
  private[core] inline def fromPy[DT <: DataType](py: PyDynamic): Tensor[DT] =
    py.asInstanceOf[Tensor[DT]]

  // --- construction ----------------------------------------------------

  inline def zeros[ST: IsSizeType](shape: ST)[DT <: DataType](using ti: DataTypeInfo[DT]): Tensor[DT] =
    fromPy(_mx.zeros(shape, ti.dtype))

  inline def ones[ST: IsSizeType](shape: ST)[DT <: DataType](using ti: DataTypeInfo[DT]): Tensor[DT] =
    fromPy(_mx.ones(shape, ti.dtype))

  inline def full[DT <: DataType, ST, V](shape: ST, value: V)(using
    st: IsSizeType[ST], ti: DataTypeInfo[DT], ev: V <:< ti.ItemType
  ): Tensor[DT] =
    fromPy(_mx.full(shape, value, ti.dtype))

  inline def zerosLike[DT <: DataType](xs: Tensor[DT]): Tensor[DT] =
    fromPy(_mx.zeros_like(xs.toPy))
  inline def onesLike[DT <: DataType](xs: Tensor[DT]): Tensor[DT] =
    fromPy(_mx.ones_like(xs.toPy))

  inline def eye[DT <: DataType](n: Int)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    fromPy(_mx.eye(n, n, 0, ti.dtype))
  inline def eye[DT <: DataType](n: Int, m: Int, k: Int)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    fromPy(_mx.eye(n, m, k, ti.dtype))
  inline def identity[DT <: DataType](n: Int)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    fromPy(_mx.identity(n, ti.dtype))

  inline def arange[DT <: DataType](start: Int, stop: Int, step: Int)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    fromPy(_mx.arange(start, stop, step, ti.dtype))
  inline def arange[DT <: DataType](stop: Int)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    fromPy(_mx.arange(0, stop, 1, ti.dtype))
  inline def aranged[DT <: DataType](start: Double, stop: Double, step: Double)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    fromPy(_mx.arange(start, stop, step, ti.dtype))

  inline def linspace[DT <: DataType](start: Double, stop: Double, num: Int)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    fromPy(_mx.linspace(start, stop, num, ti.dtype))

  /** Build a tensor from a Scala tuple literal (1-D). Scala tuples bridge to
   *  Python tuples, which `mlx.core.array` accepts as sequence input. */
  inline def array[DT <: DataType](values: Tuple)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    fromPy(_mx.array(values, ti.dtype))

  // --- elementwise unary -----------------------------------------------

  inline def abs[DT <: DataType](x: Tensor[DT]): Tensor[DT]        = fromPy(_mx.abs(x.toPy))
  inline def negative[DT <: DataType](x: Tensor[DT]): Tensor[DT]   = fromPy(_mx.negative(x.toPy))
  inline def sign[DT <: DataType](x: Tensor[DT]): Tensor[DT]       = fromPy(_mx.sign(x.toPy))
  inline def exp[DT <: DataType](x: Tensor[DT]): Tensor[DT]        = fromPy(_mx.exp(x.toPy))
  inline def expm1[DT <: DataType](x: Tensor[DT]): Tensor[DT]      = fromPy(_mx.expm1(x.toPy))
  inline def log[DT <: DataType](x: Tensor[DT]): Tensor[DT]        = fromPy(_mx.log(x.toPy))
  inline def log2[DT <: DataType](x: Tensor[DT]): Tensor[DT]       = fromPy(_mx.log2(x.toPy))
  inline def log10[DT <: DataType](x: Tensor[DT]): Tensor[DT]      = fromPy(_mx.log10(x.toPy))
  inline def log1p[DT <: DataType](x: Tensor[DT]): Tensor[DT]      = fromPy(_mx.log1p(x.toPy))
  inline def sqrt[DT <: DataType](x: Tensor[DT]): Tensor[DT]       = fromPy(_mx.sqrt(x.toPy))
  inline def rsqrt[DT <: DataType](x: Tensor[DT]): Tensor[DT]      = fromPy(_mx.rsqrt(x.toPy))
  inline def square[DT <: DataType](x: Tensor[DT]): Tensor[DT]     = fromPy(_mx.square(x.toPy))
  inline def reciprocal[DT <: DataType](x: Tensor[DT]): Tensor[DT] = fromPy(_mx.reciprocal(x.toPy))
  inline def ceil[DT <: DataType](x: Tensor[DT]): Tensor[DT]       = fromPy(_mx.ceil(x.toPy))
  inline def floor[DT <: DataType](x: Tensor[DT]): Tensor[DT]      = fromPy(_mx.floor(x.toPy))
  inline def round[DT <: DataType](x: Tensor[DT]): Tensor[DT]      = fromPy(_mx.round(x.toPy))
  inline def sin[DT <: DataType](x: Tensor[DT]): Tensor[DT]        = fromPy(_mx.sin(x.toPy))
  inline def cos[DT <: DataType](x: Tensor[DT]): Tensor[DT]        = fromPy(_mx.cos(x.toPy))
  inline def tan[DT <: DataType](x: Tensor[DT]): Tensor[DT]        = fromPy(_mx.tan(x.toPy))
  inline def sinh[DT <: DataType](x: Tensor[DT]): Tensor[DT]       = fromPy(_mx.sinh(x.toPy))
  inline def cosh[DT <: DataType](x: Tensor[DT]): Tensor[DT]       = fromPy(_mx.cosh(x.toPy))
  inline def tanh[DT <: DataType](x: Tensor[DT]): Tensor[DT]       = fromPy(_mx.tanh(x.toPy))
  inline def arcsin[DT <: DataType](x: Tensor[DT]): Tensor[DT]     = fromPy(_mx.arcsin(x.toPy))
  inline def arccos[DT <: DataType](x: Tensor[DT]): Tensor[DT]     = fromPy(_mx.arccos(x.toPy))
  inline def arctan[DT <: DataType](x: Tensor[DT]): Tensor[DT]     = fromPy(_mx.arctan(x.toPy))
  inline def sigmoid[DT <: DataType](x: Tensor[DT]): Tensor[DT]    = fromPy(_mx.sigmoid(x.toPy))
  inline def erf[DT <: DataType](x: Tensor[DT]): Tensor[DT]        = fromPy(_mx.erf(x.toPy))
  inline def erfinv[DT <: DataType](x: Tensor[DT]): Tensor[DT]     = fromPy(_mx.erfinv(x.toPy))
  inline def softmax[DT <: DataType](x: Tensor[DT]): Tensor[DT]    = fromPy(_mx.softmax(x.toPy))
  inline def softmax[DT <: DataType](x: Tensor[DT], axis: Int): Tensor[DT] =
    fromPy(_mx.softmax(x.toPy, axis))
  inline def logicalNot(x: Tensor[Bool]): Tensor[Bool] = fromPy(_mx.logical_not(x.toPy))

  inline def isnan[DT <: DataType](x: Tensor[DT]): Tensor[Bool]    = fromPy(_mx.isnan(x.toPy))
  inline def isinf[DT <: DataType](x: Tensor[DT]): Tensor[Bool]    = fromPy(_mx.isinf(x.toPy))
  inline def isfinite[DT <: DataType](x: Tensor[DT]): Tensor[Bool] = fromPy(_mx.isfinite(x.toPy))

  // --- elementwise binary (T,T) and (T,scalar) -------------------------

  inline def add[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = fromPy(_mx.add(a.toPy, b.toPy))
  inline def add[DT <: DataType, V](a: Tensor[DT], v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] =
    fromPy(_mx.add(a.toPy, v))
  inline def subtract[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = fromPy(_mx.subtract(a.toPy, b.toPy))
  inline def subtract[DT <: DataType, V](a: Tensor[DT], v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] =
    fromPy(_mx.subtract(a.toPy, v))
  inline def multiply[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = fromPy(_mx.multiply(a.toPy, b.toPy))
  inline def multiply[DT <: DataType, V](a: Tensor[DT], v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] =
    fromPy(_mx.multiply(a.toPy, v))
  inline def divide[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = fromPy(_mx.divide(a.toPy, b.toPy))
  inline def divide[DT <: DataType, V](a: Tensor[DT], v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] =
    fromPy(_mx.divide(a.toPy, v))
  inline def power[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = fromPy(_mx.power(a.toPy, b.toPy))
  inline def power[DT <: DataType, V](a: Tensor[DT], v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] =
    fromPy(_mx.power(a.toPy, v))
  inline def maximum[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = fromPy(_mx.maximum(a.toPy, b.toPy))
  inline def minimum[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = fromPy(_mx.minimum(a.toPy, b.toPy))
  inline def floorDivide[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = fromPy(_mx.floor_divide(a.toPy, b.toPy))
  inline def remainder[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT]   = fromPy(_mx.remainder(a.toPy, b.toPy))

  inline def logicalAnd(a: Tensor[Bool], b: Tensor[Bool]): Tensor[Bool] = fromPy(_mx.logical_and(a.toPy, b.toPy))
  inline def logicalOr (a: Tensor[Bool], b: Tensor[Bool]): Tensor[Bool] = fromPy(_mx.logical_or (a.toPy, b.toPy))

  // --- comparisons (return Bool) ---------------------------------------

  inline def equal[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[Bool]    = fromPy(_mx.equal(a.toPy, b.toPy))
  inline def notEqual[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[Bool] = fromPy(_mx.not_equal(a.toPy, b.toPy))
  inline def less[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[Bool]     = fromPy(_mx.less(a.toPy, b.toPy))
  inline def lessEqual[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[Bool] = fromPy(_mx.less_equal(a.toPy, b.toPy))
  inline def greater[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[Bool]   = fromPy(_mx.greater(a.toPy, b.toPy))
  inline def greaterEqual[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[Bool] = fromPy(_mx.greater_equal(a.toPy, b.toPy))

  // --- reductions ------------------------------------------------------

  inline def sum[DT <: DataType](x: Tensor[DT]): Tensor[DT] = fromPy(_mx.sum(x.toPy))
  inline def sum[DT <: DataType, ST: IsSizeType](x: Tensor[DT], axis: ST, keepdims: Boolean): Tensor[DT] =
    fromPy(_mx.sum(x.toPy, axis, keepdims))
  inline def mean[DT <: DataType](x: Tensor[DT]): Tensor[DT] = fromPy(_mx.mean(x.toPy))
  inline def mean[DT <: DataType, ST: IsSizeType](x: Tensor[DT], axis: ST, keepdims: Boolean): Tensor[DT] =
    fromPy(_mx.mean(x.toPy, axis, keepdims))
  inline def max[DT <: DataType](x: Tensor[DT]): Tensor[DT] = fromPy(_mx.max(x.toPy))
  inline def max[DT <: DataType, ST: IsSizeType](x: Tensor[DT], axis: ST, keepdims: Boolean): Tensor[DT] =
    fromPy(_mx.max(x.toPy, axis, keepdims))
  inline def min[DT <: DataType](x: Tensor[DT]): Tensor[DT] = fromPy(_mx.min(x.toPy))
  inline def min[DT <: DataType, ST: IsSizeType](x: Tensor[DT], axis: ST, keepdims: Boolean): Tensor[DT] =
    fromPy(_mx.min(x.toPy, axis, keepdims))
  inline def prod[DT <: DataType](x: Tensor[DT]): Tensor[DT] = fromPy(_mx.prod(x.toPy))
  inline def prod[DT <: DataType, ST: IsSizeType](x: Tensor[DT], axis: ST, keepdims: Boolean): Tensor[DT] =
    fromPy(_mx.prod(x.toPy, axis, keepdims))
  inline def std[DT <: DataType](x: Tensor[DT]): Tensor[DT] = fromPy(_mx.std(x.toPy))

  inline def all(x: Tensor[Bool]): Tensor[Bool] = fromPy(_mx.all(x.toPy))
  inline def any(x: Tensor[Bool]): Tensor[Bool] = fromPy(_mx.any(x.toPy))

  inline def argmax[DT <: DataType](x: Tensor[DT]): Tensor[UInt32]            = fromPy(_mx.argmax(x.toPy))
  inline def argmax[DT <: DataType](x: Tensor[DT], axis: Int): Tensor[UInt32] = fromPy(_mx.argmax(x.toPy, axis))
  inline def argmin[DT <: DataType](x: Tensor[DT]): Tensor[UInt32]            = fromPy(_mx.argmin(x.toPy))
  inline def argmin[DT <: DataType](x: Tensor[DT], axis: Int): Tensor[UInt32] = fromPy(_mx.argmin(x.toPy, axis))

  // --- shape transformations ------------------------------------------

  inline def reshape[DT <: DataType, ST: IsSizeType](x: Tensor[DT], shape: ST): Tensor[DT] =
    fromPy(_mx.reshape(x.toPy, shape))
  inline def transpose[DT <: DataType](x: Tensor[DT]): Tensor[DT] = fromPy(_mx.transpose(x.toPy))
  inline def transpose[DT <: DataType, ST: IsSizeType](x: Tensor[DT], axes: ST): Tensor[DT] =
    fromPy(_mx.transpose(x.toPy, axes))
  inline def swapaxes[DT <: DataType](x: Tensor[DT], a: Int, b: Int): Tensor[DT] =
    fromPy(_mx.swapaxes(x.toPy, a, b))
  inline def moveaxis[DT <: DataType](x: Tensor[DT], src: Int, dst: Int): Tensor[DT] =
    fromPy(_mx.moveaxis(x.toPy, src, dst))
  inline def expandDims[DT <: DataType](x: Tensor[DT], axis: Int): Tensor[DT] =
    fromPy(_mx.expand_dims(x.toPy, axis))
  inline def squeeze[DT <: DataType](x: Tensor[DT]): Tensor[DT] = fromPy(_mx.squeeze(x.toPy))
  inline def squeeze[DT <: DataType](x: Tensor[DT], axis: Int): Tensor[DT] = fromPy(_mx.squeeze(x.toPy, axis))
  inline def flatten[DT <: DataType](x: Tensor[DT]): Tensor[DT] = fromPy(_mx.flatten(x.toPy))
  inline def broadcastTo[DT <: DataType, ST: IsSizeType](x: Tensor[DT], shape: ST): Tensor[DT] =
    fromPy(_mx.broadcast_to(x.toPy, shape))

  // --- combining ------------------------------------------------------
  // MLX's `concatenate` / `stack` expect a python `list[array]`. Build one
  // via `builtins.list` from a Scala tuple of array payloads.

  inline def concatenate[DT <: DataType](a: Tensor[DT], b: Tensor[DT], axis: Int): Tensor[DT] =
    fromPy(_mx.concatenate(pyBuiltins.list((a.toPy, b.toPy)), axis))
  inline def concatenate[DT <: DataType](a: Tensor[DT], b: Tensor[DT], c: Tensor[DT], axis: Int): Tensor[DT] =
    fromPy(_mx.concatenate(pyBuiltins.list((a.toPy, b.toPy, c.toPy)), axis))
  inline def concatenate[DT <: DataType](a: Tensor[DT], b: Tensor[DT], c: Tensor[DT], d: Tensor[DT], axis: Int): Tensor[DT] =
    fromPy(_mx.concatenate(pyBuiltins.list((a.toPy, b.toPy, c.toPy, d.toPy)), axis))

  inline def stack[DT <: DataType](a: Tensor[DT], b: Tensor[DT], axis: Int): Tensor[DT] =
    fromPy(_mx.stack(pyBuiltins.list((a.toPy, b.toPy)), axis))
  inline def stack[DT <: DataType](a: Tensor[DT], b: Tensor[DT], c: Tensor[DT], axis: Int): Tensor[DT] =
    fromPy(_mx.stack(pyBuiltins.list((a.toPy, b.toPy, c.toPy)), axis))
  inline def stack[DT <: DataType](a: Tensor[DT], b: Tensor[DT], c: Tensor[DT], d: Tensor[DT], axis: Int): Tensor[DT] =
    fromPy(_mx.stack(pyBuiltins.list((a.toPy, b.toPy, c.toPy, d.toPy)), axis))

  inline def tile[DT <: DataType, ST: IsSizeType](x: Tensor[DT], reps: ST): Tensor[DT] =
    fromPy(_mx.tile(x.toPy, reps))
  inline def repeat[DT <: DataType](x: Tensor[DT], repeats: Int, axis: Int): Tensor[DT] =
    fromPy(_mx.repeat(x.toPy, repeats, axis))

  // --- linear algebra --------------------------------------------------

  inline def matmul[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] =
    fromPy(_mx.matmul(a.toPy, b.toPy))
  inline def inner[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] =
    fromPy(_mx.inner(a.toPy, b.toPy))
  inline def outer[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] =
    fromPy(_mx.outer(a.toPy, b.toPy))
  inline def tensordot[DT <: DataType](a: Tensor[DT], b: Tensor[DT], axes: Int): Tensor[DT] =
    fromPy(_mx.tensordot(a.toPy, b.toPy, axes))

  // --- selection -------------------------------------------------------

  inline def where[DT <: DataType](cond: Tensor[Bool], x: Tensor[DT], y: Tensor[DT]): Tensor[DT] =
    fromPy(_mx.where(cond.toPy, x.toPy, y.toPy))
  inline def clip[DT <: DataType](x: Tensor[DT], lo: Tensor[DT], hi: Tensor[DT]): Tensor[DT] =
    fromPy(_mx.clip(x.toPy, lo.toPy, hi.toPy))
  inline def clip[DT <: DataType, V](x: Tensor[DT], lo: V, hi: V)(using
    ti: DataTypeInfo[DT], ev: V <:< ti.ItemType
  ): Tensor[DT] =
    fromPy(_mx.clip(x.toPy, lo, hi))

  // --- evaluation ------------------------------------------------------

  inline def eval[DT <: DataType](xs: Tensor[DT]): Unit =
    _mx.eval(xs.toPy)
    ()
