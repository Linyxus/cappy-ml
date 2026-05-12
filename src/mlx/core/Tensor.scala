package mlx.core

import scala.python.*
import mlx.utils.pyBuiltins

@extern("mlx", "core")
object _mx extends PyDynamic

// ---------------------------------------------------------------------------
// Dtypes
// ---------------------------------------------------------------------------

sealed trait DataType
final class Bool      extends DataType
final class Int8      extends DataType
final class Int16     extends DataType
final class Int32     extends DataType
final class Int64     extends DataType
final class UInt8     extends DataType
final class UInt16    extends DataType
final class UInt32    extends DataType
final class UInt64    extends DataType
final class Float16   extends DataType
final class Float32   extends DataType
final class Float64   extends DataType
final class BFloat16  extends DataType

sealed trait IntegerDT  extends DataType
sealed trait FloatingDT extends DataType

trait DataTypeInfo[DT <: DataType]:
  type ItemType <: Boolean | Int | Long | Double
  val dtype: PyDynamic

object DataTypeInfo:
  type Aux[DT <: DataType, IT] = DataTypeInfo[DT] { type ItemType = IT }

  private def make[DT <: DataType, IT <: Boolean | Int | Long | Double](pyDtype: PyDynamic): Aux[DT, IT] =
    new DataTypeInfo[DT]:
      type ItemType = IT
      val dtype: PyDynamic = pyDtype

  given DataTypeInfo.Aux[Bool,     Boolean] = make(_mx.bool_)
  given DataTypeInfo.Aux[Int8,     Int]     = make(_mx.int8)
  given DataTypeInfo.Aux[Int16,    Int]     = make(_mx.int16)
  given DataTypeInfo.Aux[Int32,    Int]     = make(_mx.int32)
  given DataTypeInfo.Aux[Int64,    Long]    = make(_mx.int64)
  given DataTypeInfo.Aux[UInt8,    Int]     = make(_mx.uint8)
  given DataTypeInfo.Aux[UInt16,   Int]     = make(_mx.uint16)
  given DataTypeInfo.Aux[UInt32,   Long]    = make(_mx.uint32)
  given DataTypeInfo.Aux[UInt64,   Long]    = make(_mx.uint64)
  given DataTypeInfo.Aux[Float16,  Double]  = make(_mx.float16)
  given DataTypeInfo.Aux[Float32,  Double]  = make(_mx.float32)
  given DataTypeInfo.Aux[Float64,  Double]  = make(_mx.float64)
  given DataTypeInfo.Aux[BFloat16, Double]  = make(_mx.bfloat16)

// ---------------------------------------------------------------------------
// Shape / axis type plumbing
// ---------------------------------------------------------------------------

/** Evidence that `T` is a tuple whose every element is a subtype of `E`. */
trait IsTupleOfType[T, E]
object IsTupleOfType:
  given empty[E]: IsTupleOfType[EmptyTuple, E] = new IsTupleOfType[EmptyTuple, E] {}
  given cons[H, T <: Tuple, E](using ev: H <:< E, tail: IsTupleOfType[T, E]): IsTupleOfType[H *: T, E] =
    new IsTupleOfType[H *: T, E] {}

trait IsSizeType[T]
object IsSizeType:
  given IsSizeType[Int] with {}
  given [T <: Tuple](using IsTupleOfType[T, Int]): IsSizeType[T] with {}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

class Tensor[DT <: DataType](private[core] val data: PyDynamic):

  // --- python interop --------------------------------------------------
  def toPy: PyDynamic = data
  override def toString(): String = pyBuiltins.repr(data).asInstanceOf

  // --- scalar / dtype --------------------------------------------------
  def item(using ti: DataTypeInfo[DT]): ti.ItemType =
    data.item().asInstanceOf

  def toDataType[U <: DataType](using ti: DataTypeInfo[U]): Tensor[U] =
    Tensor(data.astype(ti.dtype))
  inline def astype[U <: DataType](using DataTypeInfo[U]): Tensor[U] =
    toDataType[U]

  // --- shape / metadata ------------------------------------------------
  /** Tensor shape as a Python tuple (use `ndim`, `size` for primitives). */
  def shapePy: PyDynamic = data.shape
  def ndim: Int = data.ndim.asInstanceOf
  def size: Int = data.size.asInstanceOf

  // --- shape transformations ------------------------------------------
  inline def reshape[ST: IsSizeType](shape: ST): Tensor[DT] =
    Tensor(data.reshape(shape))
  def flatten(): Tensor[DT] = Tensor(data.flatten())
  def squeeze(): Tensor[DT] = Tensor(data.squeeze())
  def squeeze(axis: Int): Tensor[DT] = Tensor(data.squeeze(axis))
  /** Default transpose: reverse all axes. */
  def transpose(): Tensor[DT] = Tensor(data.transpose())
  inline def transpose[ST: IsSizeType](axes: ST): Tensor[DT] =
    Tensor(_mx.transpose(data, axes))
  def T: Tensor[DT] = transpose()
  def swapaxes(a: Int, b: Int): Tensor[DT] = Tensor(data.swapaxes(a, b))
  def moveaxis(src: Int, dst: Int): Tensor[DT] = Tensor(data.moveaxis(src, dst))

  // --- reductions (instance) ------------------------------------------
  def sum(): Tensor[DT] = Tensor(data.sum())
  def sum(axis: Int): Tensor[DT] = Tensor(data.sum(axis))
  def sum(axis: Int, keepdims: Boolean): Tensor[DT] = Tensor(data.sum(axis, keepdims))
  def mean(): Tensor[DT] = Tensor(data.mean())
  def mean(axis: Int): Tensor[DT] = Tensor(data.mean(axis))
  def max(): Tensor[DT] = Tensor(data.max())
  def max(axis: Int): Tensor[DT] = Tensor(data.max(axis))
  def min(): Tensor[DT] = Tensor(data.min())
  def min(axis: Int): Tensor[DT] = Tensor(data.min(axis))
  def prod(): Tensor[DT] = Tensor(data.prod())
  def prod(axis: Int): Tensor[DT] = Tensor(data.prod(axis))
  def argmax(): Tensor[UInt32] = Tensor(data.argmax())
  def argmax(axis: Int): Tensor[UInt32] = Tensor(data.argmax(axis))
  def argmin(): Tensor[UInt32] = Tensor(data.argmin())
  def argmin(axis: Int): Tensor[UInt32] = Tensor(data.argmin(axis))

object Tensor:

  // --- construction ----------------------------------------------------

  inline def zeros[ST: IsSizeType](shape: ST)[DT <: DataType](using ti: DataTypeInfo[DT]): Tensor[DT] =
    Tensor(_mx.zeros(shape, ti.dtype))

  inline def ones[ST: IsSizeType](shape: ST)[DT <: DataType](using ti: DataTypeInfo[DT]): Tensor[DT] =
    Tensor(_mx.ones(shape, ti.dtype))

  inline def full[DT <: DataType, ST, V](shape: ST, value: V)(using
    st: IsSizeType[ST], ti: DataTypeInfo[DT], ev: V <:< ti.ItemType
  ): Tensor[DT] =
    Tensor(_mx.full(shape, value, ti.dtype))

  inline def zerosLike[DT <: DataType](xs: Tensor[DT]): Tensor[DT] =
    Tensor(_mx.zeros_like(xs.data))
  inline def onesLike[DT <: DataType](xs: Tensor[DT]): Tensor[DT] =
    Tensor(_mx.ones_like(xs.data))

  inline def eye[DT <: DataType](n: Int)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    Tensor(_mx.eye(n, n, 0, ti.dtype))
  inline def eye[DT <: DataType](n: Int, m: Int, k: Int)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    Tensor(_mx.eye(n, m, k, ti.dtype))
  inline def identity[DT <: DataType](n: Int)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    Tensor(_mx.identity(n, ti.dtype))

  inline def arange[DT <: DataType](start: Int, stop: Int, step: Int)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    Tensor(_mx.arange(start, stop, step, ti.dtype))
  inline def arange[DT <: DataType](stop: Int)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    Tensor(_mx.arange(0, stop, 1, ti.dtype))
  inline def aranged[DT <: DataType](start: Double, stop: Double, step: Double)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    Tensor(_mx.arange(start, stop, step, ti.dtype))

  inline def linspace[DT <: DataType](start: Double, stop: Double, num: Int)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    Tensor(_mx.linspace(start, stop, num, ti.dtype))

  /** Build a tensor from a Scala tuple literal (1-D). Scala tuples bridge to
   *  Python tuples, which `mlx.core.array` accepts as sequence input. */
  inline def array[DT <: DataType](values: Tuple)(using ti: DataTypeInfo[DT]): Tensor[DT] =
    Tensor(_mx.array(values, ti.dtype))

  // --- elementwise unary -----------------------------------------------
  // Each wrapper inlines a direct attribute call: the bridge handles
  // `_mx.foo(args)` as a single method-dispatch step, whereas a helper
  // that stored `_mx.foo` as a value would lose dispatch context and
  // try to `.apply` a bare Python callable.

  inline def abs[DT <: DataType](x: Tensor[DT]): Tensor[DT]        = Tensor(_mx.abs(x.data))
  inline def negative[DT <: DataType](x: Tensor[DT]): Tensor[DT]   = Tensor(_mx.negative(x.data))
  inline def sign[DT <: DataType](x: Tensor[DT]): Tensor[DT]       = Tensor(_mx.sign(x.data))
  inline def exp[DT <: DataType](x: Tensor[DT]): Tensor[DT]        = Tensor(_mx.exp(x.data))
  inline def expm1[DT <: DataType](x: Tensor[DT]): Tensor[DT]      = Tensor(_mx.expm1(x.data))
  inline def log[DT <: DataType](x: Tensor[DT]): Tensor[DT]        = Tensor(_mx.log(x.data))
  inline def log2[DT <: DataType](x: Tensor[DT]): Tensor[DT]       = Tensor(_mx.log2(x.data))
  inline def log10[DT <: DataType](x: Tensor[DT]): Tensor[DT]      = Tensor(_mx.log10(x.data))
  inline def log1p[DT <: DataType](x: Tensor[DT]): Tensor[DT]      = Tensor(_mx.log1p(x.data))
  inline def sqrt[DT <: DataType](x: Tensor[DT]): Tensor[DT]       = Tensor(_mx.sqrt(x.data))
  inline def rsqrt[DT <: DataType](x: Tensor[DT]): Tensor[DT]      = Tensor(_mx.rsqrt(x.data))
  inline def square[DT <: DataType](x: Tensor[DT]): Tensor[DT]     = Tensor(_mx.square(x.data))
  inline def reciprocal[DT <: DataType](x: Tensor[DT]): Tensor[DT] = Tensor(_mx.reciprocal(x.data))
  inline def ceil[DT <: DataType](x: Tensor[DT]): Tensor[DT]       = Tensor(_mx.ceil(x.data))
  inline def floor[DT <: DataType](x: Tensor[DT]): Tensor[DT]      = Tensor(_mx.floor(x.data))
  inline def round[DT <: DataType](x: Tensor[DT]): Tensor[DT]      = Tensor(_mx.round(x.data))
  inline def sin[DT <: DataType](x: Tensor[DT]): Tensor[DT]        = Tensor(_mx.sin(x.data))
  inline def cos[DT <: DataType](x: Tensor[DT]): Tensor[DT]        = Tensor(_mx.cos(x.data))
  inline def tan[DT <: DataType](x: Tensor[DT]): Tensor[DT]        = Tensor(_mx.tan(x.data))
  inline def sinh[DT <: DataType](x: Tensor[DT]): Tensor[DT]       = Tensor(_mx.sinh(x.data))
  inline def cosh[DT <: DataType](x: Tensor[DT]): Tensor[DT]       = Tensor(_mx.cosh(x.data))
  inline def tanh[DT <: DataType](x: Tensor[DT]): Tensor[DT]       = Tensor(_mx.tanh(x.data))
  inline def arcsin[DT <: DataType](x: Tensor[DT]): Tensor[DT]     = Tensor(_mx.arcsin(x.data))
  inline def arccos[DT <: DataType](x: Tensor[DT]): Tensor[DT]     = Tensor(_mx.arccos(x.data))
  inline def arctan[DT <: DataType](x: Tensor[DT]): Tensor[DT]     = Tensor(_mx.arctan(x.data))
  inline def sigmoid[DT <: DataType](x: Tensor[DT]): Tensor[DT]    = Tensor(_mx.sigmoid(x.data))
  inline def erf[DT <: DataType](x: Tensor[DT]): Tensor[DT]        = Tensor(_mx.erf(x.data))
  inline def erfinv[DT <: DataType](x: Tensor[DT]): Tensor[DT]     = Tensor(_mx.erfinv(x.data))
  inline def softmax[DT <: DataType](x: Tensor[DT]): Tensor[DT]    = Tensor(_mx.softmax(x.data))
  inline def softmax[DT <: DataType](x: Tensor[DT], axis: Int): Tensor[DT] =
    Tensor(_mx.softmax(x.data, axis))
  inline def logicalNot(x: Tensor[Bool]): Tensor[Bool] = Tensor(_mx.logical_not(x.data))

  inline def isnan[DT <: DataType](x: Tensor[DT]): Tensor[Bool]    = Tensor(_mx.isnan(x.data))
  inline def isinf[DT <: DataType](x: Tensor[DT]): Tensor[Bool]    = Tensor(_mx.isinf(x.data))
  inline def isfinite[DT <: DataType](x: Tensor[DT]): Tensor[Bool] = Tensor(_mx.isfinite(x.data))

  // --- elementwise binary (T,T) and (T,scalar) -------------------------

  inline def add[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = Tensor(_mx.add(a.data, b.data))
  inline def add[DT <: DataType, V](a: Tensor[DT], v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] =
    Tensor(_mx.add(a.data, v))
  inline def subtract[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = Tensor(_mx.subtract(a.data, b.data))
  inline def subtract[DT <: DataType, V](a: Tensor[DT], v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] =
    Tensor(_mx.subtract(a.data, v))
  inline def multiply[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = Tensor(_mx.multiply(a.data, b.data))
  inline def multiply[DT <: DataType, V](a: Tensor[DT], v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] =
    Tensor(_mx.multiply(a.data, v))
  inline def divide[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = Tensor(_mx.divide(a.data, b.data))
  inline def divide[DT <: DataType, V](a: Tensor[DT], v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] =
    Tensor(_mx.divide(a.data, v))
  inline def power[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = Tensor(_mx.power(a.data, b.data))
  inline def power[DT <: DataType, V](a: Tensor[DT], v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] =
    Tensor(_mx.power(a.data, v))
  inline def maximum[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = Tensor(_mx.maximum(a.data, b.data))
  inline def minimum[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = Tensor(_mx.minimum(a.data, b.data))
  inline def floorDivide[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] = Tensor(_mx.floor_divide(a.data, b.data))
  inline def remainder[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT]   = Tensor(_mx.remainder(a.data, b.data))

  inline def logicalAnd(a: Tensor[Bool], b: Tensor[Bool]): Tensor[Bool] = Tensor(_mx.logical_and(a.data, b.data))
  inline def logicalOr (a: Tensor[Bool], b: Tensor[Bool]): Tensor[Bool] = Tensor(_mx.logical_or (a.data, b.data))

  // --- comparisons (return Bool) ---------------------------------------

  inline def equal[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[Bool]    = Tensor(_mx.equal(a.data, b.data))
  inline def notEqual[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[Bool] = Tensor(_mx.not_equal(a.data, b.data))
  inline def less[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[Bool]     = Tensor(_mx.less(a.data, b.data))
  inline def lessEqual[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[Bool] = Tensor(_mx.less_equal(a.data, b.data))
  inline def greater[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[Bool]   = Tensor(_mx.greater(a.data, b.data))
  inline def greaterEqual[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[Bool] = Tensor(_mx.greater_equal(a.data, b.data))

  // --- reductions ------------------------------------------------------

  inline def sum[DT <: DataType](x: Tensor[DT]): Tensor[DT] = Tensor(_mx.sum(x.data))
  inline def sum[DT <: DataType, ST: IsSizeType](x: Tensor[DT], axis: ST, keepdims: Boolean): Tensor[DT] =
    Tensor(_mx.sum(x.data, axis, keepdims))
  inline def mean[DT <: DataType](x: Tensor[DT]): Tensor[DT] = Tensor(_mx.mean(x.data))
  inline def mean[DT <: DataType, ST: IsSizeType](x: Tensor[DT], axis: ST, keepdims: Boolean): Tensor[DT] =
    Tensor(_mx.mean(x.data, axis, keepdims))
  inline def max[DT <: DataType](x: Tensor[DT]): Tensor[DT] = Tensor(_mx.max(x.data))
  inline def max[DT <: DataType, ST: IsSizeType](x: Tensor[DT], axis: ST, keepdims: Boolean): Tensor[DT] =
    Tensor(_mx.max(x.data, axis, keepdims))
  inline def min[DT <: DataType](x: Tensor[DT]): Tensor[DT] = Tensor(_mx.min(x.data))
  inline def min[DT <: DataType, ST: IsSizeType](x: Tensor[DT], axis: ST, keepdims: Boolean): Tensor[DT] =
    Tensor(_mx.min(x.data, axis, keepdims))
  inline def prod[DT <: DataType](x: Tensor[DT]): Tensor[DT] = Tensor(_mx.prod(x.data))
  inline def prod[DT <: DataType, ST: IsSizeType](x: Tensor[DT], axis: ST, keepdims: Boolean): Tensor[DT] =
    Tensor(_mx.prod(x.data, axis, keepdims))
  inline def std[DT <: DataType](x: Tensor[DT]): Tensor[DT] = Tensor(_mx.std(x.data))

  inline def all(x: Tensor[Bool]): Tensor[Bool] = Tensor(_mx.all(x.data))
  inline def any(x: Tensor[Bool]): Tensor[Bool] = Tensor(_mx.any(x.data))

  inline def argmax[DT <: DataType](x: Tensor[DT]): Tensor[UInt32]            = Tensor(_mx.argmax(x.data))
  inline def argmax[DT <: DataType](x: Tensor[DT], axis: Int): Tensor[UInt32] = Tensor(_mx.argmax(x.data, axis))
  inline def argmin[DT <: DataType](x: Tensor[DT]): Tensor[UInt32]            = Tensor(_mx.argmin(x.data))
  inline def argmin[DT <: DataType](x: Tensor[DT], axis: Int): Tensor[UInt32] = Tensor(_mx.argmin(x.data, axis))

  // --- shape transformations ------------------------------------------

  inline def reshape[DT <: DataType, ST: IsSizeType](x: Tensor[DT], shape: ST): Tensor[DT] =
    Tensor(_mx.reshape(x.data, shape))
  inline def transpose[DT <: DataType](x: Tensor[DT]): Tensor[DT] = Tensor(_mx.transpose(x.data))
  inline def transpose[DT <: DataType, ST: IsSizeType](x: Tensor[DT], axes: ST): Tensor[DT] =
    Tensor(_mx.transpose(x.data, axes))
  inline def swapaxes[DT <: DataType](x: Tensor[DT], a: Int, b: Int): Tensor[DT] =
    Tensor(_mx.swapaxes(x.data, a, b))
  inline def moveaxis[DT <: DataType](x: Tensor[DT], src: Int, dst: Int): Tensor[DT] =
    Tensor(_mx.moveaxis(x.data, src, dst))
  inline def expandDims[DT <: DataType](x: Tensor[DT], axis: Int): Tensor[DT] =
    Tensor(_mx.expand_dims(x.data, axis))
  inline def squeeze[DT <: DataType](x: Tensor[DT]): Tensor[DT] = Tensor(_mx.squeeze(x.data))
  inline def squeeze[DT <: DataType](x: Tensor[DT], axis: Int): Tensor[DT] = Tensor(_mx.squeeze(x.data, axis))
  inline def flatten[DT <: DataType](x: Tensor[DT]): Tensor[DT] = Tensor(_mx.flatten(x.data))
  inline def broadcastTo[DT <: DataType, ST: IsSizeType](x: Tensor[DT], shape: ST): Tensor[DT] =
    Tensor(_mx.broadcast_to(x.data, shape))

  // --- combining ------------------------------------------------------
  // MLX's `concatenate` / `stack` expect a python `list[array]`. Build one
  // via `builtins.list` from a Scala tuple of array payloads.

  inline def concatenate[DT <: DataType](a: Tensor[DT], b: Tensor[DT], axis: Int): Tensor[DT] =
    Tensor(_mx.concatenate(pyBuiltins.list((a.data, b.data)), axis))
  inline def concatenate[DT <: DataType](a: Tensor[DT], b: Tensor[DT], c: Tensor[DT], axis: Int): Tensor[DT] =
    Tensor(_mx.concatenate(pyBuiltins.list((a.data, b.data, c.data)), axis))
  inline def concatenate[DT <: DataType](a: Tensor[DT], b: Tensor[DT], c: Tensor[DT], d: Tensor[DT], axis: Int): Tensor[DT] =
    Tensor(_mx.concatenate(pyBuiltins.list((a.data, b.data, c.data, d.data)), axis))

  inline def stack[DT <: DataType](a: Tensor[DT], b: Tensor[DT], axis: Int): Tensor[DT] =
    Tensor(_mx.stack(pyBuiltins.list((a.data, b.data)), axis))
  inline def stack[DT <: DataType](a: Tensor[DT], b: Tensor[DT], c: Tensor[DT], axis: Int): Tensor[DT] =
    Tensor(_mx.stack(pyBuiltins.list((a.data, b.data, c.data)), axis))
  inline def stack[DT <: DataType](a: Tensor[DT], b: Tensor[DT], c: Tensor[DT], d: Tensor[DT], axis: Int): Tensor[DT] =
    Tensor(_mx.stack(pyBuiltins.list((a.data, b.data, c.data, d.data)), axis))

  inline def tile[DT <: DataType, ST: IsSizeType](x: Tensor[DT], reps: ST): Tensor[DT] =
    Tensor(_mx.tile(x.data, reps))
  inline def repeat[DT <: DataType](x: Tensor[DT], repeats: Int, axis: Int): Tensor[DT] =
    Tensor(_mx.repeat(x.data, repeats, axis))

  // --- linear algebra --------------------------------------------------

  inline def matmul[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] =
    Tensor(_mx.matmul(a.data, b.data))
  inline def inner[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] =
    Tensor(_mx.inner(a.data, b.data))
  inline def outer[DT <: DataType](a: Tensor[DT], b: Tensor[DT]): Tensor[DT] =
    Tensor(_mx.outer(a.data, b.data))
  inline def tensordot[DT <: DataType](a: Tensor[DT], b: Tensor[DT], axes: Int): Tensor[DT] =
    Tensor(_mx.tensordot(a.data, b.data, axes))

  // --- selection -------------------------------------------------------

  inline def where[DT <: DataType](cond: Tensor[Bool], x: Tensor[DT], y: Tensor[DT]): Tensor[DT] =
    Tensor(_mx.where(cond.data, x.data, y.data))
  inline def clip[DT <: DataType](x: Tensor[DT], lo: Tensor[DT], hi: Tensor[DT]): Tensor[DT] =
    Tensor(_mx.clip(x.data, lo.data, hi.data))
  inline def clip[DT <: DataType, V](x: Tensor[DT], lo: V, hi: V)(using
    ti: DataTypeInfo[DT], ev: V <:< ti.ItemType
  ): Tensor[DT] =
    Tensor(_mx.clip(x.data, lo, hi))

  // --- evaluation ------------------------------------------------------

  inline def eval[DT <: DataType](xs: Tensor[DT]): Unit = { _mx.eval(xs.data); () }

// ---------------------------------------------------------------------------
// Operator syntax
// ---------------------------------------------------------------------------

extension [DT <: DataType](self: Tensor[DT])
  inline def + (that: Tensor[DT]): Tensor[DT] = Tensor.add(self, that)
  inline def - (that: Tensor[DT]): Tensor[DT] = Tensor.subtract(self, that)
  inline def * (that: Tensor[DT]): Tensor[DT] = Tensor.multiply(self, that)
  inline def / (that: Tensor[DT]): Tensor[DT] = Tensor.divide(self, that)
  inline def ** (that: Tensor[DT]): Tensor[DT] = Tensor.power(self, that)
  inline def unary_- : Tensor[DT] = Tensor.negative(self)

  inline def === (that: Tensor[DT]): Tensor[Bool] = Tensor.equal(self, that)
  inline def !== (that: Tensor[DT]): Tensor[Bool] = Tensor.notEqual(self, that)
  inline def <   (that: Tensor[DT]): Tensor[Bool] = Tensor.less(self, that)
  inline def <=  (that: Tensor[DT]): Tensor[Bool] = Tensor.lessEqual(self, that)
  inline def >   (that: Tensor[DT]): Tensor[Bool] = Tensor.greater(self, that)
  inline def >=  (that: Tensor[DT]): Tensor[Bool] = Tensor.greaterEqual(self, that)

  /** Matrix multiplication, infix form. */
  inline def @@ (that: Tensor[DT]): Tensor[DT] = Tensor.matmul(self, that)

extension [DT <: DataType, V](self: Tensor[DT])
  inline def + (v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] = Tensor.add(self, v)
  inline def - (v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] = Tensor.subtract(self, v)
  inline def * (v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] = Tensor.multiply(self, v)
  inline def / (v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] = Tensor.divide(self, v)
  inline def ** (v: V)(using ti: DataTypeInfo[DT], ev: V <:< ti.ItemType): Tensor[DT] = Tensor.power(self, v)
