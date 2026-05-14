package mlx.core

import scala.python.*

// ---------------------------------------------------------------------------
// Dtypes — the phantom-type vocabulary for `Tensor[DT]`.
//
// Each `DataType` is a zero-instance marker class; the type parameter is
// erased at runtime. `DataTypeInfo[DT]` carries the runtime payload: the
// Python `mlx.core.dtype` object plus the Scala scalar type produced by
// `Tensor.item`.
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
