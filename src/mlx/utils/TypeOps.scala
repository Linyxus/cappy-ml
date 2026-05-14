package mlx.utils

// ---------------------------------------------------------------------------
// Compile-time shape / axis plumbing.
//
// `IsSizeType[T]` is the constraint used everywhere MLX accepts either a
// single `Int` axis or an `(Int, Int, ...)` tuple shape — e.g.
// `Tensor.reshape`, `Tensor.zeros`, `Conv2d`'s `kernelSize`. It admits
// `Int` directly, or any tuple whose every element is an `Int` (via the
// auxiliary `IsTupleOfType` typeclass, which is inductive on `*:`).
// ---------------------------------------------------------------------------

/** Evidence that `T` is a tuple whose every element is a subtype of `E`. */
trait IsTupleOfType[T, E]
object IsTupleOfType:
  given empty[E]: IsTupleOfType[EmptyTuple, E] = new IsTupleOfType[EmptyTuple, E] {}
  given cons[H, T <: Tuple, E](using ev: H <:< E, tail: IsTupleOfType[T, E]): IsTupleOfType[H *: T, E] =
    new IsTupleOfType[H *: T, E] {}

/** Evidence that `T` is acceptable as a shape / axis argument — either a
 *  single `Int` or a tuple of `Int`s. */
trait IsSizeType[T]
object IsSizeType:
  given IsSizeType[Int] with {}
  given [T <: Tuple](using IsTupleOfType[T, Int]): IsSizeType[T] with {}
