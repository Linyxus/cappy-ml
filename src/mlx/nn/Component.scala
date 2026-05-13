package mlx.nn

import scala.python.*
import mlx.core.{Tensor, DataType}

/** Canonical handle to the Python `mlx.nn` module. Shared by every layer
 *  facade in this package — keeps the `_nn.Foo(...)` factory dispatch in
 *  one place instead of re-declaring it per file. */
@extern("mlx.nn") private[nn] object _nn extends PyDynamic

/** Untyped cast to `PyDynamic` for raw Python calls. Used inside layer
 *  facades whose bodies need to bridge both `Tensor[DT]` (whose `toPy`
 *  lives in `mlx.core`) and `Component` (whose `toPy` lives here) — a
 *  wildcard import of either side would shadow the other. `asPy` is
 *  package-private to `mlx.nn` and `inline`, so it collapses to a no-op
 *  cast at every call site. */
private[nn] inline def asPy[T](x: T): PyDynamic = x.asInstanceOf[PyDynamic]

// ---------------------------------------------------------------------------
// Facade for `mlx.nn.Module`.
// ---------------------------------------------------------------------------

@extern("mlx.nn", "Module")
class Component:

  // --- properties (Python @property) ----------------------------------

  /** Python `training` attribute — `true` iff this module is in train mode. */
  @name("training") val isTrainingMode: Boolean = native

  /** Module state: parameters + buffers as a nested Python dict/list. */
  val state: PyDynamic = native

  // --- parameter collections ------------------------------------------

  /** Every registered `mx.array` parameter, as a nested Python dict/list. */
  def parameters(): PyDynamic = native

  /** Subset of `parameters()` that is not frozen. */
  @name("trainable_parameters") def trainableParameters(): PyDynamic = native

  // --- module hierarchy ------------------------------------------------

  /** Direct child `Module` instances (Python iterable). */
  def children(): PyDynamic = native

  /** Flat list of every contained module (including `this`). */
  def modules(): PyDynamic = native

  /** `(dotted-name, module)` pairs covering the full tree. */
  @name("named_modules") def namedModules(): PyDynamic = native

  /** Submodules that themselves contain no further nested modules. */
  @name("leaf_modules") def leafModules(): PyDynamic = native

  // --- training mode (eval has one signature; train is overloaded) -----

  /** Switch to evaluation mode. Equivalent to `train(false)`. */
  def eval(): Component = native

  // --- weight save ----------------------------------------------------

  /** Save weights to a `.npz` or `.safetensors` file path. */
  @name("save_weights") def saveWeights(file: String): Unit = native

  // --- functional (apply_to_modules has one signature) ----------------

  /** Run `applyFn(name, module)` across this module and every descendant. */
  @name("apply_to_modules") def applyToModules(applyFn: Any): Component = native

// ---------------------------------------------------------------------------
// Extension surface — Scala-side dispatch for the overloaded methods.
//
// Every member here is `inline`, so the body is spliced into the caller —
// no static-method call, no allocation, just a direct `Dynamic` dispatch
// to the Python attribute. `toPy` is itself an inline no-op cast, so a
// call like `m.freeze()` collapses to `m.asInstanceOf[PyDynamic].freeze()`.
// ---------------------------------------------------------------------------

extension (c: Component)

  /** Lift `c` back to a raw `PyDynamic` for un-typed Python calls. */
  inline def toPy: PyDynamic = c.asInstanceOf[PyDynamic]

  // --- train (overloaded over `mode`) ---------------------------------

  /** Switch to training mode. */
  inline def train(): Component =
    c.toPy.train()
    c
  /** Switch to training mode if `mode`, else evaluation mode. */
  inline def train(mode: Boolean): Component =
    c.toPy.train(mode)
    c

  // --- freeze / unfreeze (keyword-only args in Python) ----------------

  inline def freeze(): Component =
    c.toPy.freeze()
    c
  inline def freeze(recurse: Boolean): Component =
    c.toPy.freeze(recurse = recurse)
    c
  inline def freeze(recurse: Boolean, strict: Boolean): Component =
    c.toPy.freeze(recurse = recurse, strict = strict)
    c

  inline def unfreeze(): Component =
    c.toPy.unfreeze()
    c
  inline def unfreeze(recurse: Boolean): Component =
    c.toPy.unfreeze(recurse = recurse)
    c
  inline def unfreeze(recurse: Boolean, strict: Boolean): Component =
    c.toPy.unfreeze(recurse = recurse, strict = strict)
    c

  // --- updates (optional `strict` flag) -------------------------------

  /** Replace parameters in-place from a nested dict (e.g. from `parameters()`). */
  inline def update(parameters: PyDynamic): Component =
    c.toPy.update(parameters)
    c
  inline def update(parameters: PyDynamic, strict: Boolean): Component =
    c.toPy.update(parameters, strict = strict)
    c

  /** Replace child modules in-place from a nested dict (e.g. from `children()`). */
  inline def updateModules(modules: PyDynamic): Component =
    c.toPy.update_modules(modules)
    c
  inline def updateModules(modules: PyDynamic, strict: Boolean): Component =
    c.toPy.update_modules(modules, strict = strict)
    c

  // --- weight load (file path vs. in-memory weight list) --------------

  /** Load weights from a `.npz` / `.safetensors` file path. */
  inline def loadWeights(file: String): Component =
    c.toPy.load_weights(file)
    c
  inline def loadWeights(file: String, strict: Boolean): Component =
    c.toPy.load_weights(file, strict = strict)
    c
  /** Load weights from a `list[(name, mx.array)]`. */
  inline def loadWeights(weights: PyDynamic): Component =
    c.toPy.load_weights(weights)
    c
  inline def loadWeights(weights: PyDynamic, strict: Boolean): Component =
    c.toPy.load_weights(weights, strict = strict)
    c

  // --- dtype (optional predicate) -------------------------------------

  /** Cast parameters to `dtype` (by default, floating-point only). */
  inline def setDtype(dtype: PyDynamic): Component =
    c.toPy.set_dtype(dtype)
    c
  /** Cast parameters that satisfy `predicate: dtype -> bool`. */
  inline def setDtype(dtype: PyDynamic, predicate: Any): Component =
    c.toPy.set_dtype(dtype, predicate)
    c

  // --- functional (apply has optional filterFn; same for filter_and_map)

  /** Apply `mapFn` to every (filtered) parameter and update in-place.
   *  Mapped to Python `Module.apply`; the Scala name avoids clashing with
   *  the `def apply` callable syntax that subclasses use for a forward pass. */
  inline def applyToParameters(mapFn: Any): Component =
    c.toPy.apply(mapFn)
    c
  inline def applyToParameters(mapFn: Any, filterFn: Any): Component =
    c.toPy.apply(mapFn, filterFn)
    c

  /** Recursively gather module contents that satisfy `filterFn`. */
  inline def filterAndMap(filterFn: Any): PyDynamic =
    c.toPy.filter_and_map(filterFn)
  /** Same, but also map each matched entry through `mapFn`. */
  inline def filterAndMap(filterFn: Any, mapFn: Any): PyDynamic =
    c.toPy.filter_and_map(filterFn, mapFn)

  // --- forward pass (callable syntax) ----------------------------------
  //
  // Single `apply` extension on the *base* type. Every layer facade
  // inherits it through subtyping — so `linear(x)`, `relu(x)`,
  // `mlp(x)`, `layerNorm(x)` all dispatch through here without each
  // layer needing its own `apply` extension. (Multiple same-named
  // `apply` extensions across sibling subtypes would not disambiguate
  // by receiver type.) Layers whose forward pass doesn't fit this
  // signature — `Embedding` (changes dtype) and `MultiHeadAttention`
  // (takes 3+ tensors) — expose their forward under a different name.

  /** Forward pass: `layer(x)` in Scala → `layer(x)` in Python.
   *  We dispatch via `__call__` explicitly because Scala's `pd(args)`
   *  syntax desugars to `pd.apply(args)`, which Scala-Py compiles to a
   *  literal Python `pd.apply(args)` — and that resolves to
   *  `Module.apply` (parameter-update), not the forward pass. Naming
   *  `__call__` reaches Python's call protocol directly. */
  inline def apply[DT <: DataType](x: Tensor[DT]): Tensor[DT] =
    c.toPy.__call__(asPy(x)).asInstanceOf[Tensor[DT]]
