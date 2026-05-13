import scala.python.*
import mlx.utils.*
import mlx.nn.*
import mlx.core.{DataTypeInfo, Float16}

// Inline reference to `mlx.nn` — keeps the test focused while still letting
// us instantiate real `Linear` children so the parameter collections aren't
// trivially empty.
@extern("mlx.nn") private object _nn extends PyDynamic

// Reference to `mlx.utils` (the Python module) for `tree_flatten`. The Scala
// `mlx.utils` package collides on the import name, so we keep this private.
@extern("mlx.utils") private object _mlxutils extends PyDynamic

class MyMlp(val inputDim: Int, val outputDim: Int, hiddenDim: Int = 32) extends Component:
  // Two `Linear` children registered through Python `__setattr__` at ctor
  // time, so the facade hierarchy methods have something to enumerate.
  val fc1: PyDynamic = _nn.Linear(inputDim, hiddenDim)
  val fc2: PyDynamic = _nn.Linear(hiddenDim, outputDim)
  override def toString(): String = s"MLP($inputDim --$hiddenDim--> $outputDim)"

/** Count the (name, array) leaves under a nested parameter dict. */
def leafCount(tree: PyDynamic): PyDynamic =
  pyBuiltins.len(_mlxutils.tree_flatten(tree))

@main def runTestNN(): Unit =
  val model = MyMlp(4, 2, 8)

  println("=== properties (val natives) ===")
  println(s"toString          : $model")
  println(s"isTrainingMode    : ${model.isTrainingMode}")
  println(s"state             : ${model.state}")
  println(s"state size        : ${pyBuiltins.len(model.state)}")

  println("\n=== module hierarchy (def natives) ===")
  println(s"children count    : ${pyBuiltins.len(model.children())}")
  println(s"modules count     : ${pyBuiltins.len(model.modules())}")
  println(s"namedModules count: ${pyBuiltins.len(model.namedModules())}")
  println(s"leafModules count : ${pyBuiltins.len(model.leafModules())}")

  println("\n=== parameter collections (def natives) ===")
  println(s"parameters arrays : ${leafCount(model.parameters())}")
  println(s"trainable arrays  : ${leafCount(model.trainableParameters())}")

  println("\n=== mode switching (eval native; train extension overload) ===")
  println(s"start             : training=${model.isTrainingMode}")
  model.eval()
  println(s"after eval()      : training=${model.isTrainingMode}")
  model.train()
  println(s"after train()     : training=${model.isTrainingMode}")
  model.train(false)
  println(s"after train(false): training=${model.isTrainingMode}")
  model.train(true)
  println(s"after train(true) : training=${model.isTrainingMode}")

  println("\n=== chaining returns the same instance ===")
  val chained = model.eval().train().freeze().unfreeze()
  println(s"chain eq model    : ${chained eq model}")

  println("\n=== freeze / unfreeze (3 arities each) ===")
  model.unfreeze()
  println(s"unfrozen          : trainable arrays=${leafCount(model.trainableParameters())}")
  model.freeze()
  println(s"freeze()          : trainable arrays=${leafCount(model.trainableParameters())}")
  model.unfreeze()
  model.freeze(recurse = false)
  println(s"freeze(false)     : trainable arrays=${leafCount(model.trainableParameters())}")
  model.unfreeze(recurse = false)
  model.freeze(recurse = true, strict = false)
  println(s"freeze(true,false): trainable arrays=${leafCount(model.trainableParameters())}")
  model.unfreeze(recurse = true, strict = false)
  println(s"unfreeze(t,f)     : trainable arrays=${leafCount(model.trainableParameters())}")

  println("\n=== weight save / load roundtrip (saveWeights native; loadWeights 4-way overload) ===")
  val tmp = "/tmp/mlx_facade_test.npz"
  model.saveWeights(tmp)
  model.loadWeights(tmp)
  model.loadWeights(tmp, strict = true)
  val asList: PyDynamic = _mlxutils.tree_flatten(model.parameters())
  model.loadWeights(asList)
  model.loadWeights(asList, strict = true)
  println(s"save+load OK      : file=$tmp")

  println("\n=== update / updateModules (2 arities each) ===")
  val snap = model.parameters()
  model.update(snap)
  model.update(snap, strict = false)
  val kids = model.children()
  model.updateModules(kids)
  model.updateModules(kids, strict = false)
  println("update + updateModules OK")

  println("\n=== setDtype (2 arities) ===")
  val f16 = summon[DataTypeInfo[Float16]].dtype
  model.setDtype(f16)
  println(s"after setDtype(f16): fc1.weight.dtype=${model.toPy.fc1.weight.dtype}")
  // Predicate takes the current dtype and returns whether to cast it.
  val castAll = pyBuiltins.eval("lambda dt: True")
  model.setDtype(f16, castAll)
  println("setDtype(dtype, predicate) OK")

  println("\n=== applyToModules (def native, side-effect callable) ===")
  // Python `print` is called once per (name, module).
  pyBuiltins.eval("__import__('sys').stdout.flush()")
  model.applyToModules(pyBuiltins.print)
  pyBuiltins.eval("__import__('sys').stdout.flush()")

  println("\n=== applyToParameters (2 arities) ===")
  // map_fn must take an mx.array and return one; filter_fn must take
  // (module, key, value) and return bool.
  val identity = pyBuiltins.eval("lambda x: x")
  val onlyArrays = pyBuiltins.eval("lambda m, k, v: hasattr(v, 'dtype')")
  model.applyToParameters(identity)
  model.applyToParameters(identity, onlyArrays)
  println("applyToParameters OK")

  println("\n=== filterAndMap (2 arities) ===")
  // filter_fn: (module, key, value) -> bool;  map_fn: value -> mapped.
  val keepAll = pyBuiltins.eval("lambda m, k, v: True")
  val tagType = pyBuiltins.eval("lambda v: type(v).__name__")
  println(s"filterAndMap(keep) arrays  : ${leafCount(model.filterAndMap(keepAll))}")
  println(s"filterAndMap(keep, tag)    : ${model.filterAndMap(keepAll, tagType)}")

  println("\n=== toPy escape hatch ===")
  // Avoid stringifying the bare class object — its `toString` resolves to an
  // unbound Scala method. Read `__name__` off the type instead.
  println(s"toPy class name   : ${pyBuiltins.`type`(model.toPy).__name__}")
  println(s"raw fc1           : ${model.toPy.fc1}")
  println(s"raw __repr__      : ${pyBuiltins.repr(model.toPy)}")

  println("\nall facade members exercised.")
