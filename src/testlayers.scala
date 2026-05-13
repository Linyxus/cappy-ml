import scala.python.*
import mlx.utils.*
import mlx.nn.*
import mlx.core.*

@main def runTestLayers(): Unit =

  println("=== Linear (typed, inherited Component surface) ===")
  val lin = Linear(4, 2)
  val linNoBias = Linear(4, 2, bias = false)
  println(s"lin                : ${pyBuiltins.`type`(lin.toPy).__name__}")
  println(s"lin.weight.shape   : ${lin.weight.shape}")
  println(s"lin.bias.shape     : ${lin.bias.shape}")
  println(s"linNoBias has bias : ${pyBuiltins.hasattr(linNoBias.toPy, "bias")}")
  // Component extension surface is inherited.
  lin.eval(); lin.train(); lin.freeze(); lin.unfreeze()
  println(s"lin trainable leaves: ${pyBuiltins.len(_layersUtils.tree_flatten(lin.trainableParameters()))}")
  val x4: Tensor[Float32] = Tensor.zeros((3, 4))
  val y4: Tensor[Float32] = lin(x4)
  println(s"lin(x).shape       : ${y4.shape}")  // (3, 2)

  println("\n=== Sequential(Linear, ReLU, Dropout, Linear) ===")
  val mlp = Sequential(
    Linear(8, 16),
    ReLU(),
    Dropout(0.2),
    Linear(16, 4),
  )
  println(s"mlp                : ${pyBuiltins.`type`(mlp.toPy).__name__}")
  println(s"mlp children       : ${pyBuiltins.len(mlp.children())}")
  val x8: Tensor[Float32] = Tensor.zeros((5, 8))
  val y8: Tensor[Float32] = mlp(x8)
  println(s"mlp(x).shape       : ${y8.shape}")  // (5, 4)

  println("\n=== Embedding (dtype changes, exposed as .lookup) ===")
  val emb = Embedding(numEmbeddings = 16, dims = 8)
  val ids: Tensor[Int32] = Tensor.array[Int32]((1, 2, 3, 0))
  val embOut: Tensor[Float32] = emb.lookup[Float32, Int32](ids)
  println(s"emb.weight.shape   : ${emb.weight.shape}")
  println(s"emb.lookup.shape   : ${embOut.shape}")  // (4, 8)
  // Tied-embedding projection.
  val tied: Tensor[Float32] = emb.asLinear(embOut)
  println(s"emb.asLinear.shape : ${tied.shape}")  // (4, 16)

  println("\n=== Normalisation: LayerNorm / RMSNorm / BatchNorm ===")
  // y4 has shape (3, 2) — last-dim is 2, so dims/numFeatures = 2.
  val ln = LayerNorm(dims = 2)
  val ln2 = LayerNorm(dims = 2, eps = 1e-6)
  val ln3 = LayerNorm(dims = 2, eps = 1e-6, affine = true, bias = false)
  val rms = RMSNorm(dims = 2, eps = 1e-6)
  val bn = BatchNorm(numFeatures = 2)
  println(s"ln(y4).shape       : ${ln(y4).shape}")
  println(s"ln2 weight.shape   : ${ln2.weight.shape}")
  println(s"ln3 weight.shape   : ${ln3.weight.shape}")
  println(s"rms(y4).shape      : ${rms(y4).shape}")
  println(s"bn(y4).shape       : ${bn(y4).shape}")
  println(s"bn.runningMean shp : ${bn.runningMean.shape}")

  println("\n=== Functional activations (Activations object) ===")
  val z: Tensor[Float32] = Tensor.array[Float32]((-1.0, 0.0, 1.0, 2.0))
  println(s"relu               : ${Activations.relu(z)}")
  println(s"gelu               : ${Activations.gelu(z)}")
  println(s"silu               : ${Activations.silu(z)}")
  println(s"sigmoid            : ${Activations.sigmoid(z)}")
  println(s"tanh               : ${Activations.tanh(z)}")
  println(s"softmax            : ${Activations.softmax(z)}")
  println(s"softmax axis=-1    : ${Activations.softmax(ln(y4), axis = -1)}")
  // (ln(y4) has shape (3,2); softmax along last axis returns same shape.)

  println("\n=== Module-style activations (slot into Sequential) ===")
  val gelu = GELU()
  val silu = SiLU()
  val sig = Sigmoid()
  val tanh = Tanh()
  val sm = Softmax()
  println(s"GELU(z)            : ${gelu(z)}")
  println(s"SiLU(z)            : ${silu(z)}")
  println(s"Sigmoid(z)         : ${sig(z)}")
  println(s"Tanh(z)            : ${tanh(z)}")
  println(s"Softmax(z)         : ${sm(z)}")

  println("\n=== Conv2d (kernelSize as Int and as (Int, Int)) ===")
  val convA = Conv2d(inChannels = 3, outChannels = 16, kernelSize = 3)
  val convB = Conv2d(3, 16, kernelSize = 3, stride = 1)
  val convC = Conv2d(3, 16, kernelSize = (3, 3), stride = (1, 1), padding = (1, 1))
  val img: Tensor[Float32] = Tensor.zeros((1, 32, 32, 3))  // NHWC
  println(s"convA(img).shape   : ${convA(img).shape}")
  println(s"convB(img).shape   : ${convB(img).shape}")
  println(s"convC(img).shape   : ${convC(img).shape}")

  println("\n=== Conv1d ===")
  val conv1 = Conv1d(inChannels = 4, outChannels = 8, kernelSize = 3, stride = 1, padding = 1)
  val seq: Tensor[Float32] = Tensor.zeros((2, 16, 4))  // NLC
  println(s"conv1(seq).shape   : ${conv1(seq).shape}")

  println("\n=== MultiHeadAttention (multi-tensor forward, exposed as .attend) ===")
  val mha = MultiHeadAttention(dims = 64, numHeads = 8)
  val tok: Tensor[Float32] = Tensor.zeros((2, 16, 64))
  val ctxNoMask: Tensor[Float32] = mha.attend(tok, tok, tok)
  val mask: Tensor[Float32] = Tensor.zeros((16, 16))
  val ctxMasked: Tensor[Float32] = mha.attend(tok, tok, tok, mask)
  println(s"mha.attend(t,t,t)   : ${ctxNoMask.shape}")
  println(s"mha.attend(t,t,t,m) : ${ctxMasked.shape}")

  println("\n=== Component surface inherited by every layer ===")
  println(s"mlp.isTrainingMode : ${mlp.isTrainingMode}")
  mlp.eval()
  println(s"after mlp.eval()   : ${mlp.isTrainingMode}")
  mlp.train()
  println(s"after mlp.train()  : ${mlp.isTrainingMode}")
  mlp.freeze()
  println(s"frozen trainable   : ${pyBuiltins.len(_layersUtils.tree_flatten(mlp.trainableParameters()))}")
  mlp.unfreeze()
  println(s"unfrozen trainable : ${pyBuiltins.len(_layersUtils.tree_flatten(mlp.trainableParameters()))}")

  println("\nall layer facade members exercised.")

// Internal references — only needed by the test for counting parameter leaves.
@extern("mlx.utils") private object _layersUtils extends PyDynamic
