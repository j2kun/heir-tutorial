# Compilation with HEIR

## Requirements

- Install [Bazelisk](https://github.com/bazelbuild/bazelisk) and put it on your
`PATH` as `bazel`.
- Python3.12

(Bazel will manage all other dependencies, including the C++ toolchain and
Lattigo/OpenFHE)

## Download train/test data

```bash
bazel run download_data
```

Puts the training/test data in `data/`

## Using the model_annotated.mlir already in this repo

Just run HEIR and view the generated C++ code

```bash
bazel build :all

less bazel-bin/fashionmnist_lib.go
less bazel-bin/fashion_mnist_heir_opt.mlir
```

Note you can also run the binary

```bash
bazel run fashion_mnist_bin
```

## Full workflow

### Export torch to MLIR

(Haven't figured out how to get this working in bazel yet because torch-mlir
does not work well with bazel's rules_python).

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels -r requirements.txt

python to_mlir.py > model.mlir
```

### Manually annotate secretness and activation degree/domain

Make a copy of `model.mlir` (`model_annotated.mlir`, referred to by name in
`BUILD.bazel`).

Manually add `{secret.secret}` annotations to the desired (to-be-encrypted)
function argument. Manually add `domain_lower`, `domain_upper`, and `degree`
annotation to each activation op in `model.mlir`.

E.g., a ReLU looks like this:

```mlir
%11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<1x64xf32>) outs(%7 : tensor<1x64xf32>) {
^bb0(%in: f32, %out: f32):
  %17 = arith.cmpf ugt, %in, %cst : f32
  %18 = arith.select %17, %in, %cst : f32
  linalg.yield %18 : f32
} -> tensor<1x64xf32>
```

And should be annotated as

```mlir
%11 = linalg.generic {
    // new stuff start
    domain_lower = -2.0 : f64,
    domain_upper = 3.0 : f64,
    degree = 3 : i32,
    // new stuff end
    indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]
} ins(%10 : tensor<1x64xf32>) outs(%7 : tensor<1x64xf32>) {
^bb0(%in: f32, %out: f32):
  %17 = arith.cmpf ugt, %in, %cst : f32
  %18 = arith.select %17, %in, %cst : f32
  linalg.yield %18 : f32
} -> tensor<1x64xf32>
```

### Run HEIR

```bash
bazel build :all
```
