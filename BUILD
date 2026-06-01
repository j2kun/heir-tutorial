load("@rules_go//go:def.bzl", "go_test")
load("@rules_heir//heir:lattigo.bzl", "heir_lattigo_lib")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_library(
    name = "model",
    srcs = ["model.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//torch",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":model",
        "@pip//torch",
        "@pip//torchvision",
    ],
)

py_binary(
    name = "infer",
    srcs = ["infer.py"],
    deps = [
        ":model",
        "@pip//torch",
        "@pip//torchvision",
    ],
)

heir_lattigo_lib(
    name = "fashion_mnist",
    go_library_name = "fashionmnist",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "model_annotated.mlir",
    split_preprocessing = False,
)

go_test(
    name = "fashion_mnist_test",
    size = "large",
    srcs = ["fashion_mnist_test.go"],
    data = [
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",
    ],
    embed = [":fashionmnist"],
)
