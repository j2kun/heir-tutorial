load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_go//go:def.bzl", "go_binary")
load("@rules_heir//heir:lattigo.bzl", "heir_lattigo_lib")
load("@rules_heir//heir:openfhe.bzl", "heir_openfhe_lib")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

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

py_binary(
    name = "download_data",
    srcs = ["download_data.py"],
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

heir_lattigo_lib(
    name = "fashion_mnist_lattigo_lib",
    go_library_name = "main",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=1024",
        "--scheme-to-lattigo",
    ],
    mlir_src = "model_annotated.mlir",
    split_preprocessing = False,
)

go_binary(
    name = "fashion_mnist_lattigo_bin",
    srcs = ["fashion_mnist_bin.go"],
    data = [
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",
    ],
    embed = [":main"],
)

heir_openfhe_lib(
    name = "fashion_mnist_openfhe_lib",
    cc_lib_target_name = "fashion_mnist_cc_lib",
    generated_lib_header = "fashion_mnist.inc.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=bgv",
        "--torch-linalg-to-ckks=ciphertext-degree=1024",
        "--scheme-to-openfhe",
    ],
    mlir_src = "model_annotated.mlir",
    pybind_target_name = "fashion_mnist_pybind",
)

py_binary(
    name = "fashion_mnist_openfhe_bin",
    srcs = ["fashion_mnist_bin.py"],
    data = [
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",
    ],
    main = "fashion_mnist_bin.py",
    deps = [
        ":fashion_mnist_pybind",
        "@pip//numpy",
        "@pip//torch",
    ],
)
