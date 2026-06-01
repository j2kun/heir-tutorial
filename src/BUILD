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
