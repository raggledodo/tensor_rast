licenses(["notice"])

package(
    default_visibility = [ "//visibility:public" ],
)

# local sources
filegroup(
    name = "srcs",
    srcs = glob([ "**/*.py" ]) + [
        ":cfgs",
        "BUILD",
    ],
)

# main source files
filegroup(
    name = "tensor_rast_py",
    srcs = [
        "tensor_rast/shaped_node.py",
        "tensor_rast/tfgen.py",
    ],
)

# test source files
filegroup(
    name = "test_py",
    srcs = [
        "tensor_rast/test.py",
    ],
)

# data
filegroup(
    name = "cfgs",
    srcs = [ "tensor.yaml" ]
)

# ===== LIBRARY =====
py_library(
    name = "tensor_rast",
    srcs = [ ":tensor_rast_py" ],
    deps = [ "@com_github_mingkaic_rast//:rast" ],
    data = [ ":cfgs" ]
)

# ===== TEST =====
py_test(
    name = "test",
    srcs = [ ":test_py" ],
    deps = [ ":tensor_rast" ],
    size = "enormous",
)
