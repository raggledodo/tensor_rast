load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def dependencies():
    # protobuf dependency
    if "com_github_mingkaic_rast" not in native.existing_rules():
        git_repository(
            name = "com_github_mingkaic_rast",
            remote = "https://github.com/raggledodo/rast",
            commit = "44054c0ece2b723159ec6942d0a7dfc21aa46b56",
        )
