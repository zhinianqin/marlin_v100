from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


ROOT_DIR = Path(__file__).parent.resolve()


def parse_cmake_args(raw_args: str) -> list[str]:
    args = shlex.split(raw_args)
    normalized_args: list[str] = []
    for arg in args:
        if (
            normalized_args
            and not arg.startswith("-")
            and normalized_args[-1].startswith("-D")
            and "=" in normalized_args[-1]
        ):
            normalized_args[-1] = f"{normalized_args[-1]} {arg}"
            continue
        normalized_args.append(arg)
    return normalized_args


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".") -> None:
        super().__init__(name, sources=[], py_limited_api=True)
        self.cmake_lists_dir = str((ROOT_DIR / cmake_lists_dir).resolve())


class CMakeBuild(build_ext):
    def build_extensions(self) -> None:
        subprocess.check_call(["cmake", "--version"])

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            f"-DVLLM_PYTHON_EXECUTABLE={sys.executable}",
        ]
        other_cmake_args = os.environ.get("CMAKE_ARGS")
        if other_cmake_args:
            cmake_args.extend(parse_cmake_args(other_cmake_args))

        subprocess.check_call(
            ["cmake", ROOT_DIR.as_posix(), "-G", "Ninja", *cmake_args],
            cwd=build_temp,
        )

        targets = [ext.name.split(".")[-1] for ext in self.extensions]
        build_args = ["--build", ".", *[f"--target={target}" for target in targets]]
        subprocess.check_call(["cmake", *build_args], cwd=build_temp)

        for ext in self.extensions:
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
            prefix = outdir
            for _ in range(ext.name.count(".")):
                prefix = prefix.parent
            subprocess.check_call(
                [
                    "cmake",
                    "--install",
                    ".",
                    "--prefix",
                    prefix.as_posix(),
                    "--component",
                    ext.name.split(".")[-1],
                ],
                cwd=build_temp,
            )


setup(
    name="marlin-v100",
    version="0.0.0",
    packages=find_packages(where=".", include=["vllm", "vllm.*"]),
    ext_modules=[
        CMakeExtension("vllm._C"),
        CMakeExtension("vllm._moe_C"),
    ],
    cmdclass={"build_ext": CMakeBuild},
)
