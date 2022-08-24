import os
import sys

from pathlib import Path
from subprocess import check_output, CalledProcessError

from llvmlite import binding as llvm

from .parser import Parser
from .llvmgen import LLVMGenerator


def compile(filename: str) -> int:
    path = Path(filename)

    with open(filename, "r") as f:
        source = f.read()

    tree = Parser(source, filename).parse()

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_all_asmprinters()

    codegen = LLVMGenerator(filename, path)
    codegen.generate(tree)

    with open(f"{path.stem}.ll", "w") as f:
        f.write(str(codegen))

    if not os.path.exists("bin"):
        os.mkdir("bin")

    out = Path("bin").joinpath(path.stem)

    try:
        check_output(["clang", f"{path.stem}.ll", "-o", f"{out}"])
    except CalledProcessError as e:
        sys.exit(e.returncode)
    finally:
        os.remove(f"{path.stem}.ll")

    return 0
