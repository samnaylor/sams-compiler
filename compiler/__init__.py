import os
from sys import platform
from pathlib import Path

from llvmlite import binding as llvm

from .parser import Parser
from .llvmgen import LLVMGenerator


def compile(filename: str) -> int:
    path = Path(filename)

    print(path.name)

    with open(filename, "r") as f:
        source = f.read()

    tree = Parser(source, filename).parse()

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_all_asmprinters()

    codegen = LLVMGenerator(filename)
    codegen.generate(tree)

    with open(f"{path.stem}.ll", "w") as f:
        f.write(str(codegen))

    if not os.path.exists("bin"):
        os.mkdir("bin")

    out = Path("bin").joinpath(path.stem)

    if platform == "windows":
        cmd = f"clang {path.stem}.ll -o {out}.exe"
    else:
        cmd = f"clang {path.stem}.ll -o {out}"

    os.system(cmd)

    os.remove(f"{path.stem}.ll")

    return 0
