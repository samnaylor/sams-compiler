from llvmlite import ir


class Namespace:
    def __init__(self, name: str) -> None:
        self.name = name

        self.structs: dict[str, ir.Function] = {}
        self.functions: dict[str, ir.Function] = {}

    def get(self, name: str) -> ir.Value | None:
        if (func := self.functions.get(name)) is not None:
            return func

        raise AttributeError(f"{self.name} has no attribute {name}")
