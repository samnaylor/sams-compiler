import os

from pathlib import Path
from typing import cast

from llvmlite import ir, binding as llvm

from .ast import (
    Extern,
    Node,
    Program,
    FunctionDefinition,
    FunctionSignature,
    TypeIdentifier,
    Block,
    Return,
    While,
    Var,
    Declarator,
    Expr_,
    BinaryOp,
    Variable,
    Assignment,
    Selection,
    ComparisonOp,
    IntLiteral,
    Call,
    ArrayLiteral,
    Index,
    IfElse,
    Break,
    Continue,
    Import,
    UnaryOp,
    FloatLiteral,
    StringLiteral,
    Cast,
    StructDefinition,
    Attr
)

from .lexer import generic_error, Location
from .parser import Parser

from .samtypes import Namespace


def llvm_generator_error(filename: str, location: Location, message: str) -> None:
    generic_error("LLVM Generation Error", filename, location, message)


class LLVMGenerator:
    def __init__(self, filename: str, callpath: Path):
        self.module = ir.Module()
        self.context = self.module.context
        self.filename = self.module.name = filename
        self.callpath = callpath

        self.builder: ir.IRBuilder | None = None

        self.locals: dict[str, ir.Value] = {}
        self.functions: dict[str, ir.Function] = {}

        self.block_start_stack: list[ir.Block] = []
        self.block_end_stack: list[ir.Block] = []

        self.dependencies: set[str] = set[str]()

        self.types: dict[str, ir.Type] = {
            "i32": ir.IntType(32),
            "i8": ir.IntType(8),
            "string": ir.IntType(8).as_pointer(),
            "float": ir.FloatType()
        }
        self.structures: dict[str, list[str]] = {}
        self.namespaces: dict[str, Namespace] = {}

        self.current_namespace: Namespace | None = None

        self._strcount = -1

    @property
    def strcount(self) -> int:
        self._strcount += 1
        return self._strcount

    def _optimise(self) -> llvm.ModuleRef:
        module = llvm.parse_assembly(str(self.module))
        module.name = self.module.name
        module.triple = llvm.Target.from_default_triple().triple

        pmb = llvm.create_pass_manager_builder()
        pm = llvm.create_module_pass_manager()

        pmb.populate(pm)
        pmb.opt_level = 3
        pmb.size_level = 2

        pm.add_instruction_combining_pass()
        pm.add_dead_arg_elimination_pass()
        pm.add_dead_code_elimination_pass()
        pm.add_cfg_simplification_pass()
        pm.add_constant_merge_pass()
        pm.add_global_optimizer_pass()
        pm.add_ipsccp_pass()
        pm.add_licm_pass()
        pm.add_loop_rotate_pass()
        pm.add_gvn_pass()
        pm.add_global_dce_pass()
        pm.add_sroa_pass()
        pm.add_sccp_pass()
        pm.add_refprune_pass()

        pm.run(module)

        module.verify()

        return module

    def __str__(self) -> str:
        return str(self._optimise())
        # return str(self.module)

    def generate_default(self, node: Node, *, flag: int = 0) -> None:
        raise NotImplementedError(node.__class__.__name__)

    def generate(self, node: Node, *, flag: int = 0) -> ir.Value | None:
        return getattr(self, f"generate_{node.__class__.__name__}", self.generate_default)(node, flag=flag)

    def generate_Program(self, node: Program, *, flag: int = 0) -> None:
        for imp in node.imports:
            self.generate(imp)

        for struct in node.structs:
            self.generate(struct, flag=flag)

        for ext in node.externs:
            self.generate(ext, flag=flag)

        for fdef in node.function_defs:
            self.generate(fdef, flag=flag)

    def generate_Import(self, node: Import, *, flag: int = 0) -> None:
        # TODO: This might need cleaning up, just wasn't sure of how else to do this.
        if node.module_name in self.dependencies:
            return

        self.current_namespace = Namespace(node.module_name)
        self.namespaces[node.module_name] = self.current_namespace

        self.dependencies.add(node.module_name)

        path = os.path.join(Path().root, *self.callpath.absolute().parts[:-1], f"{node.module_name}.sam")

        with open(path, "r") as f:
            tree = Parser(f.read(), node.module_name).parse()

        self.generate(tree, flag=1)
        self.current_namespace = None

    def generate_StructDefinition(self, node: StructDefinition, *, flag: int = 0) -> None:
        struct = self.context.get_identified_type(f"struct.{node.struct_name}")
        self.types[node.struct_name] = struct

        body: list[ir.Type] = []
        names: list[str] = []

        for (idx, param) in enumerate(node.struct_body):
            name = param.parameter_name
            type = cast(ir.Type, self.generate(param.parameter_type))

            names.append(name)
            body.append(type)

        struct.set_body(*body)

        if flag == 1:
            assert self.current_namespace is not None
            self.current_namespace.structs[node.struct_name] = names
        else:
            self.structures[node.struct_name] = names

        constructor = ir.Function(self.module, ir.FunctionType(struct, body), node.struct_name)
        builder = ir.IRBuilder(constructor.append_basic_block("entry"))

        alloca = builder.alloca(struct)

        for (idx, arg) in enumerate(constructor.args):
            ptr = builder.gep(alloca, (ir.IntType(32)(0), ir.IntType(32)(idx)))
            builder.store(arg, ptr)

        builder.ret(builder.load(alloca))

        if flag == 1:
            assert self.current_namespace is not None
            self.current_namespace.functions[node.struct_name] = constructor
        else:
            self.functions[node.struct_name] = constructor

    def generate_Extern(self, node: Extern, *, flag: int = 0) -> None:
        self.generate(node.function_signature, flag=flag)

    def generate_FunctionDefinition(self, node: FunctionDefinition, *, flag: int = 0) -> None:
        if flag == 1 and node.function_signature.function_name == "main":
            return

        self.locals = {}

        func = cast(ir.Function, self.generate(node.function_signature, flag=flag))
        self.builder = ir.IRBuilder(func.append_basic_block("entry"))

        for arg in func.args:
            alloca = self.builder.alloca(arg.type, name=arg.name)
            self.builder.store(arg, alloca)
            self.locals[arg.name] = alloca

        self.generate(node.function_body)

        for block in self.builder.function.basic_blocks:
            if not block.is_terminated:
                self.builder.position_at_end(block)
                self.builder.unreachable()

        self.builder = None

    def generate_FunctionSignature(self, node: FunctionSignature, *, flag: int = 0) -> ir.Value:
        retty = self.generate(node.function_retty)
        types = [self.generate(param.parameter_type) for param in node.function_params]

        signature = ir.FunctionType(retty, types, node.is_variadic)
        function = ir.Function(self.module, signature, node.function_name)

        for (arg, param) in zip(function.args, node.function_params):
            arg.name = param.parameter_name

        if flag == 1:
            assert self.current_namespace is not None
            self.current_namespace.functions[node.function_name] = function
        else:
            self.functions[node.function_name] = function

        return function

    def generate_TypeIdentifier(self, node: TypeIdentifier, *, flag: int = 0) -> ir.Type:
        base: ir.Type

        if (typ := self.types.get(node.typename)) is not None:
            base = typ

        if node.is_array:
            base = ir.ArrayType(base, node.array_sz)

        return base

    def generate_Block(self, node: Block, *, flag: int = 0) -> None:
        assert self.builder is not None

        for stmt in node.body:
            self.generate(stmt)

        # if not self.builder.block.is_terminated:
        #     self.builder.unreachable()

    def generate_Break(self, node: Break, *, flag: int = 0) -> None:
        assert self.builder is not None
        self.builder.branch(self.block_end_stack[-1])

    def generate_Continue(self, node: Continue, *, flag: int = 0) -> None:
        assert self.builder is not None
        self.builder.branch(self.block_start_stack[-1])

    def generate_Return(self, node: Return, *, flag: int = 0) -> None:
        assert self.builder is not None

        self.builder.ret(self.generate(node.return_value, flag=1))

    def generate_While(self, node: While, *, flag: int = 0) -> None:
        assert self.builder is not None

        cond_block = self.builder.append_basic_block("while.cond")
        body_block = self.builder.append_basic_block("while.body")
        post_block = self.builder.append_basic_block("while.post")

        self.block_start_stack.append(cond_block)
        self.block_end_stack.append(post_block)

        self.builder.branch(cond_block)
        self.builder.position_at_start(cond_block)
        cond = self.generate(node.condition, flag=1)
        self.builder.cbranch(cond, body_block, post_block)
        self.builder.position_at_start(body_block)
        self.generate(node.body)
        self.builder.branch(cond_block)
        self.builder.position_at_start(post_block)

        self.block_start_stack.pop()
        self.block_end_stack.pop()

        # TODO: this errors if we return from within the loop

    def generate_IfElse(self, node: IfElse, *, flag: int = 0) -> None:
        assert self.builder is not None

        cond = self.generate(node.if_cond, flag=1)

        if node.if_else is not None:
            with self.builder.if_else(cond) as (then, otherwise):
                with then:
                    self.generate(node.if_then)
                with otherwise:
                    self.generate(node.if_else)
        else:
            with self.builder.if_then(cond):
                self.generate(node.if_then)

    def generate_Var(self, node: Var, *, flag: int = 0) -> None:
        for decl in node.declarators:
            self.generate(decl)

    def generate_Declarator(self, node: Declarator, *, flag: int = 0) -> None:
        assert self.builder is not None

        typ = self.generate(node.type_id)
        alloca = self.builder.alloca(typ, name=node.identifier)
        value = self.generate(node.initialiser, flag=1)

        self.builder.store(value, alloca)
        self.locals[node.identifier] = alloca

    def generate_Expr_(self, node: Expr_, *, flag: int = 0) -> None:
        self.generate(node.expression)

    def generate_BinaryOp(self, node: BinaryOp, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        lhs = cast(ir.Constant, self.generate(node.lhs, flag=1))
        rhs = cast(ir.Constant, self.generate(node.rhs, flag=1))

        if lhs.type == rhs.type and (lhs.type in (ir.IntType(32), ir.IntType(8))):
            match node.op:
                case "+": return self.builder.add(lhs, rhs)
                case "-": return self.builder.sub(lhs, rhs)
                case "*": return self.builder.mul(lhs, rhs)
                case "/": return self.builder.sdiv(lhs, rhs)
                case "%": return self.builder.srem(lhs, rhs)

                case "<<": return self.builder.shl(lhs, rhs)
                case ">>": return self.builder.ashr(lhs, rhs)

                case "&": return self.builder.and_(lhs, rhs)
                case "^": return self.builder.xor(lhs, rhs)
                case "|": return self.builder.or_(lhs, rhs)

                # ? Do these work... or do we need another way of checking the "truthy-ness" of these values

                case "and":
                    return self.builder.icmp_signed("!=", self.builder.and_(lhs, rhs), ir.IntType(32)(0))

                case "xor":
                    return self.builder.icmp_signed("!=", self.builder.xor(lhs, rhs), ir.IntType(32)(0))

                case "or":
                    return self.builder.icmp_signed("!=", self.builder.or_(lhs, rhs), ir.IntType(32)(0))

        elif lhs.type == rhs.type == ir.FloatType():
            match node.op:
                case "+": return self.builder.fadd(lhs, rhs)
                case "-": return self.builder.fsub(lhs, rhs)
                case "*": return self.builder.fmul(lhs, rhs)
                case "/": return self.builder.fdiv(lhs, rhs)
                case "%": return self.builder.frem(lhs, rhs)

                case "and":
                    return self.builder.icmp_signed(
                        "!=",
                        self.builder.and_(
                            self.builder.fcmp_ordered("!=", lhs, ir.FloatType(0)),
                            self.builder.fcmp_ordered("!=", rhs, ir.FloatType(0)),
                        ), ir.IntType(0)
                    )

                case "xor":
                    return self.builder.icmp_signed(
                        "!=",
                        self.builder.xor(
                            self.builder.fcmp_ordered("!=", lhs, ir.FloatType(0)),
                            self.builder.fcmp_ordered("!=", rhs, ir.FloatType(0)),
                        ), ir.IntType(0)
                    )

                case "or":
                    return self.builder.icmp_signed(
                        "!=",
                        self.builder.or_(
                            self.builder.fcmp_ordered("!=", lhs, ir.FloatType(0)),
                            self.builder.fcmp_ordered("!=", rhs, ir.FloatType(0)),
                        ), ir.IntType(0)
                    )
        else:
            # TODO: Suggest casting - we don't support implicit casting
            llvm_generator_error(self.filename, node.location, f"Unsupported operation `{node.op}` for {lhs.type} and {rhs.type}")

    def generate_UnaryOp(self, node: UnaryOp, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        rhs = self.generate(node.rhs, flag=1)

        match node.op:
            case "-": return self.builder.neg(rhs)
            case "not": return self.builder.icmp_signed("!=", rhs, ir.IntType(32)(0))

            # ? Same question as logical operators in the binary op section...

    def generate_Variable(self, node: Variable, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        if self.current_namespace is not None:
            if (func := self.current_namespace.functions.get(node.var_name)) is not None:
                return func

            alloca = self.locals[node.var_name]
            if flag:
                return self.builder.load(alloca, name=node.var_name)
            return alloca

        if (ns := self.namespaces.get(node.var_name)) is not None:
            return ns

        if (func := self.functions.get(node.var_name)) is not None:
            return func

        alloca = self.locals[node.var_name]

        if flag:
            return self.builder.load(alloca, name=node.var_name)

        return alloca

    def generate_Assignment(self, node: Assignment, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        ptr = self.generate(node.target, flag=0)
        val = self.generate(node.value, flag=1)

        self.builder.store(val, ptr)

        return self.builder.load(ptr)

    def generate_Selection(self, node: Selection, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        cond_block = self.builder.append_basic_block("if.cond")
        true_block = self.builder.append_basic_block("if.true")
        else_block = self.builder.append_basic_block("if.else")
        post_block = self.builder.append_basic_block("if.post")

        self.builder.branch(cond_block)
        self.builder.position_at_start(cond_block)
        con = self.generate(node.condition, flag=1)
        self.builder.cbranch(con, true_block, else_block)
        self.builder.position_at_start(true_block)
        lhs = cast(ir.Constant, self.generate(node.if_true, flag=1))
        self.builder.branch(post_block)
        self.builder.position_at_start(else_block)
        rhs = cast(ir.Constant, self.generate(node.if_alt, flag=1))
        self.builder.branch(post_block)
        self.builder.position_at_start(post_block)
        phi = self.builder.phi(lhs.type)
        phi.add_incoming(lhs, true_block)
        phi.add_incoming(rhs, else_block)

        return phi

    def generate_ComparisonOp(self, node: ComparisonOp, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        lhs = cast(ir.Constant, self.generate(node.lhs, flag=1))
        rhs = cast(ir.Constant, self.generate(node.rhs, flag=1))

        if lhs.type == rhs.type and lhs.type in (ir.IntType(8), ir.IntType(32)):
            return self.builder.icmp_signed(node.op, lhs, rhs)
        elif lhs.type == rhs.type and lhs.type in (ir.FloatType(),):
            return self.builder.fcmp_ordered(node.op, lhs, rhs)
        else:
            # TODO: suggest casting
            llvm_generator_error(self.filename, node.location, f"Unsupported comparison `{node.op}` on types {lhs.type} and {rhs.type}")

    def generate_IntLiteral(self, node: IntLiteral, *, flag: int = 0) -> ir.Value:
        return ir.IntType(32)(node.int_value)

    def generate_FloatLiteral(self, node: FloatLiteral, *, flag: int = 0) -> ir.Value:
        return ir.FloatType()(node.float_value)

    def generate_StringLiteral(self, node: StringLiteral, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None
        typ = ir.ArrayType(ir.IntType(8), len(node.string_value))
        glob = ir.GlobalVariable(self.module, typ, f".str{self.strcount}")
        glob.initializer = ir.Constant(typ, bytearray(node.string_value.encode()))
        glob.global_constant = True

        return self.builder.gep(glob, (ir.IntType(32)(0), ir.IntType(32)(0)), inbounds=True)

    def generate_Call(self, node: Call, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        func = self.generate(node.callee, flag=1)
        args = [self.generate(arg, flag=1) for arg in node.callargs]

        return self.builder.call(func, args)

    def generate_ArrayLiteral(self, node: ArrayLiteral, *, flag: int = 0) -> ir.Value:
        return ir.Constant.literal_array([self.generate(value, flag=1) for value in node.arr_values])

    def generate_Index(self, node: Index, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        target = cast(ir.Constant, self.generate(node.target, flag=0))
        index = self.generate(node.index, flag=1)

        if target.type == ir.IntType(8).as_pointer().as_pointer():
            target = self.builder.load(target)
            value = self.builder.gep(target, (index,), inbounds=True)
            return self.builder.load(value)

        value = self.builder.gep(target, (ir.IntType(32)(0), index), inbounds=True)

        if flag == 1:
            return self.builder.load(value)

        return value

    def generate_Cast(self, node: Cast, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        totyp = self.generate(node.to)
        value = cast(ir.Constant, self.generate(node.value, flag=1))

        match (str(totyp), str(value.type)):
            case ("i32", "i8"): return self.builder.zext(value, totyp)
            case ("i8", "i32"): return self.builder.trunc(value, totyp)
            case ("i32", "float"): return self.builder.fptosi(value, totyp)
            case ("float", "i32"): return self.builder.sitofp(value, totyp)
            case _:
                llvm_generator_error(self.filename, node.location, f"Cast not supported! {value.type} to {totyp}")

    def generate_Attr(self, node: Attr, *, flag: int = 0) -> ir.Value:
        assert self.builder is not None

        # TODO: serious error handling here...

        target = cast(ir.Constant, self.generate(node.target, flag=0))

        if isinstance(target, Namespace):
            return target.get(node.attr)

        if self.current_namespace is not None:
            idx = self.current_namespace.structs[target.type.pointee.name.replace("struct.", "")].index(node.attr)
        else:
            idx = self.structures[target.type.pointee.name.replace("struct.", "")].index(node.attr)

        attr = self.builder.gep(target, (ir.IntType(32)(0), ir.IntType(32)(idx)), inbounds=True)

        if flag == 1:
            return self.builder.load(attr)

        return attr
