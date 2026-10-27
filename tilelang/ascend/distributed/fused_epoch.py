"""Optional proof that changing epoch0 only translates protocol signal values.

This is a cache eligibility proof, not source validation. Unknown IR returns
None so the ordinary per-epoch planner retains its diagnostics and semantics.
The caller must still validate each launch interval, progress and bindings.
"""

from dataclasses import dataclass

from tvm import arith, ir, tirx
from tvm_ffi.container import Array, Map

from tilelang.layout.ascend import _AscendFractalLayout


_I32_MIN = -(2**31)
_I32_MAX = 2**31 - 1
_SIGNAL_VALUE = {"tl.distributed.put_signal": 2, "tl.distributed.signal": 1, "tl.distributed.wait_signal": 0}


@dataclass(frozen=True)
class EpochTranslationProof:
    """Inclusive base interval preserving every source int32 intermediate."""

    min_base: int
    max_base: int


class _Unknown(Exception):
    pass


def prove_epoch_translation(program, *, scalars=None, rank=0):
    """Prove a fixed scalar/rank specialization; return None if uncertain.

    Only int32 additions/subtractions carrying one copy of epoch0 are accepted.
    Epoch-dependent control, addresses, annotations, local arithmetic, casts
    and nonlinear expressions are deliberately outside this optimization.
    """
    try:
        return _Proof(program, dict(scalars or {}), rank).run()
    except _Unknown:
        return None


class _Proof:
    def __init__(self, program, scalars, rank):
        self.program = program
        self.analyzer = arith.Analyzer()
        self.bindings = {}
        self.defined = set(program.params)
        self.epoch = None
        self.minimum, self.maximum = 0, _I32_MAX - 1
        self.rank = rank
        for param in program.params:
            if param in program.buffer_map:
                continue
            if param.name == "epoch0":
                if param.dtype != "int32" or self.epoch is not None:
                    raise _Unknown
                self.epoch = param
            elif param.name in scalars:
                value = scalars[param.name]
                if isinstance(value, bool) or not isinstance(value, int) or param.dtype != "int32" or not _I32_MIN <= value <= _I32_MAX:
                    raise _Unknown
                self.bindings[param] = tirx.IntImm("int32", value)
            else:
                raise _Unknown
        if self.epoch is None or isinstance(rank, bool) or rank not in (0, 1):
            raise _Unknown

    def expanded(self, value):
        return tirx.stmt_functor.substitute(value, self.bindings)

    def contains_epoch(self, value):
        found = []
        tirx.stmt_functor.post_order_visit(
            value, lambda node: found.append(True) if isinstance(node, tirx.Var) and node.same_as(self.epoch) else None
        )
        return bool(found)

    def free(self, value):
        """Inspect metadata explicitly: ordinary TIR visitors omit annotations."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        if isinstance(value, (dict, Map, ir.DictAttrs)):
            for key, item in value.items():
                self.free(key)
                self.free(item)
            return
        if isinstance(value, (list, tuple, Array)):
            for item in value:
                self.free(item)
            return
        if isinstance(value, ir.Range):
            self.free(value.min)
            self.free(value.extent)
            return
        if isinstance(value, tirx.BufferRegion):
            self.free(value.buffer)
            self.free(value.region)
            return
        if isinstance(value, tirx.Buffer):
            self.free(value.data)
            self.free(value.shape)
            self.free(value.strides)
            self.free(value.elem_offset)
            return
        if isinstance(value, tirx.IterVar):
            self.free(value.dom)
            return
        if type(value) is _AscendFractalLayout:
            self.free(value.get_input_shape())
            self.free(value.get_forward_index())
            return
        if not isinstance(value, tirx.PrimExpr):
            raise _Unknown
        expanded = self.expanded(value)
        if self.contains_epoch(expanded):
            raise _Unknown

        def metadata(node):
            if isinstance(node, tirx.Call):
                self.free(node.annotations)
            elif isinstance(node, tirx.BufferLoad):
                self.free(node.buffer)

        tirx.stmt_functor.post_order_visit(expanded, metadata)

    def affine(self, value):
        """Return (epoch coefficient, int64 offset), checking intermediates."""
        if not self.contains_epoch(value):
            self.free(value)
            return 0, tirx.Cast("int64", value)
        if value.dtype != "int32":
            raise _Unknown
        if isinstance(value, tirx.Var) and value.same_as(self.epoch):
            return 1, tirx.IntImm("int64", 0)
        if not isinstance(value, (tirx.Add, tirx.Sub)):
            raise _Unknown
        lhs_coefficient, lhs = self.affine(value.a)
        rhs_coefficient, rhs = self.affine(value.b)
        subtract = isinstance(value, tirx.Sub)
        coefficient = lhs_coefficient - rhs_coefficient if subtract else lhs_coefficient + rhs_coefficient
        if coefficient != 1:
            raise _Unknown
        offset = lhs - rhs if subtract else lhs + rhs
        bound = self.analyzer.const_int_bound(offset)
        lower, upper = int(bound.min_value), int(bound.max_value)
        if lower <= -(2**63) or upper >= 2**63 - 1:
            raise _Unknown
        self.minimum = max(self.minimum, _I32_MIN - lower)
        self.maximum = min(self.maximum, _I32_MAX - upper)
        if self.minimum > self.maximum:
            raise _Unknown
        return coefficient, offset

    def walk(self, node):
        if isinstance(node, tirx.SeqStmt):
            for child in node.seq:
                self.walk(child)
        elif isinstance(node, tirx.Bind):
            # Rebinding could encode a loop-carried recurrence. This proof
            # only follows fresh scalar aliases, never mutable scalar state.
            if node.var in self.defined:
                raise _Unknown
            self.defined.add(node.var)
            value = self.expanded(node.value)
            if isinstance(value, tirx.Call) and value.op.name == "tl.distributed.rank":
                self.free(value)
                value = tirx.IntImm("int32", self.rank)
            if self.contains_epoch(value):
                if node.var.dtype != "int32":
                    raise _Unknown
                self.affine(value)
            else:
                self.free(value)
            # Scalar aliases are expanded before examining their uses. Vector
            # values cannot carry epoch after the check above and need no copy.
            if node.var.dtype == "int32":
                self.bindings[node.var] = value
        elif isinstance(node, tirx.For):
            if node.loop_var in self.defined:
                raise _Unknown
            self.defined.add(node.loop_var)
            self.free(node.min)
            self.free(node.extent)
            self.free(node.annotations)
            self.free(node.thread_binding)
            minimum, extent = self.expanded(node.min), self.expanded(node.extent)
            if self.analyzer.can_prove(extent == 0):
                return
            if not self.analyzer.can_prove(extent > 0):
                raise _Unknown
            self.analyzer.bind(node.loop_var, ir.Range.from_min_extent(minimum, extent))
            self.walk(node.body)
        elif isinstance(node, tirx.IfThenElse):
            self.free(node.condition)
            condition = self.expanded(node.condition)
            if self.analyzer.can_prove(condition):
                self.walk(node.then_case)
            elif self.analyzer.can_prove(~condition):
                if node.else_case is not None:
                    self.walk(node.else_case)
            else:
                original = dict(self.bindings)
                self.walk(node.then_case)
                then_bindings = self.bindings
                self.bindings = dict(original)
                if node.else_case is not None:
                    self.walk(node.else_case)
                if set(then_bindings) != set(self.bindings) or any(
                    not ir.structural_equal(value, self.bindings[key]) for key, value in then_bindings.items()
                ):
                    raise _Unknown
        elif isinstance(node, tirx.SBlockRealize):
            self.free(node.iter_values)
            self.free(node.predicate)
            self.walk(node.block)
        elif isinstance(node, tirx.SBlock):
            self.free(node.iter_vars)
            self.free(node.reads)
            self.free(node.writes)
            self.free(node.alloc_buffers)
            self.free(node.annotations)
            if node.match_buffers or node.init is not None:
                raise _Unknown
            self.walk(node.body)
        elif isinstance(node, tirx.AttrStmt):
            self.free(node.node)
            self.free(node.value)
            self.walk(node.body)
        elif isinstance(node, tirx.Evaluate):
            value = node.value
            name = getattr(getattr(value, "op", None), "name", "")
            if name in _SIGNAL_VALUE:
                index = _SIGNAL_VALUE[name]
                self.free(value.annotations)
                for position, arg in enumerate(value.args):
                    if position == index:
                        coefficient, _ = self.affine(self.expanded(arg))
                        if coefficient != 1:
                            raise _Unknown
                    else:
                        self.free(arg)
            else:
                self.free(value)
        elif isinstance(node, tirx.BufferStore):
            self.free(node.buffer)
            self.free(node.value)
            self.free(node.indices)
        else:
            raise _Unknown

    def run(self):
        self.free(self.program.attrs)
        for buffer in self.program.buffer_map.values():
            self.free(buffer)
        self.walk(self.program.body)
        return EpochTranslationProof(self.minimum, self.maximum)
