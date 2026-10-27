"""Compiler-side contracts for the bounded, synchronous SHMEM task lowering.

Source TD validation and finite-plan protocol checks run before this module.
These checks protect the internal ABI and resource contract through the normal
Ascend scheduler and multibuffer pipeline; they do not certify physical links.
"""

from tvm import DataType, arith, ir, tirx


_TASK = "tl.ascend_distributed_task"


def _name(call):
    return getattr(call.op, "name", "")


def _error(message):
    raise ValueError("[Ascend distributed fusion] " + message)


def task_inventory(mod):
    inventory = []
    for func in mod.functions.values():
        if not isinstance(func, tirx.PrimFunc):
            continue

        def visit(node):
            if isinstance(node, tirx.Call) and _name(node) == _TASK:
                if len(node.args) < 8 or not isinstance(node.args[0], tirx.IntImm) or not 0 <= int(node.args[0]) <= 5:
                    _error("malformed synchronous task ABI (kind, dst, src, signal, scratch, nbytes, peer, epoch)")
                task_id = node.annotations.get("td.task_id")
                if not isinstance(task_id, tirx.IntImm):
                    _error("every task must carry its stable td.task_id")
                inventory.append(node)

        tirx.stmt_functor.post_order_visit(func.body, visit)
    return tuple(inventory)


def verify_fused_tasks(mod, *, auto_schedule=True):
    """Validate launch geometry and the unique scratch before pointer lowering."""
    inventory = task_inventory(mod)
    if not inventory:
        return inventory
    ids = [int(task.annotations["td.task_id"]) for task in inventory]
    if len(set(ids)) != len(ids):
        _error("communication task IDs must be unique")
    if not auto_schedule:
        _error("fused communication requires AutoSchedule and InsertSync; disabling scheduling is unsupported")
    for func in mod.functions.values():
        if not isinstance(func, tirx.PrimFunc):
            continue
        tasks, launches, scratch_uses, loads, stores = [], [], [], [], []

        def visit(node, tasks=tasks, launches=launches, scratch_uses=scratch_uses, loads=loads, stores=stores):
            if isinstance(node, tirx.Call) and _name(node) == _TASK:
                tasks.append(node)
            if isinstance(node, tirx.BufferStore):
                stores.append(node)
            if isinstance(node, tirx.BufferLoad):
                loads.append(node)
            if isinstance(node, tirx.Call) and _name(node) == "tl.access_ptr":
                scratch_uses.append(node)
            if isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent":
                launches.append(node)

        tirx.stmt_functor.post_order_visit(func.body, visit)
        if not tasks:
            continue
        for launch in launches:
            if launch.node.thread_tag in ("blockIdx.x", "cthread") and (
                not isinstance(launch.value, tirx.IntImm) or int(launch.value) != 1
            ):
                _error("P0 permits one block and exactly one active AIV (MixedKernel(1, sids=1))")
        scratch = None
        for task in tasks:
            pointer = task.args[4]
            if not isinstance(pointer, tirx.Call) or _name(pointer) != "tl.access_ptr":
                _error("scratch must retain a structured tl.access_ptr until scheduling finishes")
            load, extent, mask = pointer.args
            if not isinstance(load, tirx.BufferLoad):
                _error("scratch pointer must name one UB allocation")
            buffer = load.buffer
            if buffer.scope() not in ("shared", "shared.dyn") or buffer.dtype != "uint8" or tuple(int(x) for x in buffer.shape) != (4096,):
                _error("scratch must be exactly one unversioned 4096-byte uint8 UB allocation")
            if any(not isinstance(x, tirx.IntImm) or int(x) != 0 for x in load.indices):
                _error("scratch must start at its allocation base")
            if not isinstance(extent, tirx.IntImm) or int(extent) != 4096 or int(mask) != 3:
                _error("every communication task must declare read/write access to the entire scratch")
            if scratch is not None and not buffer.data.same_as(scratch):
                _error("all communication tasks must share one exclusive scratch and issuer")
            scratch = buffer.data
            kind, nbytes = int(task.args[0]), task.args[5]
            if not isinstance(nbytes, tirx.IntImm):
                _error("P0 transfer sizes must be static")
            length = int(nbytes)
            if kind <= 2 and (length <= 0 or length % 32 or length >= 2**32):
                _error("P0 MTE transfer sizes must be positive multiples of 32 bytes below 2**32")
            if kind >= 3 and length != 0:
                _error("signal/wait/drain tasks cannot carry a payload length")
        uses = sum(
            isinstance(pointer.args[0], tirx.BufferLoad) and pointer.args[0].buffer.data.same_as(scratch) for pointer in scratch_uses
        )
        if uses != len(tasks) or any(store.buffer.data.same_as(scratch) for store in stores):
            _error("communication scratch cannot be borrowed by local computation")
        allowed_loads = [task.args[4].args[0] for task in tasks]
        if any(load.buffer.data.same_as(scratch) and not any(load.same_as(allowed) for allowed in allowed_loads) for load in loads):
            _error("communication scratch cannot be read by local computation")
        if int(tasks[-1].args[0]) != 5:
            _error("the fused issuer must end with an explicit drain task")
    return inventory


def verify_fused_inventory(mod, expected, *, structured=False):
    """Check ordered protocol actions and their ABI through backend rewriting.

    Before pointer lowering, every memory operand must remain structurally the
    same: this includes acquire/release effects and scratch ownership. After
    flattening and allocation merging, pointer syntax necessarily changes; the
    final check retains action order, pointer-operand count and scalar ABI.
    """
    actual = task_inventory(mod)
    if len(actual) != len(expected):
        _error("a backend transform deleted or duplicated a communication action")
    analyzer = arith.Analyzer()
    for before, after in zip(expected, actual):
        if int(before.annotations["td.task_id"]) != int(after.annotations["td.task_id"]):
            _error("a backend transform reordered the issuer's communication actions")
        if len(before.args) != len(after.args):
            _error("a backend transform changed the acquire/release effect operands")
        for index in (0, 5, 6, 7):
            left, right = before.args[index], after.args[index]
            if not structured and index == 7:
                # SplitHostDevice creates a new formal parameter for epoch0.
                # P0 has exactly one dynamic scalar in this ABI; accept only
                # that formal's identity change, never a different expression.
                old_vars, new_vars = [], []
                tirx.stmt_functor.post_order_visit(left, lambda n, result=old_vars: result.append(n) if isinstance(n, tirx.Var) else None)
                tirx.stmt_functor.post_order_visit(right, lambda n, result=new_vars: result.append(n) if isinstance(n, tirx.Var) else None)
                if len(old_vars) == len(new_vars) == 1 and all(v.name == "epoch0" and v.dtype == "int32" for v in [*old_vars, *new_vars]):
                    right = tirx.stmt_functor.substitute(right, {new_vars[0]: old_vars[0]})
            if not analyzer.can_prove_equal(left, right):
                _error("a backend transform changed the communication scalar ABI or epoch")
        if structured:
            for index in (*range(1, 5), *range(8, len(before.args))):
                if not ir.structural_equal(before.args[index], after.args[index]):
                    _error("scheduling or multibuffering changed a communication memory region")


def verify_fused_storage(mod):
    """Check final physical scratch separation after allocation/pointer lowering."""
    for func in mod.functions.values():
        if not isinstance(func, tirx.PrimFunc):
            continue
        tasks, pointers, loops, allocations = [], [], [], []

        def visit(node, tasks=tasks, pointers=pointers, loops=loops, allocations=allocations):
            if isinstance(node, tirx.Call):
                if _name(node) == _TASK:
                    tasks.append(node)
                elif _name(node) == "tirx.tvm_access_ptr":
                    pointers.append(node)
            elif isinstance(node, tirx.For):
                loops.append(node)
            elif isinstance(node, tirx.AllocBuffer):
                allocations.append(node.buffer)

        tirx.stmt_functor.post_order_visit(func.body, visit)
        if not tasks:
            continue
        analyzer = arith.Analyzer()
        for loop in reversed(loops):
            analyzer.bind(loop.loop_var, ir.Range.from_min_extent(loop.min, loop.extent))
        scratch_pointers = [task.args[4] for task in tasks]
        first = scratch_pointers[0]
        if not isinstance(first, tirx.Call) or _name(first) != "tirx.tvm_access_ptr":
            _error("final scratch must retain a bounded physical access pointer")
        _, storage, start, extent, mask = first.args
        if first.args[0].dtype != "uint8" or not analyzer.can_prove_equal(extent, 4096) or int(mask) != 3:
            _error("final scratch ABI must preserve its 4096-byte read/write region")
        for pointer in scratch_pointers[1:]:
            if not ir.structural_equal(pointer, first):
                _error("final communication tasks no longer share the same scratch")
        allocation = next((buffer for buffer in allocations if buffer.data.same_as(storage)), None)
        if allocation is None:
            _error("final communication scratch must be compiler-owned UB storage")
        allocation_bytes = DataType(allocation.dtype).bits // 8
        for dimension in allocation.shape:
            allocation_bytes *= dimension
        if not analyzer.can_prove(start >= 0) or not analyzer.can_prove(start + extent <= allocation_bytes):
            _error("final communication scratch is outside its physical allocation")
        for pointer in pointers:
            if any(pointer.same_as(scratch) for scratch in scratch_pointers):
                continue
            if not pointer.args[1].same_as(storage):
                continue
            element_bytes = DataType(pointer.args[0].dtype).bits // 8
            other_start = pointer.args[2] * element_bytes
            other_end = (pointer.args[2] + pointer.args[3]) * element_bytes
            if not analyzer.can_prove((other_end <= start) | (other_start >= start + extent)):
                _error("local computation overlaps the reserved physical communication scratch")
