import sys

sys.path.append("/Users/skandel/code/PlotNeuralNet/")


from pycore.blocks import *
import dataclasses as dt
import copy
from functools import partial
import configs


@dt.dataclass
class NetworkState:
    size_filter: int = 64
    block_num: int = 0
    layer_num: int = 0
    prev_name: str = None
    prev_prev_name: str = None
    base_width: float = configs.BASE_WIDTH
    filter_widths: dict = dt.field(default_factory=lambda: configs.FILTER_WIDTHS)
    filter_heights: dict = dt.field(default_factory=lambda: configs.FILTER_HEIGHTS)
    opacity: float = configs.OPACITY
    offsets: dict = dt.field(default_factory=lambda: configs.OFFSETS)
    new_block: bool = False


def conv_box_params(
    state: NetworkState,
    n_filters: int,
    end_new_block: bool = False,
    offset_type: str = None,
    conv_type: str = None,
):
    print(f"Block {state.block_num} layer {state.layer_num} new_block {state.new_block}")

    match conv_type:
        case None:
            pass
        case "stride":
            state.size_filter = int(state.size_filter / 2)
            state.new_block = True
            state.layer_num = 0
            state.block_num += 1
        case "transpose":
            state.size_filter = int(state.size_filter * 2)
            state.new_block = True
            state.layer_num = 0
            state.block_num += 1
        case _:
            raise ValueError

    if state.new_block:
        if offset_type is None:
            offset_this = state.offsets["straight"]
        else:
            offset_this = state.offsets[offset_type]
    else:
        offset_this = state.offsets["no"]

    outs = {
        "opacity": state.opacity,
        "s_filter": state.size_filter,
        "height": state.filter_heights[state.size_filter],
        "offset": offset_this,
        "name": f"c_{state.block_num}_{state.layer_num}",
    }
    outs["depth"] = outs["height"]

    if state.prev_name is not None:
        outs["to"] = f"({state.prev_name}-east)"

    if isinstance(n_filters, int):
        outs["n_filter"] = n_filters
        outs["width"] = state.filter_widths[n_filters]
    else:
        if len(n_filters) > 2:
            raise ValueError
        outs["n_filter"] = n_filters
        outs["width"] = [state.filter_widths[n] for n in n_filters]

    if end_new_block:
        state.new_block = True
        state.layer_num = 0
        state.block_num += 1
    else:
        state.layer_num += 1
        state.new_block = False
    state.prev_prev_name = state.prev_name
    state.prev_name = outs["name"]

    print(outs)
    return outs


def connect(state: NetworkState):
    return to_connection(state.prev_prev_name, state.prev_name)


arch_init = [
    to_head(".."),
    to_cor(),
    to_begin(),
    # input
    to_input("../examples/fcn8s/cats.jpg", to="(-1.5, 0, 0)", height=4, width=4),
]

ns = NetworkState()

to_ConvS = partial(to_Conv, conv_type="real_stride")
to_ConvC = partial(to_Conv, conv_type="complex")
to_ConvCS = partial(to_Conv, conv_type="complex_stride")
to_ConvCT = partial(to_Conv, conv_type="complex_transpose")

convfn = partial(conv_box_params, ns)
conv_stride_fn = partial(conv_box_params, ns, conv_type="stride")
conv_trans_fn = partial(conv_box_params, ns, conv_type="transpose")
connect_lazy = partial(connect, ns)

arch_full = [
    # l1
    to_Conv(**convfn(32)),
    # l2
    to_ConvS(**conv_stride_fn(32)),
    connect_lazy(),
    to_Conv(**convfn(64)),
    # l3
    to_ConvS(**conv_stride_fn(64)),
    connect_lazy(),
    to_Conv(**convfn(128)),
    # l4
    to_ConvS(**conv_stride_fn(128)),
    connect_lazy(),
    to_ConvC(**convfn(112)),
    # l5
    to_ConvCT(**conv_trans_fn(112)),
    connect_lazy(),
    to_ConvC(**convfn(112)),
    # l6
    to_ConvCT(**conv_trans_fn(56)),
    connect_lazy(),
    to_ConvC(**convfn(56)),
    # l7
    to_ConvCT(**conv_trans_fn(28)),
    connect_lazy(),
    to_ConvC(**convfn(28)),
    # l8
    to_ConvCT(**conv_trans_fn(14)),
    connect_lazy(),
    to_ConvC(**convfn(1)),
    to_output("../examples/fcn8s/cats.jpg", to=f"({ns.prev_name}-east)", height=5, width=5),
]
arch = [
    *arch_init,
    *arch_full,
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    # print(arch)
    # print(len(arch))
    # raise
    to_generate(arch, namefile + ".tex")


if __name__ == "__main__":
    main()
