from .tikzeng import *

# define new block
def block_ConvPool(name, bottom, top, s_filter=256, n_filter=64, offset="(1,0,0)", size=(32, 32, 3.5), opacity=0.5):
    return [
        to_ConvRelu(
            name=f"ccr_{name}",
            s_filter=str(s_filter),
            n_filter=n_filter,
            offset=offset,
            to=f"({bottom}-east)",
            width=size[2],
            height=size[0],
            depth=size[1],
        ),
        to_Pool(
            name=f"{top}",
            offset="(0,0,0)",
            to=f"(ccr_{name}-east)",
            width=1,
            height=size[0] - int(size[0] / 4),
            depth=size[1] - int(size[0] / 4),
            opacity=opacity,
        ),
        to_connection(f"{bottom}", f"ccr_{name}"),
    ]


def block_2ConvPool(name, bottom, top, s_filter=256, n_filter=64, offset="(1,0,0)", size=(32, 32, 3.5), opacity=0.5):
    return [
        to_ConvConvRelu(
            name=f"ccr_{name}",
            s_filter=str(s_filter),
            n_filter=(n_filter, n_filter),
            offset=offset,
            to=f"({bottom}-east)",
            width=(size[2], size[2]),
            height=size[0],
            depth=size[1],
        ),
        to_Pool(
            name=f"{top}",
            offset="(0,0,0)",
            to=f"(ccr_{name}-east)",
            width=1,
            height=size[0] - int(size[0] / 4),
            depth=size[1] - int(size[0] / 4),
            opacity=opacity,
        ),
        to_connection(f"{bottom}", f"ccr_{name}"),
    ]


def block_Unconv_no_res(
    name, bottom, top, s_filter=256, n_filter=64, offset="(1,0,0)", size=(32, 32, 3.5), opacity=0.5, n_conv_layers=1
):
    l1 = to_UnPool(
        name=f"unpool_{name}",
        offset=offset,
        to=f"({bottom}-east)",
        width=1,
        height=size[0],
        depth=size[1],
        opacity=opacity,
    )

    to = f"(unpool_{name}-east)"
    lconvs = []
    for n in range(n_conv_layers - 1):
        lthis = to_ConvRelu(
            name=f"{name}_{n}",
            offset="(0,0,0)",
            to=to,
            s_filter=str(s_filter),
            n_filter=str(n_filter),
            width=size[2],
            height=size[0],
            depth=size[1],
        )
        lconvs.append(lthis)
        to = f"({name}_{n}-east)"
    if n_conv_layers > 0:
        lthis = to_ConvRelu(
            name=f"{top}",
            offset="(0,0,0)",
            to=to,
            s_filter=str(s_filter),
            n_filter=str(n_filter),
            width=size[2],
            height=size[0],
            depth=size[1],
        )
        lconvs.append(lthis)
    lconn = to_connection(f"{bottom}", f"unpool_{name}")

    return [l1, *lconvs, lconn]


def block_Unconv(name, bottom, top, s_filter=256, n_filter=64, offset="(1,0,0)", size=(32, 32, 3.5), opacity=0.5):
    return [
        to_UnPool(
            name=f"unpool_{name}",
            offset=offset,
            to=f"({bottom}-east)",
            width=1,
            height=size[0],
            depth=size[1],
            opacity=opacity,
        ),
        to_ConvRes(
            name=f"ccr_res_{name}",
            offset="(0,0,0)",
            to=f"(unpool_{name}-east)",
            s_filter=str(s_filter),
            n_filter=str(n_filter),
            width=size[2],
            height=size[0],
            depth=size[1],
            opacity=opacity,
        ),
        to_Conv(
            name=f"ccr_{name}",
            offset="(0,0,0)",
            to=f"(ccr_res_{name}-east)",
            s_filter=str(s_filter),
            n_filter=str(n_filter),
            width=size[2],
            height=size[0],
            depth=size[1],
        ),
        to_ConvRes(
            name=f"ccr_res_c_{name}",
            offset="(0,0,0)",
            to=f"(ccr_{name}-east)",
            s_filter=str(s_filter),
            n_filter=str(n_filter),
            width=size[2],
            height=size[0],
            depth=size[1],
            opacity=opacity,
        ),
        to_Conv(
            name=f"{top}",
            offset="(0,0,0)",
            to=f"(ccr_res_c_{name}-east)",
            s_filter=str(s_filter),
            n_filter=str(n_filter),
            width=size[2],
            height=size[0],
            depth=size[1],
        ),
        to_connection(f"{bottom}", f"unpool_{name}"),
    ]


def block_Res(num, name, bottom, top, s_filter=256, n_filter=64, offset="(0,0,0)", size=(32, 32, 3.5), opacity=0.5):
    lys = []
    layers = [*[f"{name}_{i}" for i in range(num - 1)], top]
    for name in layers:
        ly = [
            to_Conv(
                name=f"{name}",
                offset=offset,
                to=f"({bottom}-east)",
                s_filter=str(s_filter),
                n_filter=str(n_filter),
                width=size[2],
                height=size[0],
                depth=size[1],
            ),
            to_connection(f"{bottom}", f"{name}"),
        ]
        bottom = name
        lys += ly

    lys += [
        to_skip(of=layers[1], to=layers[-2], pos=1.25),
    ]
    return lys
