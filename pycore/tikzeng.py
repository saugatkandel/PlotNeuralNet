import os


def to_head(projectpath):
    pathlayers = os.path.join(projectpath, "layers/").replace("\\", "/")
    return (
        r"""
\documentclass[border=8pt, multi,  dvipsnames, tikz]{standalone} 
\usepackage{import}
\subimport{"""
        + pathlayers
        + r"""}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d} %for including external image 
\usetikzlibrary{calc}
"""
    )


def to_cor():
    return r"""
\def\ConvColor{Apricot}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{OrangeRed}
\def\UnpoolColor{SkyBlue}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}   
\def\SumColor{rgb:blue,5;green,15}
\def\UpsampColor{rgb:blue,2;green,1;black,0.3}
\def\ConvStrideColor{rgb,255:red,250;green,128;blue,114}
\def\ComplexConvColor{Emerald}
\def\ComplexConvStrideColor{MidnightBlue}

\def\ConvTransColor{Magenta}
\def\ComplexConvTransColor{LimeGreen}
"""


def to_begin():
    return r"""
\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
"""


# layers definition


def to_input(pathfile, to="(-3,0,0)", width=8, height=8, name="temp"):
    return (
        r"""
\node[canvas is zy plane at x=0] ("""
        + name
        + """) at """
        + to
        + """ {\includegraphics[width="""
        + str(width)
        + "cm"
        + """,height="""
        + str(height)
        + "cm"
        + """]{"""
        + pathfile
        + """}};
"""
    )


def get_conv_color(conv_type):
    conv_types_colors = {
        "real": r"\ConvColor",
        "real_stride": r"\ConvStrideColor",
        "complex": r"\ComplexConvColor",
        "complex_stride": r"\ComplexConvStrideColor",
        "real_transpose": r"\ConvTransColor",
        "complex_transpose": r"\ComplexConvTransColor",
    }
    return conv_types_colors[conv_type]


# Conv
def to_Conv(
    name,
    s_filter=256,
    n_filter=64,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=1,
    height=40,
    depth=40,
    conv_type="real",
    opacity=0.3,
    caption=" ",
):
    color = get_conv_color(conv_type)
    start = f"\pic[shift={{{offset}}}] at {to}"

    xin = f"{{{n_filter}, }}"
    box1 = f"name={name},caption={caption},xlabel={{{xin}}},zlabel={s_filter},fill={color},opacity={opacity}"
    box2 = f"height={height},width={width},depth={depth}"

    full_str = f"{start}{{Box={{{box1},{box2}}}}};"
    return full_str


def to_ConvConv(
    name,
    s_filter=256,
    n_filter=(64, 64),
    offset="(0,0,0)",
    to="(0,0,0)",
    width=(2, 2),
    height=40,
    depth=40,
    conv_type="real",
    opacity=0.8,
    caption=" ",
):
    color = get_conv_color(conv_type)
    start = f"\pic[shift={{{offset}}}] at {to}"

    xin = f"{{{n_filter[0]}, {n_filter[1]}}}"
    box1 = f"name={name},caption={caption},xlabel={{{xin}}},zlabel={s_filter},fill={color},opacity={opacity}"
    win = f"{{{width[0]}, {width[1]}}}"
    box2 = f"height={height},width={win},depth={depth}"

    full_str = f"{start}{{Box={{{box1},{box2}}}}};"
    return full_str


# Conv,relu
def to_ConvRelu(
    name, s_filter=256, n_filter=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" "
):
    return (
        r"""
\pic[shift={ """
        + offset
        + """ }] at """
        + to
        + """ 
    {RightBandedBox={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        xlabel={{ """
        + str(n_filter)
        + """, "" }},
        zlabel="""
        + str(s_filter)
        + """,
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


def to_ConvConv2(
    name,
    s_filter=256,
    n_filter=(64, 64),
    offset="(0,0,0)",
    to="(0,0,0)",
    width=(2, 2),
    height=40,
    depth=40,
    caption=" ",
):
    return (
        r"""
\pic[shift={ """
        + offset
        + """ }] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        xlabel={{ """
        + str(n_filter[0])
        + """, """
        + str(n_filter[1])
        + """ }},
        zlabel="""
        + str(s_filter)
        + """,
        fill=\ConvColor,
        height="""
        + str(height)
        + """,
        width={ """
        + str(width[0])
        + """ , """
        + str(width[1])
        + """ },
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


# Conv,Conv,relu
# Bottleneck
def to_ConvConvRelu(
    name,
    s_filter=256,
    n_filter=(64, 64),
    offset="(0,0,0)",
    to="(0,0,0)",
    width=(2, 2),
    height=40,
    depth=40,
    caption=" ",
):
    return (
        r"""
\pic[shift={ """
        + offset
        + """ }] at """
        + to
        + """ 
    {RightBandedBox={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        xlabel={{ """
        + str(n_filter[0])
        + """, """
        + str(n_filter[1])
        + """ }},
        zlabel="""
        + str(s_filter)
        + """,
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""
        + str(height)
        + """,
        width={ """
        + str(width[0])
        + """ , """
        + str(width[1])
        + """ },
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


# Pool
def to_Pool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.8, caption=" "):
    return (
        r"""
\pic[shift={ """
        + offset
        + """ }] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + r""",
        fill=\PoolColor,
        opacity="""
        + str(opacity)
        + """,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


# unpool4,
def to_UnPool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.8, caption=" "):
    return (
        r"""
\pic[shift={ """
        + offset
        + """ }] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + r""",
        caption="""
        + caption
        + r""",
        fill=\UnpoolColor,
        opacity="""
        + str(opacity)
        + """,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


def to_ConvRes(
    name,
    s_filter=256,
    n_filter=64,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=6,
    height=40,
    depth=40,
    opacity=0.2,
    caption=" ",
):
    return (
        r"""
\pic[shift={ """
        + offset
        + """ }] at """
        + to
        + """ 
    {RightBandedBox={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        xlabel={{ """
        + str(n_filter)
        + """, }},
        zlabel="""
        + str(s_filter)
        + r""",
        fill={rgb:white,1;black,3},
        bandfill={rgb:white,1;black,2},
        opacity="""
        + str(opacity)
        + """,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


# ConvSoftMax
def to_ConvSoftMax(name, s_filter=40, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        zlabel="""
        + str(s_filter)
        + """,
        fill=\SoftmaxColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


# SoftMax
def to_SoftMax(
    name, s_filter=10, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, opacity=0.8, caption=" "
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        xlabel={{" ","dummy"}},
        zlabel="""
        + str(s_filter)
        + """,
        fill=\SoftmaxColor,
        opacity="""
        + str(opacity)
        + """,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


def to_Sum(name, offset="(0,0,0)", to="(0,0,0)", radius=2.5, opacity=0.6):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {Ball={
        name="""
        + name
        + """,
        fill=\SumColor,
        opacity="""
        + str(opacity)
        + """,
        radius="""
        + str(radius)
        + """,
        logo=$+$
        }
    };
"""
    )


def to_connection(of, to):
    return (
        r"""
\draw [connection]  ("""
        + of
        + """-east)    -- node {\midarrow} ("""
        + to
        + """-west);
"""
    )


def to_skip(of, to, pos=1.25):
    return (
        r"""
\path ("""
        + of
        + """-southeast) -- ("""
        + of
        + """-northeast) coordinate[pos="""
        + str(pos)
        + """] ("""
        + of
        + """-top) ;
\path ("""
        + to
        + """-south)  -- ("""
        + to
        + """-north)  coordinate[pos="""
        + str(pos)
        + """] ("""
        + to
        + """-top) ;
\draw [copyconnection]  ("""
        + of
        + """-northeast)  
-- node {\copymidarrow}("""
        + of
        + """-top)
-- node {\copymidarrow}("""
        + to
        + """-top)
-- node {\copymidarrow} ("""
        + to
        + """-north);
"""
    )


def to_output(pathfile, to="(0,0,0)", width=8, height=8, name="temp"):
    return (
        r"""
    \node[canvas is zy plane at x=1.5] ("""
        + name
        + """) at """
        + to
        + """{\includegraphics[width="""
        + str(width)
        + "cm"
        + """,height="""
        + str(height)
        + "cm"
        + """]{"""
        + pathfile
        + """}};
"""
    )


def to_end():
    return r"""
\end{tikzpicture}
\end{document}
"""


def to_generate(arch, pathname="file.tex"):
    with open(pathname, "w") as f:
        for c in arch:
            f.write(c)
