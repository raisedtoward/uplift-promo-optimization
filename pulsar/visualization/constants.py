DEFAULT_PLOT_STYLE = "fivethirtyeight"

TREATMENT_COLOR = "lightcoral"
CONTROL_COLOR = "steelblue"
DEFAULT_COLOR = "mediumpurple"

VARIANTS_COLORS = [
    CONTROL_COLOR,
    TREATMENT_COLOR,
    "mediumpurple",
    "mediumseagreen",
    "gold",
    "dodgerblue",
    "crimson",
    "darkorchid",
    "forestgreen",
    "yellow",
    "lightskyblue",
    "teal",
    "orange",
    "magenta",
]

LINESTYLES = [
    "solid",
    "dashed",
    "dotted",
    "dashdot",
]

MARKERS = [
    "o",
    "d",
    "*",
    "s",
    "^",
    "v",
]

# https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
DIST_KWS = {
    "kde": True,
    "alpha": 0.5,
    # "bins": "auto",
    # "bins": 50,
    "bins": "scott",
    "stat": "density",
    "kde_kws": {"cut": 3},
}
