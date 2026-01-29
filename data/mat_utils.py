from scipy.io import loadmat


def print_top_level_keys(mat_path: str):
    mat = loadmat(mat_path, simplify_cells=True)
    print([k for k in mat.keys() if not k.startswith("__")])


def print_key_info(mat_path: str, key: str):
    mat = loadmat(mat_path, simplify_cells=True)
    data = mat[key]
    print(f"`{key}` keys:", list(data.keys()))

    for k, v in data.items():
        t = type(v).__name__
        shp = getattr(v, "shape", None)
        dt  = getattr(v, "dtype", None)
        if isinstance(v, (str, int, float, bool)):
            extra = f"value={v}"
        elif isinstance(v, (list, tuple)):
            extra = f"len={len(v)}"
        elif shp is not None:
            extra = f"shape={shp} dtype={dt}"
        else:
            extra = ""
        print(f"{k:20s} {t:15s} {extra}")
