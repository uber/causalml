def handle_xgboost_error(exc: Exception):
    if type(exc).__name__ != "XGBoostError":
        raise exc

    raise RuntimeError(
        "xgboost is installed, but its native library could not be loaded. "
        "On macOS, this is commonly caused by a missing OpenMP runtime. "
        "Install it with `brew install libomp` or "
        "`conda install -c conda-forge llvm-openmp`, then retry."
    ) from exc
