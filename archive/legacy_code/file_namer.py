import time

def params_key(params) -> str:
    """Rounded param string embedded in shock-tube HDF5 filenames."""
    return "_".join(f"{round(float(p), 5)}" for p in params)


def generate_simulation_filename(
    problem: str,
    params: list[float],
    ext: str = "h5",
    run_tag: str = "wf_0"
) -> str:
    """
    Generate a canonical filename for simulation output files, with a hash
    to uniquely identify runs, and always append a timestamp to differentiate
    even repeated runs with identical params.

    Parameters:
        problem (str): The name of the problem ('cdr' or 'shock_tube').
        params (list[float]): List of simulation parameters.
        ext (str): File extension (default 'h5').

    Returns:
        str: Generated filename.
    """

    stem = problem
    params_str = params_key(params)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stem}__{params_str}__{run_tag}__{timestamp}.{ext}"


if __name__ == "__main__":
    print(generate_simulation_filename("cdr", [1.0, 2.0, 3.0], "h5"))
    print(generate_simulation_filename("shock_tube", [1.0, 2.0, 3.0, 4.0, 5.0], "h5"))
