from IPython import get_ipython


def skip(line, cell=None):
    """Skips execution of the current cell if cell contains
        - $$skip True
        - $$skip %var (var = True)
    """
    ip = get_ipython()

    condition = line.strip()

    if condition.startswith("$"):
        condition = condition[1:]

    should_skip = bool(eval(condition, ip.user_global_ns, ip.user_ns))

    if should_skip:
        return

    if cell is not None:
        return ip.run_cell(cell)


def load_ipython_extension(shell):
    """Registers the skip magic when the extension loads."""
    shell.register_magic_function(skip, magic_kind="line_cell", magic_name="skip")


def unload_ipython_extension(shell):
    """Unregisters the skip magic when the extension unloads."""
    shell.magics_manager.magics["line"].pop("skip", None)
    shell.magics_manager.magics["cell"].pop("skip", None)