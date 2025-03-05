def get_path_loss(name: str | float | int):
    """Return the path loss model with the given name.

    Args:
    ---
    name: The name of the path loss model. The following models are available:
        - 'no_loss': No path loss.
        - 'free_space_loss' or 'free_space': Free space path loss.
        - float: Constant path loss in dB.
    """

    if name == "no_loss":
        from .path_loss import NoLoss
        return NoLoss()
    if name == "free_space_loss" or name == "free_space":
        from .free_space import FreeSpaceLoss
        return FreeSpaceLoss()
    if isinstance(name, (float, int)):
        from .path_loss import ConstantLoss
        return ConstantLoss(name)
    raise ValueError(f"Unknown path loss model: {name}")
