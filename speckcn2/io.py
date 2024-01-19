import yaml


def load_config(config_file_path: str) -> dict:
    """Load the configuration file.

    Parameters
    ----------
    config_file_path : str
        Path to the .yaml configuration file

    Returns
    -------
    config : dict
        Dictionary containing the configuration
    """
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
