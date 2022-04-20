import yaml


def load_config(path: str):
    with open(path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data