import yaml


def load_yaml_config(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def validate_config(config):
    defaults = {
        'deterministic': True,
        'seed': 42,
    }
    for k, v in defaults.items():
        if k not in config:
            config[k] = v
    return config
