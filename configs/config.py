import argparse
from pathlib import Path
import yaml

print("loading now")
# We only specify the yaml file from argparse and handle rest
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-f", "--config_file", default="configs/default.yaml",
                    help="Configuration file to load.")
ARGS = parser.parse_args()
print("loading done")

# Let's load the yaml file here
with open(ARGS.config_file, 'r') as f:
    config = yaml.load(f, yaml.FullLoader)
print(f"Loaded configuration file {ARGS.config_file}")

print(config)

def extern(func):
    """Wraps keyword arguments from configuration."""
    def wrapper(*args, **kwargs):
        """Injects configuration keywords."""
        # We get the file name in which the function is defined, ex: train.py
        fname = Path(func.__globals__['__file__']).name
        print(fname)
        # Then we extract arguments corresponding to the function name
        # ex: train.py -> load_data
        conf = config[fname][func.__name__]
        print(conf)
        # And update the keyword arguments with any specified arguments
        # if it isn't specified then the default values still hold
        conf.update(kwargs)