from .sampling import sample as dit 
from .scd import sample as scd
from ..config import Config


def load_sample_fct_from_config(config_path: str):
    print(f"loading {config_path}")
    cmodel = Config.from_yaml(config_path).model
    if cmodel.model_id == "dit":
        return dit
    elif cmodel.model_id == "scd":
        return scd
    else:
        raise ValueError(f"Invalid model type: {cmodel.model_id}")