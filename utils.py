
import json


def load_json_file(
        filepath: str
) -> dict:
    with open(filepath, 'r') as f:
        json_data = json.load(
            fp=f
        )

    return json_data


def store_json_file(
        json_data: dict,
        filepath: str
) -> None:
    with open(filepath, "w") as f:
        json.dump(
            obj=json_data,
            fp=f,
            indent=2
        )

        f.flush()


def load_universal_hyperparameters(

) -> dict:
    return load_json_file(
        filepath='universal_hyperparameters.json'
    )