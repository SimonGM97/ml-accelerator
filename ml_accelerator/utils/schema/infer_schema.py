import pandas as pd
import os
import yaml
from typing import List


def infer_schema(
    df: pd.DataFrame,
    name: str,
    path: List[str]
) -> None:
    # Extract dtypes
    dtypes: pd.Series = df.dtypes

    # Define schema
    schema: dict = {
        "name": name,
        "path": path,
        "fields": [
            {
                "name": col_name,
                "type": dtypes[col_name],
                "mandatory": True,
                "nullable": True,
                "min_value": df[col_name].min() if dtypes[col_name] != 'object' else None,
                "max_value": df[col_name].max() if dtypes[col_name] != 'object' else None,
                "allowed_values": df[col_name].unique().tolist() if dtypes[col_name] == 'object' else None,
                "fillna_method": 'simple_imputer'

            } for col_name in dtypes.index
        ]
    }
    
    # Save schema
    with open(os.path.join('schemas', f'{name}_schema.yaml'), 'w') as file:
        yaml.dump(schema, file)


# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python ml_accelerator/utils/schema/infer_schema.py
if __name__ == "__main__":
    import seaborn as sns

    # Load dataset
    iris: pd.DataFrame = sns.load_dataset('iris')

    # Run infer_schema function
    infer_schema(
        df=iris,
        name='iris',
        path=['datasets', 'raw_data']
    )