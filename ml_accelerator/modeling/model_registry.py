from ml_accelerator.config.params import Params
from ml_accelerator.data_processing.data_cleaning import DataCleaner
from ml_accelerator.data_processing.data_transforming import DataTransformer
from ml_accelerator.modeling.models.model import Model
from ml_accelerator.modeling.models.classification_model import ClassificationModel
from ml_accelerator.modeling.models.regression_model import RegressionModel
from ml_accelerator.pipeline.ml_pipeline import MLPipeline
from ml_accelerator.utils.aws.s3_helper import (
    load_from_s3,
    save_to_s3,
    find_keys,
    delete_from_s3
)
from ml_accelerator.utils.filesystem.filesystem_helper import (
    load_from_filesystem,
    save_to_filesystem
)
from ml_accelerator.utils.logging.logger_helper import get_logger
from tqdm import tqdm
import os
from pprint import pformat
from typing import List, Dict

import warnings

# Suppress only SyntaxWarning
warnings.filterwarnings("ignore", category=SyntaxWarning)


# Get logger
LOGGER = get_logger(
    name=__name__,
    level=Params.LEVEL,
    txt_fmt=Params.TXT_FMT,
    json_fmt=Params.JSON_FMT,
    filter_lvls=Params.FILTER_LVLS,
    log_file=Params.LOG_FILE,
    backup_count=Params.BACKUP_COUNT
)


class ModelRegistry:
    """
    Class designed to organize, manage & update model repositories in a centralized fashion.
    """
    
    def __init__(
        self,
        n_candidates: int = Params.N_CANDIDATES,
        task: str = Params.TASK,
        data_storage_env: str = Params.DATA_STORAGE_ENV,
        model_storage_env: str = Params.MODEL_STORAGE_ENV,
        bucket: str = Params.BUCKET,
        models_path: List[str] = Params.MODELS_PATH
    ) -> None:
        """
        Initialize the ModelRegistry

        :param `n_candidates`: (int) Number of candidate models that will be considered before 
         defining the production model.
        :param `intervals`: (str) Time between predictions.
        """
        # General Params
        self.n_candidates: int = n_candidates
        self.task: str = task

        # Storage Params
        self.data_storage_env: str = data_storage_env
        self.model_storage_env: str = model_storage_env
        self.bucket: str = bucket
        self.models_path: List[str] = models_path

        # Define default models
        self.prod_model: Model = None
        self.staging_models: List[Model] = None
        self.dev_models: List[Model] = None

        # Define self.registry_dict
        self.registry_dict: Dict[str, List[str]] = {
            "production": [], 
            "staging": [], 
            "development": []
        }

        self.load_registry_dict()

    """
    Model Loading Methods
    """

    def load_model(
        self, 
        model_id: str,
        light: bool = False
    ) -> Model:
        try:
            # Instanciate Model
            if self.task in ['binary_classification', 'multiclass_classification']:
                model = ClassificationModel(model_id=model_id)
            elif self.task == 'regression':
                model = RegressionModel(model_id=model_id)
            else:
                raise NotImplementedError(f'Task "{self.task}" has not been implemented yet.\n')

            # Load Model
            model.load(light=light)

            return model
        except Exception as e:
            LOGGER.error(
                'Unable to load Model %s.\n'
                'Exception: %s\n',
                model_id, e
            )
            return

    def load_dev_models(
        self,
        light: bool = False
    ) -> List[Model]:
        # Load Staging Models
        dev_models: List[Model] = []

        if not light:
            LOGGER.info('Loading dev models:')

        for model_id in tqdm(self.registry_dict['development'], disable=light):
            dev_models.append(self.load_model(
                model_id=model_id,
                light=light
            ))
        
        return [model for model in dev_models if model is not None]
    
    def load_staging_models(
        self,
        light: bool = False
    ) -> List[Model]:
        # Load Staging Models
        stage_models: List[Model] = []

        if not light:
            LOGGER.info('Loading staging models:')

        for model_id in tqdm(self.registry_dict['staging'], disable=light):
            stage_models.append(self.load_model(
                model_id=model_id,
                light=light
            ))
        
        return [model for model in stage_models if model is not None]
    
    def load_prod_model(
        self,
        light: bool = False
    ) -> Model:
        """
        Method for the production model.

        :return: (Model) Production model.
        """
        if not light:
            LOGGER.info('Loading prod model')

        if len(self.registry_dict['production']) > 0:
            # Find production model_id 
            model_id = self.registry_dict['production'][0]

            # Load and return production model
            return self.load_model(
                model_id=model_id,
                light=light
            )
        return
    
    """
    Pipeline Loading Methods
    """

    def load_pipeline(
        self,
        model_id: str
    ) -> MLPipeline:
        try:
            # Instanciate Transformers
            DC: DataCleaner = DataCleaner()
            DT: DataTransformer = DataTransformer()

            # Instanciate Estimator
            if self.task in ['binary_classification', 'multiclass_classification']:
                model = ClassificationModel(model_id=model_id)
            elif self.task == 'regression':
                model = RegressionModel(model_id=model_id)
            else:
                raise NotImplementedError(f'Task "{self.task}" has not been implemented yet.\n')
            
            # Instanciate MLPipeline
            MLP: MLPipeline = MLPipeline(DC=DC, DT=DT, model=model)

            # Load MLPipeline
            MLP.load()

            return MLP
        except Exception as e:
            LOGGER.warning(
                'Unable to load %s MLPipeline.\n'
                'Exception: %s.',
                model_id, e
            )
            return

    def load_dev_pipes(self) -> List[MLPipeline]:
        # Load Staging MLPipelines
        dev_pipes: List[MLPipeline] = []

        LOGGER.info('Loading dev pipelines:')
        for model_id in tqdm(self.registry_dict['development']):
            dev_pipes.append(self.load_pipeline(model_id=model_id))
        
        return [pipe for pipe in dev_pipes if pipe is not None]
    
    def load_staging_pipes(self) -> List[Model]:
        # Load Staging MLPipelines
        stage_pipes: List[MLPipeline] = []

        LOGGER.info('Loading staging pipelines:')
        for model_id in tqdm(self.registry_dict['staging']):
            stage_pipes.append(self.load_pipeline(model_id=model_id))
        
        return [pipe for pipe in stage_pipes if pipe is not None]
    
    def load_prod_pipe(self) -> MLPipeline:
        """
        Method for the production pipeline.

        :return: (MLPipeline) Production MLPipeline.
        """
        LOGGER.info('Loading prod pipeline')
        if len(self.registry_dict['production']) > 0:
            # Find production model_id
            model_id = self.registry_dict['production'][0]

            # Load and return production pipeline
            return self.load_pipeline(model_id=model_id)
        return
    
    """
    Registry Methods
    """

    def update_model_stage(
        self,
        model: Model,
        new_stage: str = None
    ) -> None:
        # Define initial stage
        initial_stage = model.stage

        # Check that new stage is different from actual stage
        if initial_stage == new_stage:
            return
        
        # Update model stage
        model.stage = new_stage

        # Update self.registry_dict
        try:
            self.registry_dict[initial_stage].remove(model.model_id)
        except Exception as e:
            LOGGER.warning(
                'Unable to extract model_id: %s from self.registry["%s"]: %s\n'
                'Exception: %s',
                model.model_id, initial_stage, self.registry_dict[initial_stage], e
            )
        try:
            if new_stage != 'delete':
                if new_stage == 'production':
                    self.registry_dict['production'] = [model.model_id]
                else:
                    self.registry_dict[new_stage].append(model.model_id)
        except Exception as e:
            LOGGER.warning(
                'Unable to extract model_id: %s from self.registry["%s"]: %s\n'
                'Exception: %s',
                model.model_id, new_stage, self.registry_dict[new_stage], e
            )

        # Update self.prod_model, self.staging_models & self.development_models
        try:
            if initial_stage == 'development':
                self.dev_models.remove(model)
            elif initial_stage == 'staging':
                self.staging_models.remove(model)
            elif initial_stage == 'production':
                self.prod_model = None
        except Exception as e:
            LOGGER.warning(
                'Unable to remove %s from %s models.\n'
                'Exception: %s',
                model.model_id, initial_stage, e
            )
        
        try:
            if new_stage == 'development':
                self.dev_models.append(model)
            elif new_stage == 'staging':
                self.staging_models.append(model)
            elif new_stage == 'production':
                self.prod_model = model
        except Exception as e:
            LOGGER.warning(
                'Unable to add %s to %s models.\n'
                'Exception: %s',
                model.model_id, new_stage, e
            )

    def debug(self):
        if self.prod_model is not None:
            prod_test_score: float = self.prod_model.test_score
            prod_optim_metric: str = self.prod_model.optimization_metric
        else:
            prod_test_score: float = None
            prod_optim_metric: str = None

        LOGGER.debug(
            'MLRegistry:\n'
            '%s\n'
            'Dev Models: %s\n'
            'Staging Models: %s\n'
            'Champion performance: %s (%s)\n'
            '--------------------------------------------------------------------------\n',
            pformat(self.registry_dict), len(self.dev_models), len(self.staging_models), 
            prod_test_score, prod_optim_metric
        )

    def update_model_stages(
        self,
        update_prod_model: bool = False,
        debug: bool = False
    ) -> None:
        """
        Method that will re-define model stages, applying the following logic:
            - Top n_candidate dev models will be promoted as "staging" models (also referred as "challenger" models),
              based on their mean cross validation performance.
            - The top staging model will compete with the production model (also referred as "champion" model), 
              based on their test performance.
        """
        # Load light Models
        self.prod_model: Model = self.load_prod_model(light=True)
        self.staging_models: List[Model] = self.load_staging_models(light=True)
        self.dev_models: List[Model] = self.load_dev_models(light=True)

        # Assert that all dev_models & staging_models are not None
        assert not(any([m is None for m in self.dev_models + self.staging_models]))

        # Downgrade all models to development (except for prod_model)
        for model in self.staging_models + self.dev_models:
            self.update_model_stage(
                model=model,
                new_stage='development'
            )

        # Sort Dev Models (based on validation performance)
        self.dev_models: List[Model] = self.sort_models(
            models=self.dev_models,
            by='val_score'
        )

        if debug:
            self.debug()

        # Test & promote models from staging_candidates
        for model in self.dev_models[:self.n_candidates]:
            # Diagnose model
            diagnostics_dict = model.diagnose()

            if not self.needs_repair(diagnostics_dict):
                # Promote Model
                self.update_model_stage(
                    model=model,
                    new_stage='staging'
                )
            else:
                LOGGER.warning(
                    '%s was NOT pushed to Staging.\n'
                    'diagnostics_dict:\n%s\n',
                    model.model_id, pformat(diagnostics_dict)
                )

        # Sort Staging Models (based on test performance)
        self.staging_models: List[Model] = self.sort_models(
            models=self.staging_models,
            by='test_score'
        )

        # Assert that there are no more staging models than expected
        assert (
            len(self.staging_models) <= self.n_candidates, 
            f"There are {len(self.staging_models)} staging models, but a maximum of {self.n_candidates} are allowed."
        )

        if debug:
            self.debug()

        # Find forced model ID
        forced_model_id: str = self.load_forced_model()['forced_model_id']

        if debug:
            LOGGER.debug('Foreced Model ID: %s', pformat(forced_model_id))

        # Update prod mdoel with forced model
        if forced_model_id is not None:
            LOGGER.warning('Forced model was detected: %s.', forced_model_id)

            # Check if forced model is the same as current prod model
            if self.prod_model is not None and forced_model_id == self.prod_model.model_id:
                LOGGER.info(f'Forced Model is the same as current Production Model.\n')
            else:
                # Find new prod model
                new_prod_model: Model = self.load_model(
                    model_id=forced_model_id,
                    light=False
                )

                if new_prod_model is None:
                    LOGGER.warning('Forced Model was not found in current models!')
                else:
                    # Downgrade current prod model
                    self.update_model_stage(
                        model=self.prod_model,
                        new_stage='staging'
                    )

                    # Promote new prod model
                    self.update_model_stage(
                        model=new_prod_model,
                        new_stage='production'
                    )

        # Define default prod model if current prod model is None
        if self.prod_model is None:
            LOGGER.warning(
                'There was no previous production model.\n'
                'Therefore, a new provisory production model will be chosen.\n'
            )
            
            # Find new production model
            new_prod_model: Model = self.sort_models(
                models=self.staging_models,
                by='test_score'
            )[0]

            # Promote new prod model
            self.update_model_stage(
                model=new_prod_model,
                new_stage='production'
            )

            LOGGER.info('New champion model:\n%s\n', self.prod_model)

        elif update_prod_model:
            # Pick Challenger
            challenger: Model = self.sort_models(
                models=self.staging_models,
                by='test_score'
            )[0]

            if challenger.test_score > self.prod_model.test_score:
                LOGGER.info(
                    'New production model mas found: %s - [%s: %s]\n'
                    'Previous production model:  %s - [%s: %s]',
                    challenger.model_id, challenger.optimization_metric, challenger.test_score,
                    self.prod_model.model_id, self.prod_model.optimization_metric, self.prod_model.test_score
                )

                # Downgrade current prod model
                self.update_model_stage(
                    model=self.prod_model,
                    new_stage='staging'
                )

                # Promote new challenger
                self.update_model_stage(
                    model=challenger,
                    new_stage='production'
                )

        # Delete unwanted models
        delete_models = self.sort_models(
            models=self.dev_models,
            by='val_score'
        )[5:]

        for model in delete_models:
            self.update_model_stage(
                model=model,
                new_stage='delete'
            )

        # Sort self.registry_dict
        self.registry_dict: Dict[str, List[str]] = {
            "production": [self.prod_model.model_id], 
            "staging": [m.model_id for m in self.staging_models], 
            "development": [m.model_id for m in self.dev_models]
        }

        if debug:
            self.debug()
        
        # Clean registry
        self.clean_registry()

        # Save self.registry_dict
        self.save_registry_dict()

        # Save models
        self.save_models()

    """
    Cleaning Methods
    """

    def clean_registry(self) -> None:
        """
        Method that will remove any "inactive" model or experiment from the file system, mlflow tracking
        server and mlflow model registry.
        An "inactive" model is defined as a model that cannot be tagged to any current development, staging 
        or production model.
        """
        if self.model_storage_env == 'filesystem':
            # Clean file_system
            self.clean_filesystem_models()

        elif self.model_storage_env == 'S3':
            # Clean S3
            self.clean_s3_models()

        elif self.model_storage_env == 'ml_flow':
            # Clean tracking server
            self.clean_tracking_server()

            # Clean mlflow registry
            self.clean_mlflow_registry()

    def clean_filesystem_models(self) -> None:
        """
        Method that will remove any "inactive" model from the file system.
        """
        # Load model ids
        model_ids: List[str] = (
            self.registry_dict["production"] +
            self.registry_dict["staging"] +
            self.registry_dict["development"]
        )

        for root, directories, files in os.walk(
            os.path.join(self.bucket, *self.models_path)
        ):
            for file in files:
                model_id = file.split('_')[0]
                if model_id not in model_ids and not file.startswith('.'):
                    delete_path = os.path.join(self.bucket, *self.models_path, model_id, file)
                    LOGGER.info("Deleting %s.", delete_path)
                    try:
                        os.remove(delete_path)
                    except Exception as e:
                        LOGGER.warning(
                            'Unable to remove %s.\n'
                            'Exception: %s',
                            delete_path, e
                        )

    def clean_s3_models(self) -> None:
        # Load model ids
        model_ids: List[str] = (
            self.registry_dict["production"] +
            self.registry_dict["staging"] +
            self.registry_dict["development"]
        )

        for key in find_keys(
            bucket=self.bucket, 
            subdir='/'.join(self.models_path)
        ):
            if not any([model_id in key for model_id in model_ids]):
                LOGGER.info("Deleting %s/%s.", self.bucket, key)
                delete_from_s3(path=f"{self.bucket}/{key}")

    def clean_tracking_server(self) -> None:
        pass

    def clean_mlflow_registry(self) -> None:
        pass
    
    """
    Utils Methods
    """

    @staticmethod
    def sort_models(
        models: List[Model],
        by: str = 'test_score'
    ) -> List[Model]: 
        if by not in ['val_score', 'test_score']:
            LOGGER.critical('Invalid "by" parameter: %s', by)
            raise Exception(f'Invalid "by" parameter: {by}.\n')
        
        def sort_fun(model: Model):
            score: float = getattr(model, by)
            if score is not None:
                return score
            return 0
        
        models.sort(key=sort_fun, reverse=True)

        return models

    @staticmethod
    def needs_repair(d: dict) -> bool:
        for value in d.values():
            if value:
                return True
        return False

    def find_repeated_models(
        self,
        new_model: Model, 
        models: List[Model] = None
    ) -> List[Model]:
        # Validate models
        if models is None:
            # Extract models
            model_ids = (
                self.registry_dict['production'] 
                + self.registry_dict['staging'] 
                + self.registry_dict['development']
            )
            models = [self.load_model(model_id=model_id) for model_id in model_ids]

        def extract_tuple_attrs(model: Model):
            # Define base attrs to add
            attrs = {'algorithm': model.algorithm}

            # Add hyperparameters
            attrs.update(model.hyper_parameters)

            return tuple(attrs.items())

        # Extract new tuple attrs
        new_tuple_attrs = extract_tuple_attrs(new_model)

        # Define repeated Models
        repeated_models: List[Model] = []
        for model in models:
            if new_tuple_attrs == extract_tuple_attrs(model):
                repeated_models.append(model)

        # Add repeated_models, by looking at model_id
        for model in models:
            if new_model.model_id == model.model_id:
                repeated_models.append(model)

        return repeated_models

    def drop_duplicate_models(
        self,
        models: List[Model] = None,
        score_to_prioritize: str = 'test_score'
    ) -> List[Model]:
        # Validate models
        if models is None:
            # Extract models
            model_ids = (
                self.registry_dict['production'] 
                + self.registry_dict['staging'] 
                + self.registry_dict['development']
            )
            models = [self.load_model(model_id=model_id) for model_id in model_ids]
        
        # Find repeated models
        repeated_models_dict: Dict[Model, List[Model]] = {}

        for model in models:
            # Extract idx
            model_idx = models.index(model)

            # Add repeated models
            repeated_models_dict[model] = self.find_repeated_models(
                new_model=model,
                models=[models[idx] for idx in range(len(models)) if idx != model_idx]
            )
        
        # Drop repeated models
        for model, repeated_models in repeated_models_dict.items():
            if len(repeated_models) > 0:
                LOGGER.warning('Model %s (%s) has repeated models.', model.model_id, model.stage)

                # Sort models
                sorted_models = self.sort_models(
                    models=[model] + repeated_models,
                    by=score_to_prioritize
                )

                for drop_model in sorted_models[1:]:
                    try:
                        models.remove(drop_model)
                    except Exception as e:
                        LOGGER.warning(
                            'Unable to delete Model %s (%s).\n'
                            'Exception: %s.\n',
                            drop_model.model_id, drop_model.stage, e
                        )

        # Delete repeated_models_dict & sorted_models
        del repeated_models_dict
        try:
            del sorted_models
        except:
            pass
        
        return models

    """
    Loading & Saving Methods
    """

    def load_forced_model(self) -> dict:
        # Read forced_model
        try:
            if self.data_storage_env == 'filesystem':
                forced_model: dict = load_from_filesystem(
                    path=os.path.join(self.bucket, "utils", "model_registry", "forced_model.json"),
                    partition_cols=None,
                    filters=None
                )
            elif self.data_storage_env == 'S3':
                forced_model: dict = load_from_s3(
                    path=f"{self.bucket}/utils/model_registry/forced_model.json",
                    partition_cols=None,
                    filters=None
                )
            else:
                raise Exception(f'Invalid self.data_storage_env was received: "{self.data_storage_env}".\n')
        except Exception as e:
            LOGGER.warning(
                "Unable to load forced_model.\n"
                "Loading default forced_model instead.\n"
                "Exception: %s", e
            )

            # Load dummy forced_model
            forced_model: dict = {"forced_model_id": None}
        
        return forced_model

    def load_registry_dict(self) -> None:
        # Read registry
        try:
            if self.data_storage_env == 'filesystem':
                self.registry_dict: Dict[str, List[str]] = load_from_filesystem(
                    path=os.path.join(self.bucket, "utils", "model_registry", "model_registry.json"),
                    partition_cols=None,
                    filters=None
                )
            elif self.data_storage_env == 'S3':
                self.registry_dict: Dict[str, List[str]] = load_from_s3(
                    path=f"{self.bucket}/utils/model_registry/model_registry.json",
                    partition_cols=None,
                    filters=None
                )
            else:
                raise Exception(f'Invalid self.data_storage_env was received: "{self.data_storage_env}".\n')
        except Exception as e:
            LOGGER.warning(
                "Unable to load registry_dict.\n"
                "Loading default registry_dict instead.\n"
                "Exception: %s", e
            )

            # Load dummy registry_dict
            self.registry_dict: Dict[str, List[str]] = {
                "production": [], 
                "staging": [], 
                "development": []
            }

    def save_registry_dict(self) -> None:
        # Write self.registry
        if self.data_storage_env == 'filesystem':
            save_to_filesystem(
                asset=self.registry_dict,
                path=os.path.join(self.bucket, "utils", "model_registry", "model_registry.json"),
                partition_cols=None,
                overwrite=True
            )
        elif self.data_storage_env == 'S3':
            save_to_s3(
                asset=self.registry_dict,
                path=f"{self.bucket}/utils/model_registry/model_registry.json",
                partition_cols=None,
                overwrite=True
            )
        else:
            raise Exception(f'Invalid self.data_storage_env was received: "{self.data_storage_env}".\n')

    def save_models(self) -> None:
        # Save production model
        if self.prod_model is not None:
            self.prod_model.save()

        # Save staging models
        for model in self.staging_models:
            model.save()

        # Save development models
        for model in self.dev_models:
            model.save()

    """
    Other methods
    """

    def __repr__(self) -> str:
        output: str = "Model Registry:"

        # Prod Model
        champion = self.load_prod_model(light=True)
        if champion is not None:
            output += f"\nChampion Model ({champion.model_id}): Test score - {round(champion.test_score * 100, 2)} [{champion.optimization_metric}]\n\n"
        else:
            LOGGER.warning('loaded champion is None!.')

        # Staging Models
        for model in self.load_staging_models(light=True):
            if model is not None:
                output += f"Staging Model ({model.model_id}): Test score - {round(model.test_score * 100, 2)} [{model.optimization_metric}]\n"
        output += "\n"
        
        # Dev Models
        for model in self.load_dev_models(light=True):
            if model is not None:
                output += f"Dev Model ({model.model_id}): Test score - {round(model.test_score * 100, 2)} [{model.optimization_metric}]\n"
        output += "\n"
        
        return output
    