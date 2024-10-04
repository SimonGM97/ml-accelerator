from ml_accelerator.config.params import Params
from ml_accelerator.modeling.models.model import Model
from ml_accelerator.modeling.models.classification_model import ClassificationModel
from ml_accelerator.modeling.models.regression_model import RegressionModel
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
import shutil
import os
from pprint import pprint, pformat
from typing import List, Dict, Tuple


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

        # Define self.registry
        self.registry: Dict[str, List[List[str, str]]] = {
            "production": [], 
            "staging": [], 
            "development": []
        }

        self.load_registry()

    def load_model(
        self, 
        model_id: str
    ) -> Model:
        try:
            # Instanciate Model
            if self.task == 'classification':
                model = ClassificationModel(model_id=model_id)
            elif self.task == 'regression':
                model = RegressionModel(model_id=model_id)

            # Load Model
            model.load()

            return model
        except Exception as e:
            LOGGER.error(
                'Unable to load Model %s.\n'
                'Exception: %s\n',
                model_id, e
            )
            return

    def load_dev_models(self) -> List[Model]:
        # Load Staging Models
        dev_models = []

        LOGGER.info('Loading dev models:')
        for model_id in tqdm(self.registry['development']):
            dev_models.append(self.load_model(model_id=model_id))
        
        return [model for model in dev_models if model is not None]
    
    def load_staging_models(self) -> List[Model]:
        # Load Staging Models
        stage_models = []

        LOGGER.info('Loading staging models:')
        for model_id in tqdm(self.registry['staging']):
            stage_models.append(self.load_model(model_id=model_id))
        
        return [model for model in stage_models if model is not None]
    
    def load_prod_model(self) -> Model:
        """
        Method for the production model.

        :return: (Model) Production model.
        """
        LOGGER.info('Loading prod model')
        if len(self.registry['production']) > 0:
            # Find champion reg
            model_id = self.registry['production'][0]

            # Load and return champion model
            return self.load_model(model_id=model_id)
        return None
    
    @staticmethod
    def sort_models(
        models: List[Model],
        by: str = 'test_score'
    ) -> List[Model]: 
        if by not in ['val_score', 'test_score']:
            LOGGER.critical('Invalid "by" parameter: %s', by)
            raise Exception(f'Invalid "by" parameter: {by}.\n')
        
        def sort_fun(model: Model):
            if by == 'val_score':
                # Cross validation metric
                if model.val_score is not None:
                    return model.val_score
            elif by == 'test_score':
                # Test score
                if model.test_score is not None:
                    return model.test_score
            return 0
        
        models.sort(key=sort_fun, reverse=True)

        return models

    @staticmethod
    def needs_repair(d: dict):
        for value in d.values():
            if value:
                return True
        return False

    def update_model_stages(
        self,
        update_champion: bool = False,
        debug: bool = False
    ) -> None:
        """
        Method that will re-define model stages, applying the following logic:
            - Top n_candidate dev models will be promoted as "staging" models (also referred as "challenger" models),
              based on their mean cross validation performance.
            - The top staging model will compete with the production model (also referred as "champion" model), 
              based on their test performance.
        """
        def _debug():
            if champion is not None:
                champ_val_score: float = champion.val_score
                champ_test_score: float = champion.test_score
            else:
                champ_val_score: float = None
                champ_test_score: float = None

            LOGGER.debug(
                'MLRegistry:\n'
                '%s\n'
                'Dev Models: %s\n'
                'Staging Models: %s\n'
                'Champion performance: %s [test] | %s [tuning]\n'
                '--------------------------------------------------------------------------\n',
                pformat(self.registry), len(dev_models), len(staging_models), 
                champ_val_score, champ_test_score
            )

        # Load light Models
        champion: Model = self.load_prod_model()
        staging_models: List[Model] = []
        dev_models: List[Model] = self.load_staging_models() + self.load_dev_models()

        # Assert that all dev_models contain a val_table
        assert not(any([m is None for m in dev_models]))

        # Degrade all models to development (except for champion)
        for model in dev_models:
            # Re-asign stage
            model.stage = 'development'

        # Sort Dev Models (based on validation performance)
        dev_models: List[Model] = self.sort_models(
            models=dev_models,
            by='val_score'
        )

        # Test & promote models from staging_candidates
        for model in dev_models[: self.n_candidates]:
            # Diagnose model
            diagnostics_dict = model.diagnose()

            if not self.needs_repair(diagnostics_dict):
                # Promote Model
                model.stage = 'staging'

                # Add model to staging_models
                staging_models.append(model)

                # Remove model from dev_models
                dev_models.remove(model)
            else:
                LOGGER.warning(
                    '%s was NOT pushed to Staging.\n'
                    'diagnostics_dict:\n%s\n',
                    model.model_id, pformat(diagnostics_dict)
                )

        # Sort Staging Models (based on test performance)
        staging_models: List[Model] = self.sort_models(
            models=staging_models,
            by='test_score'
        )

        # Show registry
        if debug:
            _debug()

        # Find forced model ID
        forced_model_id: str = self.load_forced_model()['forced_model_id']

        if debug:
            LOGGER.debug('Foreced Model:\n%s\n', pformat(forced_model_id))

        # Update Champion with forced model
        if forced_model_id is not None:
            LOGGER.warning('Forced model was detected: %s.', forced_model_id)

            # Check if forced model is the same as current champion
            if champion is not None and forced_model_id == champion.model_id:
                LOGGER.info(f'Forced Model is the same as current Champion.\n')
            else:
                # Re-define old & new champion models
                new_champion: Model = None
                for model in dev_models + staging_models:
                    if model.model_id == forced_model_id:
                        new_champion: Model = model

                if new_champion is None:
                    LOGGER.warning('Forced Model was not found in current models!')
                else:
                    # Define old champion
                    old_champion: Model = champion

                    # Record Previous Stage
                    prev_new_champion_stage = new_champion.stage

                    # Promote New Champion
                    new_champion.stage = 'production'                    

                    # Remove new champion from dev_models or staging_models
                    if prev_new_champion_stage == 'development':
                        dev_models.remove(new_champion)
                    elif prev_new_champion_stage == 'staging':
                        staging_models.remove(new_champion)
                    else:
                        LOGGER.critical(
                            "new_champion (%s) had an invalid stage: %s.",
                            new_champion.model_id, prev_new_champion_stage
                        )
                        raise Exception(f"new_champion ({new_champion.model_id}) had an invalid stage: {prev_new_champion_stage}.\n\n")
                    
                    # Add old champion to staging_models
                    staging_models.append(old_champion)
                    
                    # Save New Champion
                    new_champion.save()

                    if old_champion is not None:
                        # Demote Current Champion
                        old_champion.stage = 'staging'
                    else:
                        LOGGER.warning('Old champion was not found!')

                    # Re-assign champion variable
                    champion: Model= new_champion

        # Define default champion if current champion is None
        if champion is None:
            LOGGER.warning(
                'There was no previous champion.\n'
                'Therefore, a new provisory champion will be chosen.\n'
            )
            
            # Promote New Champion
            new_champion: Model = self.sort_models(
                models=staging_models,
                by='test_score'
            )[0]

            new_champion.stage = 'production'

            # Remove model from staging_models
            staging_models.remove(new_champion)

            # Save new_champion
            new_champion.save()

            # Re-assign champion variable
            champion: Model = new_champion

            LOGGER.info('New champion model:\n%s\n', champion)

        elif update_champion:
            # Pick Challenger
            challenger: Model = self.sort_models(
                models=staging_models,
                by='test_score'
            )[0]

            if challenger.test_score > champion.test_score:
                LOGGER.info(
                    'New Champion mas found: %s - [%s: %s]\n'
                    'Previous Champion:  %s - [%s: %s]',
                    challenger.model_id, challenger.optimization_metric, challenger.test_score,
                    champion.model_id, champion.optimization_metric, champion.test_score

                )

                # Promote Challenger
                challenger.stage = 'production'

                # Remove challenger from staging_models
                staging_models.remove(challenger)

                # Demote Champion
                champion.stage = 'staging'

                # Add old champion to staging_models
                staging_models.append(champion)

                # Save New Champion
                challenger.save()

                # Re-assign champion variable
                champion = challenger

        """
        Save Models & Update self.registry
        """
        dev_models = self.sort_models(
            models=dev_models,
            by='val_score'
        )[: 5]

        # Update Dev Registry
        self.registry['development'] = [
            m.model_id for m in dev_models if m.val_score > 0
        ]

        # Save Dev Models
        for model in dev_models:
            assert model.stage == 'development'

            # Save Model
            model.save()

        # Update Staging Registry
        self.registry['staging'] = [
            m.model_id for m in staging_models 
            if m.test_score > 0 # and m.val_table.trading_metric > 0
        ]

        # Save Staging Models
        for model in staging_models:
            assert model.stage == 'staging'

            # Save Model
            model.save()

        # Update Production Registry
        self.registry['production'] = [champion.model_id]
        
        # Save Production Model
        assert champion.stage == 'production'

        champion.save()

        if debug:
            _debug()
        
        # Clean registry
        self.clean_registry()

        # Save self.registry
        self.save_registry()

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
            self.registry["production"] +
            self.registry["staging"] +
            self.registry["development"]
        )

        for root, directories, files in os.walk(
            os.path.join(self.bucket, *self.models_path)
        ):
            for file in files:
                model_id = file.split('_')[0]
                if model_id not in model_ids:
                    delete_path = os.path.join(self.models_path, file)
                    LOGGER.info("Deleting %s.", delete_path)
                    os.remove(delete_path)

    def clean_s3_models(self) -> None:
        # Load model ids
        model_ids: List[str] = (
            self.registry["production"] +
            self.registry["staging"] +
            self.registry["development"]
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

    def find_repeated_models(
        self,
        new_model: Model, 
        models: List[Model] = None
    ) -> List[Model]:
        # Validate models
        if models is None:
            # Extract models
            model_ids = self.registry['production'] + self.registry['staging'] + self.registry['development']            
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
    ):
        # Validate models
        if models is None:
            # Extract models
            model_ids = self.registry['production'] + self.registry['staging'] + self.registry['development']            
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
        
        return forced_model

    def load_registry(self) -> None:
        # Read registry
        if self.data_storage_env == 'filesystem':
            self.registry: Dict[str, List[List[str, str]]] = load_from_filesystem(
                path=os.path.join(self.bucket, "utils", "model_registry", "model_registry.json"),
                partition_cols=None,
                filters=None
            )
        elif self.data_storage_env == 'S3':
            self.registry: Dict[str, List[List[str, str]]] = load_from_s3(
                path=f"{self.bucket}/utils/model_registry/model_registry.json",
                partition_cols=None,
                filters=None
            )
        else:
            raise Exception(f'Invalid self.data_storage_env was received: "{self.data_storage_env}".\n')

    def save_registry(self) -> None:
        # Write self.registry
        if self.data_storage_env == 'filesystem':
            save_to_filesystem(
                asset=self.registry,
                path=os.path.join(self.bucket, "utils", "model_registry", "model_registry.json"),
                partition_column=None,
                overwrite=True
            )
        elif self.data_storage_env == 'S3':
            save_to_s3(
                asset=self.registry,
                path=f"{self.bucket}/utils/model_registry/model_registry.json",
                partition_column=None,
                overwrite=True
            )
        else:
            raise Exception(f'Invalid self.data_storage_env was received: "{self.data_storage_env}".\n')

    """
    Other methods
    """

    def __repr__(self) -> str:
        LOGGER.info('Model Registry:')

        # Prod Model
        champion = self.load_prod_model()
        if champion is not None:
            LOGGER.info(
                'Champion Model: %s\n'
                '    - Validation score: %s\n'
                '    - Test score: %s\n',
                champion.model_id, champion.val_score, champion.test_score
            )
        else:
            LOGGER.warning('loaded champion is None!.')

        # Staging Models
        for model in self.load_staging_models():
            if model is not None:
                LOGGER.info(
                    'Staging Model: %s\n'
                    '    - Validation score: %s\n'
                    '    - Test score: %s\n',
                    model.model_id, model.val_score, model.test_score
                )
        
        # Dev Models
        for model in self.load_dev_models():
            if model is not None:
                LOGGER.info(
                    'Dev Model: %s\n'
                    '    - Validation score: %s\n',
                    model.model_id, model.val_score
                )

        return '\n\n'
    