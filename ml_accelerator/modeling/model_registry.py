from ml_accelerator.config.params import Params
from ml_accelerator.modeling.model import Model
from ml_accelerator.modeling.classification_model import ClassificationModel
from ml_accelerator.modeling.regression_model import RegressionModel
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
        storage_env: str = Params.STORAGE_ENV,
        bucket: str = Params.BUCKET
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

        self.storage_env: str = storage_env
        self.bucket: str = bucket

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
        if by not in ['tuning_score', 'test_score']:
            LOGGER.critical('Invalid "by" parameter: %s', by)
            raise Exception(f'Invalid "by" parameter: {by}.\n')
        
        def sort_fun(model: Model):
            if by == 'tuning_score':
                # Cross validation metric
                if model.tuning_score is not None:
                    return model.tuning_score
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
                champ_tuning_score: float = champion.tuning_score
                champ_test_score: float = champion.test_score
            else:
                champ_tuning_score: float = None
                champ_test_score: float = None

            LOGGER.debug(
                'MLRegistry:\n'
                '%s\n'
                'Dev Models: %s\n'
                'Staging Models: %s\n'
                'Champion performance: %s [test] | %s [tuning]\n'
                '--------------------------------------------------------------------------\n',
                pformat(self.registry), len(dev_models), len(staging_models), 
                champ_tuning_score, champ_test_score
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
        dev_models = self.sort_models(
            models=dev_models,
            by='tuning_score'
        )

        # Find top n candidates
        staging_candidates = dev_models[: self.n_candidates]

        # Assert that all staging_candidates contain a test_table & a opt_table
        assert not(any([m is None for m in staging_candidates]))
        
        # Test & promote models from staging_candidates
        for model in staging_candidates:
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
        staging_models = self.sort_models(
            models=staging_models,
            by='test_score'
        )

        # Show registry
        if debug:
            _debug()

        # Find forced model
        forced_model = load_from_s3(path=f"{Params.bucket}/utils/forced_model/forced_model.json")['forced_model']

        if debug:
            print('Foreced Model:')
            pprint(forced_model)
            print('\n\n')

        # Update Champion with forced model
        if forced_model is not None:
            LOGGER.warning('Forced model was detected: %s.', forced_model)

            # Find forced Model
            forced_model_id, forced_model_class = forced_model

            # Check if forced model is the same as current champion
            if champion is not None and forced_model_id == champion.model_id:
                print(f'Forced Model is the same as current Champion.\n')
            else:
                # Re-define old & new champion models
                new_champion = None
                for model in dev_models + staging_models:
                    if model.model_id == forced_model_id:
                        new_champion = model

                if new_champion is None:
                    LOGGER.warning('Forced Model was not found in current models!')
                else:
                    # Define old champion
                    old_champion = champion

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
                    new_champion.save(
                        pickle_files=True,
                        parquet_files=False,
                        trading_tables=False,
                        model=False
                    )

                    if old_champion is not None:
                        # Demote Current Champion
                        old_champion.stage = 'staging'
                    else:
                        LOGGER.warning('Old champion was not found!')

                    # Re-assign champion variable
                    champion = new_champion

        # Define default champion if current champion is None
        if champion is None:
            LOGGER.warning(
                'There was no previous champion.\n'
                'Therefore, a new provisory champion will be chosen.\n'
            )
            
            # Promote New Champion
            new_champion = self.sort_models(
                models=staging_models,
                trading_metric=True,
                by_table='opt'
            )[0]

            new_champion.stage = 'production'

            # Remove model from staging_models
            staging_models.remove(new_champion)

            # Save new_champion
            new_champion.save(
                pickle_files=True,
                parquet_files=False,
                trading_tables=False,
                model=False
            )

            # Re-assign champion variable
            champion = new_champion

            print(f'New champion model:')
            print(champion)
            print('\n\n')

        elif update_champion:
            # Pick Challenger
            challenger = self.sort_models(
                models=staging_models,
                trading_metric=True,
                by_table='opt'
            )[0]

            if (
                # Challenger trading metric should be greater than the champion trading metric
                challenger.optimized_table.trading_metric > champion.optimized_table.trading_metric

                # Challenger est_monthly_ret should be more than 5% better than the champion trading metric
                and challenger.optimized_table.est_monthly_ret > 1.05 * champion.optimized_table.est_monthly_ret

                # Challenger test_table trading_metric should be greater than 0
                # and challenger.test_table.trading_metric > 0
            ):
                print(f'New Champion mas found (opt performance: {challenger.optimized_table.trading_metric}):')
                print(challenger)
                print(f'Previous Champion (opt performance: {champion.optimized_table.trading_metric}):')
                print(champion)

                # Promote Challenger
                challenger.stage = 'production'

                # Remove challenger from staging_models
                staging_models.remove(challenger)

                # Demote Champion
                champion.stage = 'staging'

                # Add old champion to staging_models
                staging_models.append(champion)

                # Save New Champion
                challenger.save(
                    pickle_files=True,
                    parquet_files=False,
                    trading_tables=False,
                    model=False
                )

                # Re-assign champion variable
                champion = challenger

                print(f'New champion model:')
                print(champion)
                print('\n\n')

        """
        Save Models & Update self.registry
        """
        dev_models = self.sort_models(
            models=dev_models,
            trading_metric=True,
            by_table='val'
        )[: 5]

        # Update Dev Registry
        self.registry['development'] = [
            (m.model_id, m.model_class) for m in dev_models 
            if m.val_table.tuning_metric > 0
        ]

        # Save Dev Models
        for model in dev_models:
            assert model.stage == 'development'

            # Save Model
            model.save(
                pickle_files=True,
                parquet_files=False,
                trading_tables=False,
                model=False
            )

        # Update Staging Registry
        self.registry['staging'] = [
            (m.model_id, m.model_class) for m in staging_models 
            if m.optimized_table.trading_metric > 0 # and m.val_table.trading_metric > 0
        ]

        # Save Staging Models
        for model in staging_models:
            assert model.stage == 'staging'

            # Save Model
            model.save(
                pickle_files=True,
                parquet_files=False,
                trading_tables=False,
                model=False
            )

        # Update Production Registry
        self.registry['production'] = [(champion.model_id, champion.model_class)]
        
        # Save Production Model
        assert champion.stage == 'production'

        champion.save(
            pickle_files=True,
            parquet_files=False,
            trading_tables=False,
            model=False
        )

        if debug:
            debug_()
        
        # Clean File System
        self.clean_s3_models()

        # Save self.registry
        self.save()

    def clean_s3_models(self) -> None:
        keep_regs = (
            self.registry["development"] +
            self.registry["staging"] + 
            self.registry["production"]
        )
        keep_ids = [reg[0] for reg in keep_regs]

        LOGGER.info('keep_ids:\n%s\n', pformat(keep_ids))

        # Clean Models
        models_subdir = f"modeling/models/{self.intervals}"

        for key in find_keys(bucket=Params.bucket, subdir=models_subdir):
            if not any([model_id in key for model_id in keep_ids]):
                print(f"Deleting {Params.bucket}/{key}.\n")
                delete_from_s3(path=f"{Params.bucket}/{key}")

        # Clean TradingTables
        trading_tables_subdir = f"trading/trading_table/{self.intervals}"

        for key in find_keys(bucket=Params.bucket, subdir=trading_tables_subdir):
            if 'trading_returns' not in key and not any([model_id in key for model_id in keep_ids]):
                print(f"Deleting {Params.bucket}/{key}.\n")
                delete_from_s3(path=f"{Params.bucket}/{key}")

        # Clean Model backup
        model_backup_subdir = f"backup/models/{self.intervals}"

        for key in find_keys(bucket=Params.bucket, subdir=model_backup_subdir):
            if not any([model_id in key for model_id in keep_ids]):
                print(f"Deleting {Params.bucket}/{key}.\n")
                delete_from_s3(path=f"{Params.bucket}/{key}")

    def find_repeated_models(
        self,
        new_model: Model, 
        models: List[Model] = None, 
        from_: str = None
    ) -> List[Model]:
        # Validate models
        if models is None:
            if from_ is None:
                LOGGER.critical('If "models" parameter is None, then "from_" parameter cannot be None as well.')
                raise Exception('If "models" parameter is None, then "from_" parameter cannot be None as well.\n\n')
            
            model_regs = self.registry['production'] + self.registry['staging'] + self.registry['development']
            
            if from_ == 'GFM':
                models = [
                    self.load_model(
                        model_id=reg[0], 
                        model_class=reg[1],
                        light=True
                    ) for reg in model_regs if reg[1] == 'GFM'
                ]
            elif from_ == 'LFM':
                models = [
                    self.load_model(
                        model_id=reg[0], 
                        model_class=reg[1],
                        light=True
                    ) for reg in model_regs if reg[1] == 'LFM'
                ]
            else:
                LOGGER.critical('"from_" parameter got an invalid value: %s (expected "GFM" or "LFM").', from_)
                raise Exception(f'"from_" parameter got an invalid value: {from_} (expected "GFM" or "LFM").\n\n')

        def extract_tuple_attrs(model: Model):
            # Define base attrs to add
            attrs = {
                'coin_name': model.coin_name,
                'intervals': model.intervals,
                'lag': model.lag,
                'algorithm': model.algorithm,
                'method': model.method,
                # 'pca': model.pca
            }

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
        from_: str = None,
        trading_metric: bool = True,
        by_table: str = 'opt',
        debug: bool = False
    ):
        # Validate models
        if models is None:
            if from_ is None:
                LOGGER.critical('If "models" parameter is None, then "from_" parameter cannot be None as well.')
                raise Exception('If "models" parameter is None, then "from_" parameter cannot be None as well.\n\n')
            
            model_regs = self.registry['production'] + self.registry['staging'] + self.registry['development']
            
            if from_ == 'GFM':
                models = [
                    self.load_model(
                        model_id=reg[0], 
                        model_class=reg[1],
                        light=True
                    ) for reg in model_regs if reg[1] == 'GFM'
                ]
            elif from_ == 'LFM':
                models = [
                    self.load_model(
                        model_id=reg[0], 
                        model_class=reg[1],
                        light=True
                    ) for reg in model_regs if reg[1] == 'LFM'
                ]
            else:
                LOGGER.critical('"from_" parameter got an invalid value: %s (expected "GFM" or "LFM").', from_)
                raise Exception(f'"from_" parameter got an invalid value: {from_} (expected "GFM" or "LFM").\n\n')
        
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
                LOGGER.warning('Model %s (%s | %s) has repeated models.', model.model_id, model.stage, model.model_class)

                # Sort models
                sorted_models = self.sort_models(
                    models=[model] + repeated_models,
                    trading_metric=trading_metric,
                    by_table=by_table
                )

                for drop_model in sorted_models[1:]:
                    try:
                        models.remove(drop_model)
                    except Exception as e:
                        LOGGER.warning(
                            'Unable to delete Model %s (%s | %s).\n'
                            'Exception: %s.\n',
                            drop_model.model_id, drop_model.stage, drop_model.model_class, e
                        )

        # Delete repeated_models_dict & sorted_models
        del repeated_models_dict
        try:
            del sorted_models
        except:
            pass
        
        return models

    def load_forced_model(self) -> dict:
        # Read forced_model
        if self.storage_env == 'filesystem':
            forced_model: dict = load_from_filesystem(
                path=os.path.join(Params.BUCKET, "models", "forced_model.json")
            )
        elif self.storage_env == 'S3':
            forced_model: dict = load_from_s3(
                path=f"{self.bucket}/models/forced_model.json"
            )
        else:
            raise Exception(f'Invalid self.storage_env was received: "{self.storage_env}".\n')
        
        return forced_model

    def load_registry(self) -> None:
        # Read registry
        if self.storage_env == 'filesystem':
            self.registry: Dict[str, List[List[str, str]]] = load_from_filesystem(
                path=os.path.join(Params.BUCKET, "models", "model_registry.json")
            )
        elif self.storage_env == 'S3':
            self.registry: Dict[str, List[List[str, str]]] = load_from_s3(
                path=f"{self.bucket}/models/model_registry.json"
            )
        else:
            raise Exception(f'Invalid self.storage_env was received: "{self.storage_env}".\n')

    def save(self) -> None:
        # Write self.registry
        if self.storage_env == 'filesystem':
            save_to_filesystem(
                asset=self.registry,
                path=os.path.join(Params.BUCKET, "models", "model_registry.json"),
                partition_column=None
            )
        elif self.storage_env == 'S3':
            save_to_s3(
                asset=self.registry,
                path=os.path.join(Params.BUCKET, "models", "model_registry.json"),
                partition_column=None
            )
        else:
            raise Exception(f'Invalid self.storage_env was received: "{self.storage_env}".\n')

    def __repr__(self) -> str:
        LOGGER.info('Model Registry:')

        # Prod Model
        champion = self.load_prod_model()
        if champion is not None:
            LOGGER.info(
                'Champion Model: %s\n'
                '    - Validation score: %s\n'
                '    - Test score: %s\n',
                champion.model_id, champion.tuning_score, champion.test_score
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
                    model.model_id, model.tuning_score, model.test_score
                )
        
        # Dev Models
        for model in self.load_dev_models():
            if model is not None:
                LOGGER.info(
                    'Dev Model: %s\n'
                    '    - Validation score: %s\n',
                    model.model_id, model.tuning_score
                )

        return '\n\n'
    