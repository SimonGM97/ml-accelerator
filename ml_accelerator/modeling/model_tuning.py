from ml_accelerator.config.params import Params
from ml_accelerator.modeling.models.model import Model
from ml_accelerator.modeling.models.classification_model import ClassificationModel
from ml_accelerator.modeling.models.regression_model import RegressionModel
from ml_accelerator.modeling.model_registry import ModelRegistry
from ml_accelerator.utils.logging.logger_helper import get_logger
from ml_accelerator.utils.timing.timing_helper import timing
from ml_accelerator.config.env import Env

from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK, atpe
from hyperopt.fmin import generate_trials_to_calculate
from hyperopt.early_stop import no_progress_loss
from hyperopt.pyll.base import scope
import pandas as pd
import numpy as np
from functools import partial
import time
from typing import Dict, List
from copy import deepcopy
from pprint import pformat

import warnings

# Suppress only UserWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Get logger
LOGGER = get_logger(name=__name__)


class ModelTuner:
    """
    Class designed to find the most performant ML models, leveraging hyperopt's TPE based search engine 
    to optimize both the model flavor (or algorithm) & set of hyperparameters in order to train robust 
    models with strong generalization capabilities.
    """

    def __init__(
        self,
        algorithms: List[str] = Params.ALGORITHMS,
        search_space: List[dict] = Params.SEARCH_SPACE,
        target_column: str = Params.TARGET_COLUMN,
        task: str = Params.TASK,
        optimization_metric: str = Params.OPTIMIZATION_METRIC,
        importance_method: str = Params.IMPORTANCE_METHOD,
        n_candidates: int = Params.N_CANDIDATES,
        min_performance: float = Params.MIN_PERFORMANCE,
        val_splits: int = Params.VAL_SPLITS
    ) -> None:
        # Define attributes
        self.algorithms: List[str] = algorithms
        self.search_space: List[dict] = search_space

        self.target_column: str = target_column
        self.task: str = task

        self.optimization_metric: str = optimization_metric
        self.importance_method: str = importance_method

        self.n_candidates: int = n_candidates
        self.min_performance: float = min_performance
        self.val_splits: int = val_splits
        
        # Environment parameters
        self.model_storage_env: str = Env.get("MODEL_STORAGE_ENV")
        self.data_storage_env: str = Env.get("DATA_STORAGE_ENV")
        self.bucket: str = Env.get("BUCKET_NAME")
        self.models_path: str = Env.get('MODELS_PATH')

        # Define search space parameters
        self.int_parameters: List[str] = None
        self.choice_parameters: Dict[str, List[str]] = None
        self.model_type_choices: List[dict] = None
        self.search_space: Dict[str, List[dict]] = None

        self.extract_search_space_params(
            search_space=search_space
        )

        # Define default attrs
        self.model_registry: ModelRegistry = ModelRegistry(
            n_candidates=self.n_candidates,
            task=self.task
        )
        self.models: List[Model] = None

        # Load self.models
        self.load()

    def extract_search_space_params(
        self,
        search_space: List[dict]
    ) -> Dict[str, List[dict]]:
        def extract_space(
            algorithm: str,
            parameter_name: str,
            space: List[dict]
        ):
            # Extract dist
            dist: str = space['dist']

            if dist == 'choice':
                self.choice_parameters[f"{algorithm}.{parameter_name}"] = space['choices']
                return hp.choice(f"{algorithm}.{parameter_name}", space['choices'])
            elif dist == 'quniform':
                self.int_parameters.append(f"{algorithm}.{parameter_name}")
                return scope.int(hp.quniform(f"{algorithm}.{parameter_name}", space['min'], space['max'], 1))
            elif dist == 'uniform':
                return hp.uniform(f"{algorithm}.{parameter_name}", space['min'], space['max'])
            elif dist == 'loguniform':
                return hp.loguniform(f"{algorithm}.{parameter_name}", np.log(space['min']), np.log(space['max']))
            else:
                raise Exception(f'Invalid "dist" was found: "{dist}"')

        # Populate self.model_type_choices, self.int_parameters & self.choice_parameters
        self.model_type_choices: List[dict] = []
        self.int_parameters: List[str] = []
        self.choice_parameters: Dict[str, List[str]] = {}

        for choice in search_space:
            # Extract algorithm & hyper_parameters
            algorithm: str = choice['algorithm']
            hyper_parameters: List[dict] = choice['hyper_parameters']

            # Define new choice
            new_choice = {"algorithm": algorithm}

            # Add hyper_parameters
            hyper_parameters = {
                f"{algorithm}.{hyper_parameter['parameter_name']}": extract_space(
                    algorithm=algorithm,
                    parameter_name=hyper_parameter['parameter_name'],
                    space=hyper_parameter['space']
                )
                for hyper_parameter in hyper_parameters
            }

            new_choice.update(**hyper_parameters)

            # Append new choice
            self.model_type_choices.append(new_choice)

        # Define seach space
        self.search_space: Dict[str, List[dict]] = {
            "model_type": hp.choice('model_type', self.model_type_choices)
        }

        return search_space

    def parameter_configuration(
        self, 
        parameters_list: list[dict],
        complete_parameters: bool = False,
        choice_parameters: str = 'index'
    ) -> List[dict]:
        """
        Method designed to interprete and complete the keys, values and indexes for each iteration of the search 
        parameters; following expected input & output structure expected by hyperopt.

        :param `parameters_list`: (list) List of parameters to standardize.
        :param `complete_parameters`: (bool) Wether or not to complete any missing keys in the parameters.
        :param `choice_parameters`: (str) Used to set how the choice parameters will be outputed.

        :return: (pd.DataFrame) List of parameters with standardized structure.
        """
        if choice_parameters not in ['index', 'values']:
            raise Exception(f'Invalid "choice_parameters": {choice_parameters}.\n\n')

        int_types = [int, np.int64, np.int32] #, float, np.float32 ,np.float64]

        for parameters in parameters_list:
            # Check "algorithm" parameter
            if 'algorithm' not in parameters.keys() and type(parameters['model_type']) in int_types:
                parameters['algorithm'] = self.algorithms[parameters['model_type']]
            elif 'algorithm' in parameters.keys() and type(parameters['algorithm']) == str and type(parameters['model_type']) in int_types:
                parameters['model_type'] = self.algorithms.index(parameters['algorithm'])

            # Check "model_type" parameter
            if parameters['model_type'] is None:
                parameters['model_type'] = self.algorithms.index(parameters['algorithm'])
            if type(parameters['model_type']) == dict:
                parameters.update(**parameters['model_type'])
                parameters['model_type'] = self.algorithms.index(parameters['algorithm'])

        # Complete Dummy Parameters
        if complete_parameters:
            dummy_list = []
            for model_type in self.model_type_choices:
                dummy_list.extend(list(model_type.keys()))

            for parameters in parameters_list:
                for dummy_parameter in dummy_list:
                    if dummy_parameter not in parameters.keys():
                        parameters[dummy_parameter] = 0
        else:
            for parameters in parameters_list:
                filtered_keys = list(self.search_space.keys())
                filtered_keys += list(self.model_type_choices[parameters['model_type']].keys())

                dummy_parameters = parameters.copy()
                for parameter in dummy_parameters.keys():
                    if parameter not in filtered_keys:
                        parameters.pop(parameter)

        # Check Choice Parameters
        if choice_parameters == 'index':                   
            for parameters in parameters_list:
                choice_keys = [k for k in self.choice_parameters.keys() 
                               if k in parameters.keys() and type(parameters[k]) not in int_types]
                for choice_key in choice_keys:
                    parameters[choice_key] = self.choice_parameters[choice_key].index(parameters[choice_key])
        else:            
            for parameters in parameters_list:
                choice_keys = [k for k in self.choice_parameters.keys() 
                               if k in parameters.keys() and type(parameters[k]) in int_types]
                for choice_key in choice_keys:
                    parameters[choice_key] = self.choice_parameters[choice_key][parameters[choice_key]]

        # Check int parameters
        for parameters in parameters_list:
            for parameter in parameters:
                if parameter in self.int_parameters and parameters[parameter] is not None:
                    parameters[parameter] = int(parameters[parameter])

        return parameters_list

    def prepare_hyper_parameters(
        self,
        parameters: dict
    ) -> dict:
        """
        Method that standardizes the structure of hyper-parameters, so that it can be consumed while 
        instanciating new ML classification models.

        :param `parameters`: (dict) Parameters with hyper-parameters to standardize.

        :return: (dict) Parameters with standardized hyper-parameters.
        """
        hyper_param_choices = [d for d in self.model_type_choices if d['algorithm'] == parameters['algorithm']][0]

        parameters['hyper_parameters'] = {
            hyper_param: parameters.pop(hyper_param) 
            for hyper_param in hyper_param_choices.keys()
            if hyper_param != 'algorithm'
        }

        parameters['hyper_parameters'] = {
            k.replace(f'{parameters["algorithm"]}.', ''): v
            for k, v in parameters['hyper_parameters'].items()
        }

        return parameters

    def prepare_parameters(
        self,
        parameters: dict,
        selected_features: List[str],
        debug: bool = False
    ) -> dict:
        """
        Method designed to standardize the structure, complete required keys and interpret values 
        of the set of parameters & hyperparameters that are being searched by the hyperopt TPE 
        powered seach engine.

        :param `parameters`: (dict) Parameters with raw structure, uncomplete keys & uninterpreted values.
        :param `debug`: (bool) Wether or not to show input and output parameters for debugging purposes.

        :return: (dict) Parameters with standardized structure, complete keys & interpreted values.
        """
        if debug:
            t1 = time.time()
            print("parameters:\n"
                  "{")
            for key in parameters:
                if key != 'selected_features':
                    print(f"    '{key}': {parameters[key]}")
            print('}\n\n')

        parameters = self.parameter_configuration(
            parameters_list=[parameters],
            complete_parameters=False,
            choice_parameters='values'
        )[0]

        # Add register Parameters
        if 'model_id' not in parameters.keys():
            parameters['model_id'] = None
        if 'version' not in parameters.keys():
            parameters['version'] = 0
        if 'stage' not in parameters.keys():
            parameters['stage'] = 'development'

        # Storage Parameters
        if 'task' not in parameters.keys():
            parameters['task'] = self.task

        # Data Parameters
        if 'target_column' not in parameters.keys():
            parameters['target_column'] = self.target_column
        if 'selected_features' not in parameters.keys():
            parameters['selected_features'] = deepcopy(selected_features)

        # Performance Parameters
        if 'optimization_metric' not in parameters.keys():
            parameters['optimization_metric'] = self.optimization_metric

        # Feature importance Parameters
        if 'importance_method' not in parameters.keys():
            parameters['importance_method'] = self.importance_method

        # Others
        if (
            'cutoff' not in parameters.keys() and 
            self.task in ['binary_classification', 'multiclass_classification']
        ):
            parameters['cutoff'] = 0.5
        if 'model_type' in parameters.keys():
            parameters.pop('model_type')

        # Prepare Hyper Parameters
        parameters = self.prepare_hyper_parameters(
            parameters=parameters
        )

        if debug:
            print("new parameters:\n"
                  "{")
            for key in parameters:
                if key == 'selected_features' and parameters[key] is not None:
                    print(f"    '{key}' (len): {len(parameters[key])}")
                else:
                    print(f"    '{key}': {parameters[key]}")
            print('}\n')
            print(f'Time taken to prepare parameters: {round(time.time() - t1, 1)} sec.\n\n')
        
        return parameters

    def objective(
        self,
        parameters: dict,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        selected_features: List[str],
        debug: bool = False
    ) -> dict:
        """
        Method defined as the objective function for the hyperopt's TPE based search engine; which will:
            - Standardize, complete & interprete inputed parameters
            - Leverage MLPipeline to build a ML classification model with the inputed parameters
            - Log the resulting model in the mlflow tracking server, if the validation performance is over
              a defined threshold.
            - Output the validation performance (mean cross validation score) as the loss function.

        :param `parameters`: (dict) Parameters with raw structure, uncomplete keys & uninterpreted values.
        :param `debug`: (bool) Wether or not to show intermediate logs for debugging purposes.

        :return: (dict) Loss function with the validation performance of the ML classification model.
        """
        # try:
        # Parameter configuration
        parameters = self.prepare_parameters(
            parameters=parameters,
            selected_features=selected_features,
            debug=debug
        )

        # Instanciate model
        if self.task in ['binary_classification', 'multiclass_classification']:
            model = ClassificationModel(**parameters)
        elif self.task == 'regression':
            model = RegressionModel(**parameters)
        else:
            raise NotImplementedError(f'Task "{self.task}" has not been implemented.')
        
        # Build Model
        model.build(debug=debug)

        # Evaluate Model utilizing cross validation
        model.evaluate_val(
            X_train=X_train,
            y_train=y_train,
            splits=self.val_splits,
            debug=debug
        )

        # Log dev model
        if (
            self.min_performance is None 
            or (
                self.min_performance is not None
                and model.val_score >= self.min_performance
            )
        ):
            self.update_models(new_candidate=model)

        # Return Loss
        return {'loss': -model.val_score, 'status': STATUS_OK}
        # except Exception as e:
        #     LOGGER.warning(
        #         'Skipping iteration.\n'
        #         'Exception: %s\n'
        #         'Parameters:\n%s\n',
        #         e, pformat(parameters)
        #     )
        #     return {'loss': np.inf, 'status': STATUS_OK}

    def update_models(
        self,
        new_candidate: Model
    ) -> None:
        if (
            new_candidate.stage == 'development'
            and new_candidate.model_id not in [m.model_id for m in self.models]
        ):
            if (
                len(self.models) < self.n_candidates
                or new_candidate.val_score > self.models[-1].val_score
            ):
                # Add new_candidate to self.models
                self.models.append(new_candidate)
            
                # Drop duplicate Models (keeping most performant)
                self.models = self.model_registry.drop_duplicate_models(
                    models=self.models,
                    score_to_prioritize='val_score'
                )
            
                # Sort Models
                self.models = self.model_registry.sort_models(
                    models=self.models,
                    by='val_score'
                )

                if len(self.models) > self.n_candidates:
                    self.models = self.models[:self.n_candidates]
                
                if new_candidate.model_id in [m.model_id for m in self.models]:
                    LOGGER.info(
                        'Model %s (%s) was added to self.models (Models: %s).',
                        new_candidate.model_id,new_candidate.stage, len(self.models)
                    )
        else:
            LOGGER.warning(
                '%s is already in dev_models.\n'
                'Note: This should only happen for evaluation of warm models.\n',
                new_candidate.model_id
            )

    def evaluate_dev_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        debug: bool = False
    ) -> None:
        """
        Method that evaluates the development models on the test set, defined in the MLPipeline.

        :param `debug`: (bool) Wether or not to show self.dev_models performances logs for debugging purposes.
        """
        for model in self.models:
            if model.stage == 'development':
                # Fit model
                model.fit(X=X_train, y=y_train)

                # Predict test
                y_pred: np.ndarray = model.predict(X=X_test)
                
                # Evaluate model on test set & find model.test_score
                model.evaluate_test(
                    y_pred=y_pred,
                    y_test=y_test,
                    debug=debug
                )

                # Find Model feature importance
                model.find_feature_importance(
                    X_test=X_test,
                    find_new_shap_values=True,
                    debug=debug
                )

    @timing
    def tune_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        selected_features: List[str],
        use_warm_start: bool = True,
        max_evals: int = Params.MAX_EVALS,
        loss_threshold: float = Params.LOSS_THRESHOLD,
        timeout_mins: float = Params.TIMEOUT_MINS,
        debug: bool = False
    ) -> None:
        # Sort self.models by val_score
        self.models: List[Model] = self.model_registry.sort_models(
            models=self.models,
            by='val_score'
        )

        # Define warm start
        warm_models = [model for model in self.models if model.algorithm in self.algorithms]
        if (
            use_warm_start
            and len(warm_models) > 0 
            and warm_models[0].warm_start_params is not None
        ):
            best_parameters_to_evaluate = self.parameter_configuration(
                parameters_list=[warm_models[0].warm_start_params],
                complete_parameters=True,
                choice_parameters='index'
            )
            trials = generate_trials_to_calculate(best_parameters_to_evaluate)

            if debug:
                LOGGER.debug(
                    'best_parameters_to_evaluate:\n%s\n',
                    pformat(best_parameters_to_evaluate)
                )
        else:
            trials = None

        # Prepare objective
        fmin_objective = partial(
            self.objective,
            X_train=X_train,
            y_train=y_train,
            selected_features=selected_features,
            debug=False
        )

        # Run hyperopt searching engine
        LOGGER.info('Tuning Models (max_evals: %s):', max_evals)

        # try:
        result = fmin(
            fn=fmin_objective,
            space=self.search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            timeout=timeout_mins * 60,
            loss_threshold=-loss_threshold,
            trials=trials,
            verbose=True,
            show_progressbar=True,
            early_stop_fn=None
        )
        # except Exception as e:
        #     LOGGER.warning(
        #         'Exception occured while tuning hyperparameters.\n'
        #         'Exception: %s\n', e
        #     )

        # Evaluate dev models
        # self.evaluate_dev_models(
        #     X_train=X_train,
        #     X_test=X_test,
        #     y_train=y_train,
        #     y_test=y_test,
        #     debug=debug
        # )

        # Save dev models
        for model in self.models:
            if model.stage == 'development':
                model.save()

        # Update registry_dict with new development models
        self.model_registry.registry_dict["development"] = (
            self.model_registry.registry_dict["development"] +
            [m.model_id for m in self.models if m.stage == 'development']
        )

        LOGGER.info('New self.model_registry.registry_dict:\n%s\n', pformat(self.model_registry.registry_dict))

        # Update model stages
        # self.model_registry.update_model_stages(
        #     update_prod_model=False,
        #     debug=debug
        # )

        # Save self.registry_dict
        self.model_registry.save_registry_dict()

        # Show ModelRegistry
        LOGGER.info('%s', self.model_registry)

    def load(self) -> None:
        # Load registry
        self.model_registry.load_registry_dict()
        
        if self.model_registry.registry_dict is None:
            raise Exception('arreloco')

        # Load Registry Models
        self.models: List[Model] = (
            [self.model_registry.load_prod_model()]
            + self.model_registry.load_staging_models()
        )

        # Drop null models
        self.models = [m for m in self.models if m is not None]

        # if self.model_registry.registry_dict is not None:
            
        # else:
        #     LOGGER.warning('self.model_registry.registry_dict is None.')
        #     self.models: List[Model] = []
