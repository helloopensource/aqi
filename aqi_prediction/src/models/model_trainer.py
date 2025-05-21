"""
Model training and prediction module using AutoGluon.
"""
import os
import logging
import shutil
from typing import Dict, List, Optional, Tuple, Any, Union

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon.common import space

from .air_quality import AQScenario
from ..config.settings import (
    MODEL_DIR, 
    DEFAULT_ML_TARGET_LABEL, 
    REGRESSION_TARGET_LABELS, 
    DEFAULT_REGRESSION_EVAL_METRIC
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and manages AutoGluon models for AQI prediction.
    """
    def __init__(
        self, 
        scenario: AQScenario, 
        target_label: str = DEFAULT_ML_TARGET_LABEL,
        is_regression: bool = False
    ):
        """
        Initialize the model trainer.
        
        Args:
            scenario: AQ scenario to use for training
            target_label: ML target label (classification or regression target)
            is_regression: Whether the model is a regression model 
        """
        self.scenario = scenario
        self.target_label = target_label
        self.is_regression = is_regression
        
        # For regression models, include the parameter name in the model path
        if is_regression:
            self.model_path = os.path.join(
                MODEL_DIR, 
                f"aq_{scenario.name}_{scenario.year_start}-{scenario.year_end}_regression_{target_label}"
            )
        else:
            self.model_path = os.path.join(
                MODEL_DIR, 
                f"aq_{scenario.name}_{scenario.year_start}-{scenario.year_end}"
            )
        
        self._predictor = None  # Cache for the loaded model
        self._model_loaded = False  # Flag to track if model is loaded
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def train_model(
        self, 
        train_df: pd.DataFrame, 
        validation_df: pd.DataFrame, 
        time_limit_secs: Optional[int] = None,
        **kwargs
    ) -> TabularPredictor:
        """
        Train an AutoGluon model using the provided data.
        
        Args:
            train_df: Training data
            validation_df: Validation data
            time_limit_secs: Time limit for training in seconds
            **kwargs: Additional arguments to pass to AutoGluon
            
        Returns:
            Trained TabularPredictor
        """
        if train_df.empty:
            logger.error("Cannot train model: training DataFrame is empty")
            raise ValueError("Training DataFrame is empty")
        
        if self.target_label not in train_df.columns:
            logger.error(f"Target label '{self.target_label}' not found in training data")
            raise ValueError(f"Target label '{self.target_label}' not found in training data")
        
        # Handle regression vs classification differently
        if self.is_regression:
            logger.info(f"Training regression model to predict {self.target_label}")
            
            # Default hyperparameters for regression if not provided
            hyperparameters = kwargs.get("hyperparameters", {
                'GBM': {
                       'num_boost_round': 100,
                       'num_leaves': space.Int(lower=26, upper=66, default=36),
                },
                'RF': {},
                'XGB': {},
                'CAT': {},
                'NN_TORCH': {
                    'num_epochs': 50,
                    'learning_rate': space.Real(1e-4, 1e-2, default=1e-3, log=True),
                    'activation': space.Categorical('relu', 'softrelu', 'tanh'),
                    'dropout_prob': space.Real(0.0, 0.5, default=0.1),
                },
                'FASTAI': {}
            })
            
            # Default problem type for air quality regression
            problem_type = "regression"
            
            # Default eval metric for regression
            eval_metric = kwargs.get("eval_metric", DEFAULT_REGRESSION_EVAL_METRIC)
            
        else:
            # Check if the target variable has enough unique values for classification
            unique_values = train_df[self.target_label].unique()
            if len(unique_values) < 2:
                logger.warning(f"Target variable '{self.target_label}' only contains one unique value: {unique_values}")
                logger.warning("Creating synthetic data to enable binary classification...")
                
                # Determine which class is missing (assuming binary classification with 0/1)
                if 0 in unique_values and 1 not in unique_values:
                    missing_class = 1  # All data points are healthy (0), need to create unhealthy (1)
                    existing_class = 0
                elif 1 in unique_values and 0 not in unique_values:
                    missing_class = 0  # All data points are unhealthy (1), need to create healthy (0)
                    existing_class = 1
                else:
                    logger.error("Unexpected target variable values")
                    raise ValueError(f"Unexpected target variable values: {unique_values}")
                
                # Create synthetic examples for the missing class
                synthetic_count = min(10, len(train_df))  # Create a small number of synthetic examples
                
                # Copy some existing rows and change their target value
                synthetic_rows = train_df.sample(synthetic_count, replace=False).copy()
                synthetic_rows[self.target_label] = missing_class
                
                # If creating unhealthy examples, modify some features to make them more realistic
                if missing_class == 1:  # Creating unhealthy examples
                    # For air quality data, adjust relevant weather features
                    if 'TEMP_AVG' in synthetic_rows.columns:
                        # Higher temperatures often correlate with worse air quality
                        synthetic_rows['TEMP_AVG'] = synthetic_rows['TEMP_AVG'] * 1.2
                    
                    if 'WDSP' in synthetic_rows.columns:
                        # Lower wind speeds often correlate with worse air quality
                        synthetic_rows['WDSP'] = synthetic_rows['WDSP'] * 0.7
                
                # Combine original and synthetic data
                augmented_train_df = pd.concat([train_df, synthetic_rows], ignore_index=True)
                
                # Also add synthetic examples to validation data
                if not validation_df.empty:
                    synthetic_val_rows = validation_df.sample(
                        min(5, len(validation_df)), replace=False
                    ).copy()
                    synthetic_val_rows[self.target_label] = missing_class
                    augmented_validation_df = pd.concat(
                        [validation_df, synthetic_val_rows], ignore_index=True
                    )
                else:
                    augmented_validation_df = validation_df
                
                logger.info(f"Created {synthetic_count} synthetic examples for class {missing_class}")
                logger.info(f"Augmented training data shape: {augmented_train_df.shape}")
                
                # Use the augmented data for training
                train_df = augmented_train_df
                validation_df = augmented_validation_df
            
            # Default hyperparameters for classification if not provided
            hyperparameters = kwargs.get("hyperparameters", {
                'GBM': {
                       'num_boost_round': 100,  # number of boosting rounds (controls training time of GBM models)
                       'num_leaves': space.Int(lower=26, upper=66, default=36),  # number of leaves in trees (integer hyperparameter)
                },
                'RF': {
                    'n_estimators': space.Int(lower=100, upper=500, default=200),
                    'max_depth': space.Int(lower=3, upper=10, default=6),
                    'min_samples_split': space.Int(lower=2, upper=10, default=5),
                    'min_samples_leaf': space.Int(lower=1, upper=5, default=2),
                    'max_features': space.Real(0.1, 1.0, default=0.5),
                    'bootstrap': space.Categorical(True, False),
                },
                'XGB': {
                    'num_boost_round': 100,
                    'max_depth': space.Int(lower=3, upper=10, default=6),
                    'learning_rate': space.Real(1e-4, 1e-2, default=1e-3, log=True),
                    'subsample': space.Real(0.5, 1.0, default=0.8),
                    'colsample_bytree': space.Real(0.5, 1.0, default=0.8),
                    'reg_alpha': space.Real(0.0, 1.0, default=0.0),
                    'reg_lambda': space.Real(0.0, 1.0, default=1.0),
                },
                'CAT': {
                    'num_trees': space.Int(lower=100, upper=500, default=200),
                    'max_depth': space.Int(lower=3, upper=10, default=6),
                    'learning_rate': space.Real(1e-4, 1e-2, default=1e-3, log=True),
                    'l2_leaf_reg': space.Real(0.0, 1.0, default=1.0),
                    'random_strength': space.Real(0.0, 1.0, default=0.5),
                },
                'NN_TORCH': {
                    'num_epochs': 50,  # number of training epochs (controls training time of NN models)
                    'learning_rate': space.Real(1e-4, 1e-2, default=1e-3, log=True),  # learning rate used in training (real-valued hyperparameter searched on log-scale)
                    'activation': space.Categorical('relu', 'softrelu', 'tanh'),  # activation function used in NN (categorical hyperparameter, default = first entry)
                    'dropout_prob': space.Real(0.0, 0.5, default=0.1),  # dropout probability (real-valued hyperparameter)
                }
            })
            
            # Default problem type for air quality classification
            problem_type = kwargs.get("problem_type", "binary")
            
            # Default eval metric for classification
            eval_metric = kwargs.get("eval_metric", "accuracy")
        
        logger.info(f"Training model for scenario: {self.scenario.name}")
        logger.info(f"Target label: {self.target_label}, Problem type: {problem_type}")
        logger.info(f"Training data shape: {train_df.shape}")
        logger.info(f"Validation data shape: {validation_df.shape}")
        logger.info(f"Model will be saved to: {self.model_path}")
        
        try:
            # Set a default time limit if none provided
            if time_limit_secs is None:
                time_limit_secs = 300  # 5 minutes default
            
            # Create predictor with time limit
            predictor = TabularPredictor(
                label=self.target_label,
                path=self.model_path,
                problem_type=problem_type,
                eval_metric=eval_metric
            )
            
            search_strategy = 'auto'  # to tune hyperparameters using random search routine with a local scheduler
            hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
                'num_trials': 10,  # Number of HPO trials
                'scheduler' : 'local',
                'searcher': search_strategy,
            }  # Refer to TabularPredictor.fit docstring for all valid values

            # Train with timeout and k-fold cross validation
            predictor.fit(
                auto_stack=True,
                train_data=train_df,
                tuning_data=validation_df,
                time_limit=time_limit_secs,
                hyperparameters=hyperparameters,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                num_bag_folds=5,  # Enable 5-fold cross validation
                num_bag_sets=1,  # Number of times to repeat k-fold bagging
                verbosity=3,
                use_bag_holdout=True  # Allow validation data with bagging
            )
            
            # Generate feature importance
            if not validation_df.empty:
                try:
                    feature_importance = predictor.feature_importance(validation_df)
                    feature_importance.to_csv(os.path.join(self.model_path, "feature_importance.csv"))
                    logger.info("Feature importance saved to model directory")
                except Exception as e:
                    logger.warning(f"Could not generate feature importance: {str(e)}")
            
            # Generate leaderboard
            try:
                leaderboard = predictor.leaderboard(validation_df, silent=True)
                leaderboard.to_csv(os.path.join(self.model_path, "leaderboard.csv"))
                logger.info("Model leaderboard saved to model directory")
            except Exception as e:
                logger.warning(f"Could not generate leaderboard: {str(e)}")
            
            # Evaluate model
            try:
                evaluation = predictor.evaluate(validation_df, auxiliary_metrics=True)
                with open(os.path.join(self.model_path, "evaluation.txt"), "w") as f:
                    f.write(str(evaluation))
                logger.info("Model evaluation saved to model directory")
            except Exception as e:
                logger.warning(f"Could not evaluate model: {str(e)}")
            
            logger.info("Model training completed successfully")
            return predictor
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def load_model(self) -> Optional[TabularPredictor]:
        """
        Load a trained model from disk.
        
        Returns:
            Loaded TabularPredictor or None if not found
        """
        if self._model_loaded and self._predictor is not None:
            logger.info("Using cached model")
            return self._predictor

        if not os.path.exists(self.model_path):
            logger.warning(f"Model path does not exist: {self.model_path}")
            return None
        
        try:
            logger.info(f"Loading model from: {self.model_path}")
            # Clear any existing model from memory
            if self._predictor is not None:
                del self._predictor
                import gc
                gc.collect()
            
            # Load model with minimal memory usage
            self._predictor = TabularPredictor.load(self.model_path, require_version_match=False)
            self._model_loaded = True
            return self._predictor
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the trained model.
        
        Args:
            data: DataFrame with features
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Starting prediction process...")
        
        if data.empty:
            logger.warning("No data provided for prediction")
            return pd.DataFrame()
        
        # Log data shape and columns for debugging
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input data columns: {data.columns.tolist()}")
        
        # Load model if not already loaded
        logger.info("Loading model...")
        predictor = self.load_model()
        if predictor is None:
            logger.error("No model found for prediction")
            return pd.DataFrame()
        
        try:
            # Log model features for comparison
            model_features = predictor.features() if hasattr(predictor, 'features') else []
            logger.info(f"Model features: {model_features}")
            
            # Check for missing required features
            if hasattr(predictor, 'features'):
                missing_features = [f for f in predictor.features() if f not in data.columns and f != self.target_label]
                if missing_features:
                    logger.warning(f"Missing features in input data: {missing_features}")
                    # Add missing features with default values
                    for feature in missing_features:
                        logger.info(f"Adding missing feature {feature} with default value 0")
                        data[feature] = 0
            
            # Make predictions - optimize the prediction process
            logger.info("Making actual prediction call...")
            # Use a smaller dataset for faster predictions if possible
            if data.shape[0] > 100:
                logger.info(f"Large dataset detected ({data.shape[0]} rows). Using batch processing.")
                batch_size = 100
                num_batches = (data.shape[0] + batch_size - 1) // batch_size
                predictions_list = []
                proba_list = []
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, data.shape[0])
                    batch_data = data.iloc[start_idx:end_idx]
                    
                    logger.info(f"Processing batch {i+1}/{num_batches} (rows {start_idx+1}-{end_idx})")
                    if self.is_regression:
                        batch_preds = predictor.predict(batch_data)
                        predictions_list.append(batch_preds)
                    else:
                        batch_preds = predictor.predict(batch_data)
                        batch_proba = predictor.predict_proba(batch_data)
                        predictions_list.append(batch_preds)
                        proba_list.append(batch_proba)
                
                if self.is_regression:
                    # Combine regression predictions
                    predictions = pd.concat(predictions_list, ignore_index=True)
                else:
                    # Combine classification predictions
                    predictions = pd.concat(predictions_list, ignore_index=True)
                    prediction_probs = pd.concat(proba_list, ignore_index=True)
            else:
                # Small dataset, predict all at once
                if self.is_regression:
                    # For regression models, we want the actual predicted values
                    logger.info(f"Making regression predictions for {self.target_label}")
                    predictions = predictor.predict(data)
                else:
                    # For classification models, get both class and probability
                    logger.info("Making classification predictions")
                    predictions = predictor.predict(data)
                    prediction_probs = predictor.predict_proba(data)
            
            logger.info("Prediction completed successfully")
            
            # Create results dataframe based on model type
            if self.is_regression:
                # For regression models
                results = pd.DataFrame({f"{self.target_label}_predicted": predictions})
                
                # Add prediction uncertainty if available
                try:
                    logger.info("Adding prediction intervals...")
                    prediction_intervals = predictor.predict_quantile(data, quantiles=[0.1, 0.9])
                    results[f"{self.target_label}_lower_bound"] = prediction_intervals[0.1]
                    results[f"{self.target_label}_upper_bound"] = prediction_intervals[0.9]
                    logger.info("Added prediction uncertainty intervals")
                except Exception as e:
                    logger.warning(f"Could not generate prediction intervals: {str(e)}")
                
                # Add EPA AQI category for the pollutant values if this is a major pollutant
                param_name = self.target_label.split('_')[0] if '_value' in self.target_label else None
                if param_name and param_name in ['pm25', 'pm10', 'o3']:
                    try:
                        from ..config.settings import UNHEALTHY_THRESHOLDS
                        threshold = UNHEALTHY_THRESHOLDS.get(param_name, 35.5)
                        results['isUnhealthy_predicted'] = np.where(results[f"{self.target_label}_predicted"] > threshold, 1, 0)
                        logger.info(f"Added isUnhealthy classification using threshold {threshold} for {param_name}")
                    except Exception as e:
                        logger.warning(f"Could not add isUnhealthy classification: {str(e)}")
            else:
                # For classification models
                # Get the probability of the positive class (unhealthy)
                positive_class_name = '1'  # Assuming binary classification with 1 = unhealthy
                try:
                    if positive_class_name in prediction_probs.columns:
                        unhealthy_probs = prediction_probs[positive_class_name]
                    else:
                        # If column names are not as expected, just take the second column
                        # (typical for binary classification where the second column is positive class)
                        unhealthy_probs = prediction_probs.iloc[:, 1]
                except Exception as e:
                    logger.warning(f"Could not extract positive class probabilities: {str(e)}")
                    unhealthy_probs = pd.Series([0.5] * len(predictions))  # Default fallback
                
                # Create results dataframe
                results = pd.DataFrame({
                    'isUnhealthy_predicted': predictions,
                    'unhealthy_probability': unhealthy_probs
                })
            
            logger.info(f"Prediction result shape: {results.shape}")
            logger.info(f"Prediction result columns: {results.columns.tolist()}")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        predictor = self.load_model()
        
        if predictor is None:
            logger.warning("Cannot get model info: model not found")
            return {"error": "Model not found"}
        
        try:
            # Get basic info first
            model_info = {
                "scenario": self.scenario.name,
                "target": self.target_label,
                "path": self.model_path,
            }
            
            # Try to get additional info with individual error handling
            try:
                model_info["problem_type"] = str(predictor.problem_type)
            except Exception as e:
                logger.warning(f"Could not get problem type: {str(e)}")
                model_info["problem_type"] = "unknown"
            
            try:
                model_info["eval_metric"] = str(predictor.eval_metric)
            except Exception as e:
                logger.warning(f"Could not get eval metric: {str(e)}")
                model_info["eval_metric"] = "unknown"
            
            try:
                model_info["features"] = predictor.features()
            except Exception as e:
                logger.warning(f"Could not get features: {str(e)}")
                model_info["features"] = []
            
            # Try to get model names
            try:
                # Different versions of AutoGluon have different methods for getting model names
                if hasattr(predictor, 'get_model_names'):
                    model_info["model_types"] = list(predictor.get_model_names())
                elif hasattr(predictor, 'model_names'):
                    model_info["model_types"] = list(predictor.model_names)
                elif hasattr(predictor, 'get_model_best'):
                    best_model = predictor.get_model_best()
                    model_info["model_types"] = [best_model]
                elif hasattr(predictor, 'model_best'):
                    model_info["model_types"] = [str(predictor.model_best)]
                elif hasattr(predictor, 'model_classes_'):
                    model_info["model_types"] = list(predictor.model_classes_.keys())
                else:
                    # Get all properties and methods of the predictor to find model info
                    import inspect
                    properties = [
                        attr for attr in dir(predictor) 
                        if not attr.startswith('_') and not callable(getattr(predictor, attr))
                    ]
                    logger.info(f"Available properties: {properties}")
                    
                    if 'model_graph' in properties and hasattr(predictor.model_graph, 'nodes'):
                        model_info["model_types"] = list(predictor.model_graph.nodes.keys())
                    else:
                        model_info["model_types"] = ["GBM", "RF", "XGB", "CAT", "NN_TORCH", "FASTAI"]
            except Exception as e:
                logger.warning(f"Could not get model types: {str(e)}")
                model_info["model_types"] = ["GBM", "RF", "XGB", "CAT", "NN_TORCH", "FASTAI"]
            
            # Try to get best model
            try:
                if hasattr(predictor, 'get_model_best'):
                    model_info["best_model"] = str(predictor.get_model_best())
                elif hasattr(predictor, 'model_best'):
                    model_info["best_model"] = str(predictor.model_best)
                elif hasattr(predictor, 'best_model'):
                    model_info["best_model"] = str(predictor.best_model)
                elif 'model_types' in model_info and len(model_info['model_types']) > 0:
                    model_info["best_model"] = model_info['model_types'][0]
                else:
                    model_info["best_model"] = "GBM"  # Default to GBM if can't determine
            except Exception as e:
                logger.warning(f"Could not get best model: {str(e)}")
                model_info["best_model"] = "GBM"  # Default to GBM if can't determine
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}
    
    def delete_model(self) -> bool:
        """
        Delete the model from disk.
        
        Returns:
            True if deleted successfully, False otherwise
        """
        if not os.path.exists(self.model_path):
            logger.warning(f"Cannot delete model: path does not exist: {self.model_path}")
            return False
        
        try:
            shutil.rmtree(self.model_path)
            logger.info(f"Model deleted: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting model: {str(e)}")
            return False 