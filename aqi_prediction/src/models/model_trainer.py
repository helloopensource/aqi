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

from .air_quality import AQScenario
from ..config.settings import MODEL_DIR, DEFAULT_ML_TARGET_LABEL

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and manages AutoGluon models for AQI prediction.
    """
    def __init__(self, scenario: AQScenario, target_label: str = DEFAULT_ML_TARGET_LABEL):
        """
        Initialize the model trainer.
        
        Args:
            scenario: AQ scenario to use for training
            target_label: ML target label (classification target)
        """
        self.scenario = scenario
        self.target_label = target_label
        self.model_path = os.path.join(MODEL_DIR, f"aq_{scenario.name}_{scenario.year_start}-{scenario.year_end}")
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
        
        # Default hyperparameters if not provided
        hyperparameters = kwargs.get("hyperparameters", {
            'GBM': {
                'num_boost_round': 100,
                'num_leaves': 31,
            },
            'RF': {},
            'XGB': {},
            'CAT': {},
            'NN_TORCH': {},
            'FASTAI': {}
        })
        
        # Default problem type for air quality classification
        problem_type = kwargs.get("problem_type", "binary")
        
        # Default eval metric
        eval_metric = kwargs.get("eval_metric", "accuracy")
        
        logger.info(f"Training model for scenario: {self.scenario.name}")
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
            
            # Train with timeout
            predictor.fit(
                train_data=train_df,
                tuning_data=validation_df,
                time_limit=time_limit_secs,
                hyperparameters=hyperparameters
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
        
        if not self._model_loaded or self._predictor is None:
            logger.info("Loading model...")
            self._predictor = self.load_model()
        
        if self._predictor is None:
            logger.error("Cannot make predictions: model not found")
            raise ValueError("Model not found")
        
        try:
            # Check if we need to add missing columns required by the model
            if hasattr(self._predictor, 'features'):
                logger.info("Checking required features...")
                required_features = self._predictor.features()
                # Find missing columns
                missing_columns = [col for col in required_features if col not in data.columns]
                if missing_columns:
                    logger.error(f"Missing required features for prediction: {missing_columns}")
                    raise ValueError(f"Missing required features for prediction: {missing_columns}")
            
            # Handle version differences with feature preprocessing
            try:
                logger.info("Preprocessing features...")
                # Some versions have transform_features method
                if hasattr(self._predictor, 'transform_features'):
                    data = self._predictor.transform_features(data)
                    logger.info("Feature transformation completed")
                else:
                    logger.info("No feature transformation needed")
            except Exception as e:
                logger.warning(f"Could not transform features: {str(e)}. Using raw features.")
            
            # Make predictions with minimal memory usage
            logger.info("Making predictions...")
            try:
                predictions = self._predictor.predict(data)
                logger.info("Getting prediction probabilities...")
                probabilities = self._predictor.predict_proba(data)
                
                # Create result DataFrame
                logger.info("Creating result DataFrame...")
                result = pd.DataFrame({
                    'prediction': predictions,
                    'probability_unhealthy': probabilities[1] if len(probabilities.shape) > 1 else probabilities
                })
                
                logger.info("Prediction process completed successfully")
                return result
            except Exception as e:
                logger.error(f"Error during prediction step: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
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
                    model_info["model_types"] = [predictor.get_model_best()]
                else:
                    model_info["model_types"] = ["unknown"]
            except Exception as e:
                logger.warning(f"Could not get model types: {str(e)}")
                model_info["model_types"] = ["unknown"]
            
            # Try to get best model
            try:
                model_info["best_model"] = str(predictor.get_model_best())
            except Exception as e:
                logger.warning(f"Could not get best model: {str(e)}")
                model_info["best_model"] = "unknown"
            
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