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
            # Initialize and train the predictor
            predictor = TabularPredictor(
                label=self.target_label, 
                path=self.model_path,
                problem_type=problem_type,
                eval_metric=eval_metric
            )
            
            predictor.fit(
                train_data=train_df,
                tuning_data=validation_df,
                time_limit=time_limit_secs,
                hyperparameters=hyperparameters,
                auto_stack=True,
                use_bag_holdout=True,
                verbosity=2
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
        if not os.path.exists(self.model_path):
            logger.warning(f"Model path does not exist: {self.model_path}")
            return None
        
        try:
            logger.info(f"Loading model from: {self.model_path}")
            predictor = TabularPredictor.load(self.model_path)
            return predictor
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
        predictor = self.load_model()
        
        if predictor is None:
            logger.error("Cannot make predictions: model not found")
            raise ValueError("Model not found")
        
        try:
            # Check if we need to add missing columns required by the model
            if hasattr(predictor, 'features'):
                required_features = predictor.features()
                # Add missing columns with default values
                missing_columns = [col for col in required_features if col not in data.columns]
                if missing_columns:
                    for col in missing_columns:
                        logger.warning(f"Adding missing column: {col} with default value of 0")
                        data[col] = 0
                    logger.info(f"Added {len(missing_columns)} missing columns with default values")
            
            # Handle version differences with feature preprocessing
            try:
                # Some versions have transform_features method
                if hasattr(predictor, 'transform_features'):
                    data = predictor.transform_features(data)
                    logger.info("Applied feature transformation using predictor's transform_features")
            except Exception as e:
                logger.warning(f"Could not transform features: {str(e)}. Using raw features.")
            
            # Make predictions
            predictions = predictor.predict(data)
            
            # Get prediction probabilities for binary classification
            if predictor.problem_type == 'binary':
                try:
                    probabilities = predictor.predict_proba(data)
                    result = pd.DataFrame({
                        'prediction': predictions,
                        'probability_unhealthy': probabilities.iloc[:, 1]
                    })
                except Exception as e:
                    logger.warning(f"Could not get prediction probabilities: {str(e)}")
                    result = pd.DataFrame({
                        'prediction': predictions
                    })
            else:
                result = pd.DataFrame({
                    'prediction': predictions
                })
            
            return result
            
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
            model_info = {
                "scenario": self.scenario.name,
                "target": self.target_label,
                "path": self.model_path,
                "problem_type": predictor.problem_type,
                "eval_metric": predictor.eval_metric,
                "features": predictor.features()
            }
            
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
            
            # Try to get leaderboard
            try:
                model_info["best_model"] = predictor.get_model_best()
            except Exception as e:
                logger.warning(f"Could not get best model: {str(e)}")
            
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