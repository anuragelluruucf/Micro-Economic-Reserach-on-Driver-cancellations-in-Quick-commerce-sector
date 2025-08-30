#!/usr/bin/env python3
"""
Predictive Models for Strategic Cancellation Detection
Includes both full model and cold-start model implementations
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, 
                           classification_report, confusion_matrix, auc)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class StrategicCancellationPredictor:
    """Main class for strategic cancellation prediction models."""
    
    def __init__(self, data_path='../data/shadowfax_processed-data-final.csv'):
        self.data_path = data_path
        self.df = None
        self.full_model = None
        self.balanced_model = None
        self.cold_start_model = None
        self.feature_columns = None
        self.cold_start_features = None
        
    def load_and_prepare_data(self):
        """Load data and create necessary features."""
        print("Loading and preparing data...")
        self.df = pd.read_csv(self.data_path)
        
        # Create time-based features
        self.df['hour'] = pd.to_datetime(self.df['order_time']).dt.hour
        self.df['is_peak_hour'] = self.df['hour'].isin([12, 13, 14, 18, 19, 20, 21])
        
        # Create strategic labels
        self._create_strategic_labels()
        
        # Engineer additional features
        self._engineer_features()
        
        print(f"Data prepared. Shape: {self.df.shape}")
        print(f"Strategic orders: {self.df['is_strategic_order'].sum():,} ({self.df['is_strategic_order'].mean():.2%})")
        
    def _create_strategic_labels(self):
        """Create strategic rider and order labels."""
        # Calculate rider-level metrics
        rider_stats = self.df.groupby('rider_id').agg({
            'cancelled': ['sum', 'count'],
            'cancel_after_pickup': 'sum'
        }).reset_index()
        rider_stats.columns = ['rider_id', 'total_cancellations', 'total_orders', 'post_pickup_cancels']
        
        # Count bike issues
        bike_issues = self.df[
            self.df['reason_text'] == 'Cancel order due to bike issue'
        ].groupby('rider_id').size()
        rider_stats['bike_issues'] = rider_stats['rider_id'].map(bike_issues).fillna(0)
        
        # Calculate rates
        rider_stats['post_pickup_rate'] = np.where(
            rider_stats['total_cancellations'] > 0,
            rider_stats['post_pickup_cancels'] / rider_stats['total_cancellations'],
            0
        )
        rider_stats['bike_issue_rate'] = np.where(
            rider_stats['total_cancellations'] > 0,
            rider_stats['bike_issues'] / rider_stats['total_cancellations'],
            0
        )
        
        # Identify strategic riders
        strategic_riders = rider_stats[
            (rider_stats['bike_issues'] >= 2) &
            (rider_stats['post_pickup_rate'] > 0.7) &
            (rider_stats['bike_issue_rate'] > 0.2)
        ]['rider_id'].values
        
        # Add to main dataframe
        self.df['is_strategic_rider'] = self.df['rider_id'].isin(strategic_riders)
        self.df['is_strategic_order'] = self.df['is_strategic_rider'] & (self.df['cancelled'] == 1)
        
        # Merge rider stats
        self.df = self.df.merge(
            rider_stats[['rider_id', 'bike_issue_rate', 'post_pickup_rate']], 
            on='rider_id', 
            how='left'
        )
        
    def _engineer_features(self):
        """Engineer additional features for modeling."""
        # Time-based features
        self.df['time_to_accept'] = (
            pd.to_datetime(self.df['accept_time']) - 
            pd.to_datetime(self.df['allot_time'])
        ).dt.total_seconds() / 60
        
        self.df['time_to_pickup'] = (
            pd.to_datetime(self.df['pickup_time']) - 
            pd.to_datetime(self.df['accept_time'])
        ).dt.total_seconds() / 60
        
        # Clean outliers
        self.df.loc[self.df['time_to_accept'] < 0, 'time_to_accept'] = np.nan
        self.df.loc[self.df['time_to_accept'] > 60, 'time_to_accept'] = np.nan
        self.df.loc[self.df['time_to_pickup'] < 0, 'time_to_pickup'] = np.nan
        self.df.loc[self.df['time_to_pickup'] > 120, 'time_to_pickup'] = np.nan
        
        # Interaction features
        self.df['distance_x_peak'] = self.df['total_distance'] * self.df['is_peak_hour']
        self.df['distance_x_session'] = self.df['total_distance'] * self.df['session_time']
        
        # Fill missing values
        numeric_cols = ['time_to_accept', 'time_to_pickup', 'session_time']
        for col in numeric_cols:
            self.df[col].fillna(self.df[col].median(), inplace=True)
    
    def train_full_model(self):
        """Train the full strategic detection model with all features."""
        print("\n" + "="*60)
        print("TRAINING FULL MODEL")
        print("="*60)
        
        # Define features
        self.feature_columns = [
            'lifetime_order_count', 'bike_issue_rate', 'post_pickup_rate',
            'total_distance', 'first_mile_distance', 'last_mile_distance',
            'session_time', 'hour', 'is_peak_hour',
            'time_to_accept', 'time_to_pickup',
            'distance_x_peak', 'distance_x_session'
        ]
        
        # Prepare data
        X = self.df[self.feature_columns].fillna(0)
        y = self.df['is_strategic_order'].astype(int)
        
        # Temporal split
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        print(f"Class balance: {y_train.mean():.2%} strategic")
        
        # Train model
        self.full_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.full_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = self.full_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Get predictions at default threshold
        y_pred = self.full_model.predict(X_test)
        
        print(f"\nFull Model Performance:")
        print(f"AUC-ROC: {auc_score:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
        print(f"FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.full_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Features:")
        print(feature_importance.head())
        
        return {
            'model': self.full_model,
            'auc': auc_score,
            'feature_importance': feature_importance,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba
        }
    
    def train_balanced_model(self):
        """Train model with balanced sampling for better precision."""
        print("\n" + "="*60)
        print("TRAINING BALANCED MODEL")
        print("="*60)
        
        # Use same features as full model
        X = self.df[self.feature_columns].fillna(0)
        y = self.df['is_strategic_order'].astype(int)
        
        # Temporal split
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Downsample majority class to 3:1 ratio
        strategic_indices = np.where(y_train == 1)[0]
        non_strategic_indices = np.where(y_train == 0)[0]
        
        n_strategic = len(strategic_indices)
        n_sample_non_strategic = n_strategic * 3
        
        sampled_non_strategic = resample(
            non_strategic_indices,
            n_samples=n_sample_non_strategic,
            random_state=42
        )
        
        balanced_indices = np.concatenate([strategic_indices, sampled_non_strategic])
        np.random.shuffle(balanced_indices)
        
        X_train_balanced = X_train.iloc[balanced_indices]
        y_train_balanced = y_train.iloc[balanced_indices]
        
        print(f"Balanced training samples: {len(X_train_balanced):,}")
        print(f"Class balance: {y_train_balanced.mean():.1%} strategic")
        
        # Train model
        self.balanced_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
        
        self.balanced_model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate
        y_pred_proba = self.balanced_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Use higher threshold for balanced model
        threshold = 0.25
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        print(f"\nBalanced Model Performance (threshold={threshold}):")
        print(f"AUC-ROC: {auc_score:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'model': self.balanced_model,
            'auc': auc_score,
            'threshold': threshold,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba
        }
    
    def train_cold_start_model(self):
        """Train model for new riders without history."""
        print("\n" + "="*60)
        print("TRAINING COLD-START MODEL")
        print("="*60)
        
        # Get first orders only
        first_orders = self.df.groupby('rider_id').first().reset_index()
        
        # Features available at first order
        self.cold_start_features = [
            'total_distance', 'first_mile_distance', 'last_mile_distance',
            'session_time', 'time_to_accept', 'time_to_pickup',
            'hour', 'is_peak_hour'
        ]
        
        # Prepare data
        X = first_orders[self.cold_start_features].fillna(first_orders[self.cold_start_features].median())
        
        # For cold-start, we'll use a proxy label based on eventual behavior
        y = first_orders['is_strategic_rider'].astype(int)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        print(f"Strategic rider rate: {y_train.mean():.2%}")
        
        # Train model
        self.cold_start_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.cold_start_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = self.cold_start_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Use threshold of 0.3 for cold-start
        threshold = 0.3
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        print(f"\nCold-Start Model Performance (threshold={threshold}):")
        print(f"AUC-ROC: {auc_score:.3f}")
        
        # Calculate precision and recall at threshold
        precision = (y_test & y_pred).sum() / y_pred.sum() if y_pred.sum() > 0 else 0
        recall = (y_test & y_pred).sum() / y_test.sum() if y_test.sum() > 0 else 0
        
        print(f"Precision at {threshold}: {precision:.1%}")
        print(f"Recall at {threshold}: {recall:.1%}")
        print(f"Orders flagged: {y_pred.sum()}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.cold_start_features,
            'importance': self.cold_start_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop Features:")
        print(feature_importance)
        
        return {
            'model': self.cold_start_model,
            'auc': auc_score,
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'feature_importance': feature_importance
        }
    
    def cross_validate_models(self):
        """Perform cross-validation for all models."""
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS")
        print("="*60)
        
        # Prepare data
        X_full = self.df[self.feature_columns].fillna(0)
        y_full = self.df['is_strategic_order'].astype(int)
        
        # 5-fold stratified CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Full model CV
        full_scores = cross_val_score(
            self.full_model, X_full, y_full, cv=cv, scoring='roc_auc', n_jobs=-1
        )
        print(f"Full Model AUC: {full_scores.mean():.3f} (+/- {full_scores.std()*2:.3f})")
        
        # Cold-start model CV (on first orders)
        first_orders = self.df.groupby('rider_id').first().reset_index()
        X_cold = first_orders[self.cold_start_features].fillna(
            first_orders[self.cold_start_features].median()
        )
        y_cold = first_orders['is_strategic_rider'].astype(int)
        
        cold_scores = cross_val_score(
            self.cold_start_model, X_cold, y_cold, cv=cv, scoring='roc_auc', n_jobs=-1
        )
        print(f"Cold-Start Model AUC: {cold_scores.mean():.3f} (+/- {cold_scores.std()*2:.3f})")
    
    def save_models(self, path='models/'):
        """Save trained models."""
        import os
        os.makedirs(path, exist_ok=True)
        
        if self.full_model:
            joblib.dump(self.full_model, f'{path}full_model.pkl')
            print(f"Full model saved to {path}full_model.pkl")
            
        if self.balanced_model:
            joblib.dump(self.balanced_model, f'{path}balanced_model.pkl')
            print(f"Balanced model saved to {path}balanced_model.pkl")
            
        if self.cold_start_model:
            joblib.dump(self.cold_start_model, f'{path}cold_start_model.pkl')
            print(f"Cold-start model saved to {path}cold_start_model.pkl")
        
        # Save feature lists
        import json
        with open(f'{path}features.json', 'w') as f:
            json.dump({
                'full_features': self.feature_columns,
                'cold_start_features': self.cold_start_features
            }, f)
        print(f"Feature lists saved to {path}features.json")
    
    def predict_new_orders(self, new_data, model_type='full'):
        """Make predictions on new data."""
        if model_type == 'full':
            model = self.full_model
            features = self.feature_columns
        elif model_type == 'balanced':
            model = self.balanced_model
            features = self.feature_columns
        elif model_type == 'cold_start':
            model = self.cold_start_model
            features = self.cold_start_features
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if model is None:
            raise ValueError(f"Model {model_type} not trained yet")
        
        # Prepare features
        X = new_data[features].fillna(0)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Add to dataframe
        new_data[f'{model_type}_risk_score'] = y_pred_proba
        
        return new_data

def main():
    """Execute model training pipeline."""
    # Initialize predictor
    predictor = StrategicCancellationPredictor()
    
    # Load and prepare data
    predictor.load_and_prepare_data()
    
    # Train models
    full_results = predictor.train_full_model()
    balanced_results = predictor.train_balanced_model()
    cold_start_results = predictor.train_cold_start_model()
    
    # Cross-validation
    predictor.cross_validate_models()
    
    # Save models
    predictor.save_models()
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)
    
    # Summary
    print(f"\nModel Performance Summary:")
    print(f"Full Model AUC: {full_results['auc']:.3f}")
    print(f"Balanced Model AUC: {balanced_results['auc']:.3f}")
    print(f"Cold-Start Model AUC: {cold_start_results['auc']:.3f}")
    
    return predictor, {
        'full': full_results,
        'balanced': balanced_results,
        'cold_start': cold_start_results
    }

if __name__ == "__main__":
    predictor, results = main()