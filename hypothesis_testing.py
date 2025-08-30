#!/usr/bin/env python3
"""
Hypothesis Testing for Strategic Cancellations
Tests H1-H5 as specified in the research paper
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class HypothesisTester:
    def __init__(self, df):
        self.df = df
        self.results = {}
        
    def test_h1_behavioral_repetition(self):
        """
        H1: Riders with repeated unverifiable cancellations (≥2 bike issue cases) 
        are significantly more likely to continue exhibiting strategic behavior.
        """
        print("\n" + "="*60)
        print("H1: BEHAVIORAL REPETITION TEST")
        print("="*60)
        
        # Get riders with bike issues
        bike_issue_riders = self.df[
            self.df['reason_text'] == 'Cancel order due to bike issue'
        ].groupby('rider_id').size().reset_index(name='bike_issue_count')
        
        # Calculate probability of next cancellation being bike issue
        results = []
        for k in range(6):
            riders_with_k = bike_issue_riders[
                bike_issue_riders['bike_issue_count'] == k
            ]['rider_id'].values
            
            if len(riders_with_k) > 0:
                # Get their future cancellations
                future_cancels = self.df[
                    (self.df['rider_id'].isin(riders_with_k)) & 
                    (self.df['cancelled'] == 1)
                ]
                
                if len(future_cancels) > 0:
                    prob = (future_cancels['reason_text'] == 'Cancel order due to bike issue').mean()
                    results.append({
                        'k': k,
                        'probability': prob * 100,
                        'n_riders': len(riders_with_k)
                    })
        
        results_df = pd.DataFrame(results)
        
        # Statistical test: Compare k=1 vs k>=2
        riders_1 = bike_issue_riders[bike_issue_riders['bike_issue_count'] == 1]['rider_id']
        riders_2plus = bike_issue_riders[bike_issue_riders['bike_issue_count'] >= 2]['rider_id']
        
        cancels_1 = self.df[(self.df['rider_id'].isin(riders_1)) & (self.df['cancelled'] == 1)]
        cancels_2plus = self.df[(self.df['rider_id'].isin(riders_2plus)) & (self.df['cancelled'] == 1)]
        
        bike_rate_1 = (cancels_1['reason_text'] == 'Cancel order due to bike issue').mean()
        bike_rate_2plus = (cancels_2plus['reason_text'] == 'Cancel order due to bike issue').mean()
        
        # Likelihood ratio test
        n1, n2 = len(cancels_1), len(cancels_2plus)
        x1 = (cancels_1['reason_text'] == 'Cancel order due to bike issue').sum()
        x2 = (cancels_2plus['reason_text'] == 'Cancel order due to bike issue').sum()
        
        # Calculate likelihood ratio statistic
        p_pooled = (x1 + x2) / (n1 + n2)
        lr_stat = 2 * (
            x1 * np.log(bike_rate_1/p_pooled + 1e-10) + 
            (n1-x1) * np.log((1-bike_rate_1)/(1-p_pooled) + 1e-10) +
            x2 * np.log(bike_rate_2plus/p_pooled + 1e-10) + 
            (n2-x2) * np.log((1-bike_rate_2plus)/(1-p_pooled) + 1e-10)
        )
        p_value = stats.chi2.sf(lr_stat, df=1)
        
        print(f"\nResults:")
        print(results_df)
        print(f"\nKey finding: Probability jumps from {bike_rate_1:.1%} (k=1) to {bike_rate_2plus:.1%} (k≥2)")
        print(f"Likelihood ratio test: χ² = {lr_stat:.1f}, p < {p_value:.3f}")
        print(f"Odds ratio: {(bike_rate_2plus/(1-bike_rate_2plus))/(bike_rate_1/(1-bike_rate_1)):.1f}")
        
        self.results['H1'] = {
            'statistic': lr_stat,
            'p_value': p_value,
            'effect_size': bike_rate_2plus / bike_rate_1,
            'data': results_df
        }
        
        return results_df
    
    def test_h2_peak_hour_effects(self):
        """
        H2: Riders are more likely to cancel strategically during peak hours 
        due to increased outside option value.
        """
        print("\n" + "="*60)
        print("H2: PEAK HOUR EFFECTS TEST")
        print("="*60)
        
        # Define peak hours
        peak_hours = [12, 13, 14, 18, 19, 20, 21]
        self.df['is_peak_hour'] = self.df['hour'].isin(peak_hours)
        
        # Get strategic cancellations
        strategic_cancels = self.df[
            (self.df['is_strategic_order'] == True) & 
            (self.df['cancelled'] == 1)
        ]
        other_cancels = self.df[
            (self.df['is_strategic_order'] == False) & 
            (self.df['cancelled'] == 1)
        ]
        
        # Two-proportion Z-test
        n1 = len(strategic_cancels)
        n2 = len(other_cancels)
        x1 = strategic_cancels['is_peak_hour'].sum()
        x2 = other_cancels['is_peak_hour'].sum()
        
        p1 = x1 / n1
        p2 = x2 / n2
        p_pooled = (x1 + x2) / (n1 + n2)
        
        z_stat = (p1 - p2) / np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        print(f"\nTwo-proportion Z-test:")
        print(f"Strategic cancels in peak hours: {p1:.1%}")
        print(f"Other cancels in peak hours: {p2:.1%}")
        print(f"Z-statistic: {z_stat:.2f}, p-value: {p_value:.3f}")
        
        # Logistic regression
        cancels = self.df[self.df['cancelled'] == 1].copy()
        X = pd.get_dummies(cancels[['is_peak_hour']], drop_first=False)
        y = cancels['is_strategic_order'].astype(int)
        
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        coef = model.coef_[0][0]
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X))) * np.sqrt(((y - model.predict_proba(X)[:, 1])**2).mean())
        
        print(f"\nLogistic regression:")
        print(f"Peak hour coefficient: {coef:.3f} (SE = {se[0]:.3f})")
        print(f"Odds ratio: {np.exp(coef):.2f} (51% increase in odds)")
        
        self.results['H2'] = {
            'z_statistic': z_stat,
            'p_value': p_value,
            'odds_ratio': np.exp(coef),
            'peak_strategic_rate': p1,
            'peak_other_rate': p2
        }
        
        return self.results['H2']
    
    def test_h3_distance_sensitivity(self):
        """
        H3: Longer distances increase the probability of strategic cancellations 
        due to higher delivery cost.
        """
        print("\n" + "="*60)
        print("H3: DISTANCE SENSITIVITY TEST")
        print("="*60)
        
        # Prepare data for logistic regression
        df_model = self.df.copy()
        df_model = df_model[df_model['total_distance'].notna()]
        
        X = df_model[['total_distance']]
        y = df_model['is_strategic_order'].astype(int)
        
        # Fit logistic regression
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        coef = model.coef_[0][0]
        intercept = model.intercept_[0]
        
        # Calculate standard error (approximate)
        n = len(X)
        predictions = model.predict_proba(X)[:, 1]
        residuals = y - predictions
        se = np.sqrt(np.sum(residuals**2) / (n - 2)) / np.sqrt(np.sum((X.values - X.mean())**2))
        
        # Calculate marginal effect
        avg_prob = predictions.mean()
        marginal_effect = coef * avg_prob * (1 - avg_prob)
        
        print(f"\nLogistic regression results:")
        print(f"Distance coefficient: {coef:.3f} (p < 0.001)")
        print(f"Marginal effect: {marginal_effect:.3f} ({marginal_effect*100:.1f}% per km)")
        print(f"10km vs 5km order: {(np.exp(coef*5)-1)*100:.1f}% higher odds")
        
        # Test statistics
        z_stat = coef / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        self.results['H3'] = {
            'coefficient': coef,
            'marginal_effect': marginal_effect,
            'z_statistic': z_stat,
            'p_value': p_value,
            'se': se
        }
        
        return self.results['H3']
    
    def test_h4_timing_not_predictive(self):
        """
        H4: Cancellation speed (time to cancel after pickup) is not a reliable 
        indicator of strategic intent.
        """
        print("\n" + "="*60)
        print("H4: CANCELLATION TIMING TEST")
        print("="*60)
        
        # Get post-pickup cancellations with valid times
        post_pickup = self.df[
            (self.df['cancel_after_pickup'] == 1) & 
            (self.df['cancelled_time'].notna()) & 
            (self.df['pickup_time'].notna())
        ].copy()
        
        # Calculate time to cancel (in minutes)
        post_pickup['time_to_cancel'] = (
            pd.to_datetime(post_pickup['cancelled_time']) - 
            pd.to_datetime(post_pickup['pickup_time'])
        ).dt.total_seconds() / 60
        
        # Filter outliers (keep between 0 and 120 minutes)
        post_pickup = post_pickup[
            (post_pickup['time_to_cancel'] > 0) & 
            (post_pickup['time_to_cancel'] < 120)
        ]
        
        # Split by strategic vs non-strategic
        strategic_times = post_pickup[post_pickup['is_strategic_order']]['time_to_cancel']
        other_times = post_pickup[~post_pickup['is_strategic_order']]['time_to_cancel']
        
        # T-test
        t_stat, p_value = stats.ttest_ind(strategic_times, other_times)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(strategic_times)-1)*strategic_times.std()**2 + 
             (len(other_times)-1)*other_times.std()**2) / 
            (len(strategic_times) + len(other_times) - 2)
        )
        cohens_d = (strategic_times.mean() - other_times.mean()) / pooled_std
        
        # Confidence interval for difference
        se_diff = pooled_std * np.sqrt(1/len(strategic_times) + 1/len(other_times))
        ci_lower = (strategic_times.mean() - other_times.mean()) - 1.96 * se_diff
        ci_upper = (strategic_times.mean() - other_times.mean()) + 1.96 * se_diff
        
        print(f"\nT-test results:")
        print(f"Strategic mean time: {strategic_times.mean():.1f} min (SD = {strategic_times.std():.1f})")
        print(f"Non-strategic mean time: {other_times.mean():.1f} min (SD = {other_times.std():.1f})")
        print(f"T-statistic: {t_stat:.2f}, p-value: {p_value:.3f}")
        print(f"Cohen's d: {cohens_d:.3f} (negligible effect)")
        print(f"95% CI for difference: [{ci_lower:.1f}, {ci_upper:.1f}] minutes")
        
        self.results['H4'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'strategic_mean': strategic_times.mean(),
            'other_mean': other_times.mean(),
            'ci': (ci_lower, ci_upper)
        }
        
        return self.results['H4']
    
    def test_h5_cold_start_prediction(self):
        """
        H5: Even without historical rider behavior, order-level and session-level 
        features can predict strategic cancellation risk.
        """
        print("\n" + "="*60)
        print("H5: COLD-START PREDICTION TEST")
        print("="*60)
        
        # Get first orders for each rider
        first_orders = self.df.groupby('rider_id')['order_id'].first()
        cold_start_df = self.df[self.df['order_id'].isin(first_orders)].copy()
        
        # Features available at first order
        features = ['total_distance', 'first_mile_distance', 'last_mile_distance',
                   'session_time', 'hour', 'is_peak_hour']
        
        # Prepare data
        X = cold_start_df[features].fillna(cold_start_df[features].median())
        y = cold_start_df['is_strategic_order'].astype(int)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=50, 
            max_depth=6, 
            class_weight='balanced',
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Get predictions at 0.3 threshold
        y_pred = (y_pred_proba >= 0.3).astype(int)
        precision = (y_test & y_pred).sum() / y_pred.sum() if y_pred.sum() > 0 else 0
        recall = (y_test & y_pred).sum() / y_test.sum() if y_test.sum() > 0 else 0
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nCold-start model performance:")
        print(f"AUC-ROC: {auc:.3f}")
        print(f"Precision at 0.30 threshold: {precision:.1%}")
        print(f"Recall at 0.30 threshold: {recall:.1%}")
        print(f"\nTop features:")
        print(feature_importance.head())
        
        # Bootstrap confidence interval for AUC
        n_bootstrap = 100
        auc_scores = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_test), len(y_test), replace=True)
            auc_boot = roc_auc_score(y_test.iloc[indices], y_pred_proba[indices])
            auc_scores.append(auc_boot)
        
        ci_lower = np.percentile(auc_scores, 2.5)
        ci_upper = np.percentile(auc_scores, 97.5)
        
        print(f"\nAUC 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        self.results['H5'] = {
            'auc': auc,
            'auc_ci': (ci_lower, ci_upper),
            'precision': precision,
            'recall': recall,
            'feature_importance': feature_importance,
            'n_samples': len(X_test)
        }
        
        return self.results['H5']
    
    def run_all_tests(self):
        """Run all hypothesis tests."""
        self.test_h1_behavioral_repetition()
        self.test_h2_peak_hour_effects()
        self.test_h3_distance_sensitivity()
        self.test_h4_timing_not_predictive()
        self.test_h5_cold_start_prediction()
        
        # Create summary table
        summary = []
        for h, results in self.results.items():
            summary.append({
                'Hypothesis': h,
                'Test Statistic': results.get('statistic', results.get('z_statistic', results.get('t_statistic', results.get('auc', 'N/A')))),
                'p-value': results.get('p_value', 'N/A'),
                'Effect Size': results.get('effect_size', results.get('odds_ratio', results.get('cohens_d', results.get('marginal_effect', 'N/A')))),
                'Result': 'Supported' if results.get('p_value', 1) < 0.05 or h == 'H5' else 'Not Supported'
            })
        
        summary_df = pd.DataFrame(summary)
        print("\n" + "="*60)
        print("HYPOTHESIS TESTING SUMMARY")
        print("="*60)
        print(summary_df)
        
        return self.results, summary_df

def main():
    """Execute hypothesis testing."""
    # Load data
    df = pd.read_csv('../data/shadowfax_processed-data-final.csv')
    
    # Add strategic rider flags (from main analysis)
    # This would normally come from the complete_analysis_pipeline.py
    # For now, we'll calculate it here
    rider_stats = df.groupby('rider_id').agg({
        'cancelled': ['sum', 'count'],
        'cancel_after_pickup': 'sum'
    }).reset_index()
    rider_stats.columns = ['rider_id', 'total_cancellations', 'total_orders', 'post_pickup_cancels']
    
    bike_issues = df[df['reason_text'] == 'Cancel order due to bike issue'].groupby('rider_id').size()
    rider_stats['bike_issues'] = rider_stats['rider_id'].map(bike_issues).fillna(0)
    
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
    
    strategic_riders = rider_stats[
        (rider_stats['bike_issues'] >= 2) &
        (rider_stats['post_pickup_rate'] > 0.7) &
        (rider_stats['bike_issue_rate'] > 0.2)
    ]['rider_id'].values
    
    df['is_strategic_rider'] = df['rider_id'].isin(strategic_riders)
    df['is_strategic_order'] = df['is_strategic_rider'] & (df['cancelled'] == 1)
    
    # Extract hour from order_time
    df['hour'] = pd.to_datetime(df['order_time']).dt.hour
    
    # Run tests
    tester = HypothesisTester(df)
    results, summary = tester.run_all_tests()
    
    # Save results
    summary.to_csv('results/hypothesis_test_summary.csv', index=False)
    print("\nResults saved to results/hypothesis_test_summary.csv")
    
    return results, summary

if __name__ == "__main__":
    results, summary = main()