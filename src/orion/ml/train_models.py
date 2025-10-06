"""
Training script for Orion ML models.
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orion.ml.route_optimization import RouteOptimizationModel
from orion.ml.cost_prediction import CostPredictionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_all_models():
    """Train all ML models for Orion."""
    logger.info("🚀 Starting Orion ML model training...")

    # Train Route Optimization Model
    logger.info("🔧 Training route optimization model...")
    route_optimizer = RouteOptimizationModel()
    
    # Create synthetic data for route optimization
    n_samples = 10000
    np.random.seed(42)
    
    data_route = {
        "amount": np.random.uniform(10, 100000, n_samples),
        "vendor_risk_score": np.random.uniform(0.1, 0.9, n_samples),
        "vendor_payment_history_score": np.random.uniform(0.3, 1.0, n_samples),
        "market_volatility": np.random.uniform(0.1, 0.8, n_samples),
        "network_congestion": np.random.uniform(0.1, 0.7, n_samples),
        "regulatory_compliance_required": np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        "historical_ach_success_rate": np.random.uniform(0.95, 0.99, n_samples),
        "historical_wire_success_rate": np.random.uniform(0.97, 0.995, n_samples),
        "historical_rtp_success_rate": np.random.uniform(0.94, 0.98, n_samples),
        "historical_v_card_success_rate": np.random.uniform(0.92, 0.96, n_samples),
        "current_ach_cost": np.random.uniform(0.2, 0.4, n_samples),
        "current_wire_cost": np.random.uniform(12, 18, n_samples),
        "current_rtp_cost": np.random.uniform(0.4, 0.8, n_samples),
        "current_v_card_cost": np.random.uniform(1.5, 3.0, n_samples),
        "is_business_hours": np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
        "is_weekend": np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        "is_holiday": np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        "is_month_end": np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        "is_quarter_end": np.random.choice([True, False], n_samples, p=[0.05, 0.95]),
        "invoice_due_date_hours": np.random.uniform(1, 720, n_samples),
        "fraud_likelihood": np.random.uniform(0.01, 0.3, n_samples),
        "compliance_risk": np.random.uniform(0.05, 0.4, n_samples),
        "currency_volatility": np.random.uniform(0.1, 0.6, n_samples),
        "company_cash_flow_position": np.random.uniform(-0.5, 1.0, n_samples),
        "treasury_optimization_enabled": np.random.choice([True, False], n_samples, p=[0.8, 0.2]),
        # One-hot encoded features
        "urgency_normal": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        "urgency_high": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "vendor_category_supplier": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "vendor_category_contractor": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "vendor_preferred_rail_ach": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        "vendor_preferred_rail_wire": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "currency_USD": np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        "currency_EUR": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        "transaction_type_invoice": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "transaction_type_payroll": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "payment_frequency_one_time": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        "payment_frequency_recurring": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        "cost_sensitivity_medium": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        "cost_sensitivity_high": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "speed_sensitivity_medium": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        "speed_sensitivity_high": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    }
    
    df_route = pd.DataFrame(data_route)
    
    # Generate synthetic rail targets based on amount, urgency, and cost
    # Small amounts -> ACH, Medium amounts -> RTP, Large amounts -> Wire, Quick payments -> V-Card
    rail_targets = []
    for _, row in df_route.iterrows():
        amount = row['amount']
        urgency_high = row['urgency_high']
        cost_sensitivity_high = row['cost_sensitivity_high']
        speed_sensitivity_high = row['speed_sensitivity_high']
        
        if urgency_high and speed_sensitivity_high:
            rail = 'rtp'  # Fast payments
        elif amount > 50000:
            rail = 'wire'  # Large amounts
        elif amount < 1000 and not cost_sensitivity_high:
            rail = 'v_card'  # Small amounts, not cost-sensitive
        else:
            rail = 'ach'  # Default for medium amounts
        
        rail_targets.append(rail)
    
    df_route['rail_target'] = rail_targets
    route_optimizer.train_model(df_route[route_optimizer.feature_names], df_route['rail_target'])
    route_optimizer.save_model()
    logger.info("✅ Route optimization model trained and saved")

    # Train Cost Prediction Model
    logger.info("🔧 Training cost prediction model...")
    cost_predictor = CostPredictionModel()
    
    # Create synthetic data for cost prediction
    data_cost = {
        "amount": np.random.uniform(10, 100000, n_samples),
        "vendor_risk_score": np.random.uniform(0.1, 0.9, n_samples),
        "market_volatility": np.random.uniform(0.1, 0.8, n_samples),
        "network_congestion": np.random.uniform(0.1, 0.7, n_samples),
        "base_rail_cost": np.random.uniform(0.2, 20.0, n_samples),
        "is_business_hours": np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
        "is_weekend": np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        "is_holiday": np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        "is_month_end": np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        "is_quarter_end": np.random.choice([True, False], n_samples, p=[0.05, 0.95]),
        "hour_of_day": np.random.randint(0, 24, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "invoice_due_date_hours": np.random.uniform(1, 720, n_samples),
        "fraud_likelihood": np.random.uniform(0.01, 0.3, n_samples),
        "compliance_risk": np.random.uniform(0.05, 0.4, n_samples),
        "currency_volatility": np.random.uniform(0.1, 0.6, n_samples),
        "vendor_historical_avg_cost": np.random.uniform(0.5, 25.0, n_samples),
        "rail_historical_avg_cost": np.random.uniform(0.3, 20.0, n_samples),
        "vendor_historical_success_rate": np.random.uniform(0.9, 0.99, n_samples),
        "rail_historical_success_rate": np.random.uniform(0.92, 0.99, n_samples),
        "company_cash_flow_position": np.random.uniform(-0.5, 1.0, n_samples),
        "treasury_optimization_enabled": np.random.choice([True, False], n_samples, p=[0.8, 0.2]),
        "monthly_transaction_volume": np.random.uniform(1000, 1000000, n_samples),
        "vendor_transaction_frequency": np.random.uniform(1, 30, n_samples),
        "regulatory_compliance_required": np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        "international_transfer": np.random.choice([True, False], n_samples, p=[0.15, 0.85]),
        "sanctions_screening_required": np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        # One-hot encoded features
        "rail_type_ach": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        "rail_type_wire": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "rail_type_rtp": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "rail_type_v_card": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        "currency_USD": np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        "currency_EUR": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        "vendor_category_supplier": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "vendor_category_contractor": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "vendor_volume_tier_medium": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        "vendor_volume_tier_high": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "transaction_type_invoice": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "transaction_type_payroll": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "payment_frequency_one_time": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        "payment_frequency_recurring": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        "cost_sensitivity_medium": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        "cost_sensitivity_high": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "speed_sensitivity_medium": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        "speed_sensitivity_high": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    }
    
    df_cost = pd.DataFrame(data_cost)
    
    # Generate synthetic cost targets
    # Base cost + risk adjustments + timing adjustments + volume adjustments
    df_cost['cost_target'] = (
        df_cost['base_rail_cost'] * 1.0 +  # Base cost
        df_cost['vendor_risk_score'] * 2.0 +  # Risk adjustment
        df_cost['market_volatility'] * 1.5 +  # Market volatility
        df_cost['network_congestion'] * 0.5 +  # Network congestion
        (df_cost['is_weekend'] * 0.5) +  # Weekend premium
        (df_cost['is_holiday'] * 1.0) +  # Holiday premium
        (df_cost['regulatory_compliance_required'] * 2.0) +  # Compliance cost
        (df_cost['international_transfer'] * 3.0) +  # International fee
        (df_cost['sanctions_screening_required'] * 1.5) +  # Screening cost
        np.random.normal(0, 0.5, n_samples)  # Random noise
    )
    df_cost['cost_target'] = np.clip(df_cost['cost_target'], 0.1, 50.0)
    
    cost_predictor.train_model(df_cost[cost_predictor.feature_names], df_cost['cost_target'])
    cost_predictor.save_model()
    logger.info("✅ Cost prediction model trained and saved")

    logger.info("🎉 All ML models trained successfully!")


if __name__ == "__main__":
    train_all_models()
