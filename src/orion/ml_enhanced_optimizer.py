"""
ML-enhanced optimizer for Orion.
Integrates ML models with traditional optimization logic for improved routing decisions.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from orion.optimize import score_rails, get_best_rail, validate_context, RAIL_CONFIGS
from orion.ml.route_optimization import get_route_optimizer, RouteOptimizationFeatures, RouteOptimizationResult
from orion.ml.cost_prediction import get_cost_predictor, CostPredictionFeatures, CostPredictionResult

logger = logging.getLogger(__name__)


class MLEnhancedOptimizer:
    """
    ML-enhanced optimizer that combines traditional optimization with ML predictions.
    
    Uses ML models to:
    1. Optimize route selection with ML predictions
    2. Predict actual costs vs. advertised costs
    3. Enhance routing decisions with ML insights
    """

    def __init__(self, ml_weight: float = 0.7, use_ml: bool = True):
        """Initialize ML-enhanced optimizer."""
        self.ml_weight = ml_weight
        self.use_ml = use_ml
        self.route_optimizer = get_route_optimizer()
        self.cost_predictor = get_cost_predictor()
        logger.info(f"MLEnhancedOptimizer initialized with ml_weight={self.ml_weight}, use_ml={self.use_ml}")

    def optimize_payout(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize payout routing with ML enhancement.
        
        Args:
            context: Context containing amount, urgency, vendor_id, etc.
            
        Returns:
            Optimization result with ML-enhanced routing decisions
        """
        if not self.use_ml:
            logger.info("ML is disabled, falling back to traditional optimization.")
            return self._traditional_optimization(context)

        start_time = datetime.now()
        logger.info(
            f"Running ML-enhanced payout optimization for vendor {context.get('vendor_id', 'unknown')}",
            extra={
                "vendor_id": context.get("vendor_id"),
                "amount": context.get("amount"),
                "urgency": context.get("urgency"),
            }
        )

        # 1. Run traditional optimization to get baseline results
        traditional_result = self._traditional_optimization(context)
        
        if not traditional_result.get("best"):
            logger.warning("No suitable rails available from traditional optimization")
            return traditional_result

        # 2. Prepare features for ML optimization
        route_features = self._prepare_route_optimization_features(context)
        
        # 3. Get ML route optimization predictions
        route_result = self.route_optimizer.optimize_route(route_features)
        
        # 4. Predict costs for each rail
        cost_predictions = self._predict_all_rail_costs(context, route_result)
        
        # 5. Apply ML-enhanced routing decision
        ml_enhanced_decision = self._determine_ml_enhanced_rail(
            traditional_result, 
            route_result, 
            cost_predictions
        )
        
        # 6. Generate enhanced explanation with ML insights
        enhanced_explanation = self._generate_ml_enhanced_explanation(
            traditional_result,
            route_result,
            cost_predictions,
            ml_enhanced_decision
        )

        optimization_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(
            f"ML-enhanced optimization completed",
            extra={
                "traditional_best": traditional_result["best"]["rail_id"],
                "ml_recommended": route_result.recommended_rail,
                "final_decision": ml_enhanced_decision["rail_id"],
                "route_confidence": route_result.confidence_score,
                "optimization_time_ms": optimization_time,
            }
        )

        # Build enhanced response
        enhanced_result = {
            "best": ml_enhanced_decision,
            "ranked": traditional_result["ranked"],
            "explanation": enhanced_explanation,
            "context": context,
            "ml_metadata": {
                "route_optimization": route_result.model_dump(),
                "cost_predictions": [pred.model_dump() for pred in cost_predictions],
                "ml_weight": self.ml_weight,
                "optimization_time_ms": optimization_time
            }
        }

        return enhanced_result

    def _traditional_optimization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run traditional optimization logic."""
        # Validate and normalize context
        validated_context = validate_context(context)

        # Score and rank rails
        ranked_rails = score_rails(validated_context)

        if not ranked_rails:
            return {
                "error": "No suitable payment rails available for this amount",
                "context": validated_context,
            }

        # Get best rail
        best_rail = get_best_rail(ranked_rails)

        if not best_rail:
            return {
                "error": "No suitable payment rails available for this amount",
                "context": validated_context,
            }

        return {
            "best": best_rail,
            "ranked": ranked_rails,
            "context": validated_context,
        }

    def _prepare_route_optimization_features(self, context: Dict[str, Any]) -> RouteOptimizationFeatures:
        """Prepare features for route optimization."""
        # Placeholder for actual feature engineering
        # In a real system, these would come from a feature store or external services
        return RouteOptimizationFeatures(
            amount=context.get("amount", 0.0),
            urgency=context.get("urgency", "normal"),
            vendor_id=context.get("vendor_id", "unknown"),
            currency=context.get("currency", "USD"),
            vendor_category="supplier",  # Placeholder
            vendor_risk_score=0.2,  # Placeholder
            vendor_payment_history_score=0.8,  # Placeholder
            vendor_preferred_rail="ach",  # Placeholder
            market_volatility=0.3,  # Placeholder
            network_congestion=0.2,  # Placeholder
            regulatory_compliance_required=False,  # Placeholder
            historical_ach_success_rate=0.98,  # From RAIL_CONFIGS
            historical_wire_success_rate=0.99,  # From RAIL_CONFIGS
            historical_rtp_success_rate=0.97,  # From RAIL_CONFIGS
            historical_v_card_success_rate=0.95,  # From RAIL_CONFIGS
            current_ach_cost=RAIL_CONFIGS["ach"]["cost_per_transaction"],
            current_wire_cost=RAIL_CONFIGS["wire"]["cost_per_transaction"],
            current_rtp_cost=RAIL_CONFIGS["rtp"]["cost_per_transaction"],
            current_v_card_cost=RAIL_CONFIGS["v_card"]["cost_per_transaction"],
            is_business_hours=9 <= datetime.now().hour <= 17,
            is_weekend=datetime.now().weekday() >= 5,
            is_holiday=False,  # Placeholder
            is_month_end=datetime.now().day >= 28,
            is_quarter_end=False,  # Placeholder
            transaction_type="invoice",  # Placeholder
            payment_frequency="one_time",  # Placeholder
            invoice_due_date_hours=72.0,  # Placeholder
            fraud_likelihood=0.05,  # Placeholder
            compliance_risk=0.1,  # Placeholder
            currency_volatility=0.15,  # Placeholder
            company_cash_flow_position=0.2,  # Placeholder
            treasury_optimization_enabled=True,  # Placeholder
            cost_sensitivity="medium",  # Placeholder
            speed_sensitivity="medium"  # Placeholder
        )

    def _predict_all_rail_costs(self, context: Dict[str, Any], route_result: RouteOptimizationResult) -> List[CostPredictionResult]:
        """Predict costs for all available rails."""
        cost_predictions = []
        
        rails = ["ach", "wire", "rtp", "v_card"]
        
        for rail in rails:
            # Skip if rail exceeds amount limits
            if context.get("amount", 0.0) > RAIL_CONFIGS[rail]["max_amount"]:
                continue
                
            features = CostPredictionFeatures(
                amount=context.get("amount", 0.0),
                rail_type=rail,
                currency=context.get("currency", "USD"),
                vendor_id=context.get("vendor_id", "unknown"),
                vendor_category="supplier",  # Placeholder
                vendor_risk_score=0.2,  # Placeholder
                vendor_volume_tier="medium",  # Placeholder
                market_volatility=0.3,  # Placeholder
                network_congestion=0.2,  # Placeholder
                base_rail_cost=RAIL_CONFIGS[rail]["cost_per_transaction"],
                is_business_hours=9 <= datetime.now().hour <= 17,
                is_weekend=datetime.now().weekday() >= 5,
                is_holiday=False,  # Placeholder
                is_month_end=datetime.now().day >= 28,
                is_quarter_end=False,  # Placeholder
                hour_of_day=datetime.now().hour,
                day_of_week=datetime.now().weekday(),
                transaction_type="invoice",  # Placeholder
                payment_frequency="one_time",  # Placeholder
                invoice_due_date_hours=72.0,  # Placeholder
                fraud_likelihood=0.05,  # Placeholder
                compliance_risk=0.1,  # Placeholder
                currency_volatility=0.15,  # Placeholder
                vendor_historical_avg_cost=RAIL_CONFIGS[rail]["cost_per_transaction"],
                rail_historical_avg_cost=RAIL_CONFIGS[rail]["cost_per_transaction"],
                vendor_historical_success_rate=RAIL_CONFIGS[rail]["success_rate"],
                rail_historical_success_rate=RAIL_CONFIGS[rail]["success_rate"],
                company_cash_flow_position=0.2,  # Placeholder
                treasury_optimization_enabled=True,  # Placeholder
                cost_sensitivity="medium",  # Placeholder
                speed_sensitivity="medium",  # Placeholder
                monthly_transaction_volume=10000.0,  # Placeholder
                vendor_transaction_frequency=5.0,  # Placeholder
                regulatory_compliance_required=False,  # Placeholder
                international_transfer=False,  # Placeholder
                sanctions_screening_required=False  # Placeholder
            )
            
            prediction = self.cost_predictor.predict_cost(features)
            cost_predictions.append(prediction)
        
        return cost_predictions

    def _determine_ml_enhanced_rail(
        self, 
        traditional_result: Dict[str, Any], 
        route_result: RouteOptimizationResult,
        cost_predictions: List[CostPredictionResult]
    ) -> Dict[str, Any]:
        """Determine final rail using both traditional and ML results."""
        traditional_best = traditional_result["best"]
        
        # If ML confidence is high and recommendation differs, consider ML recommendation
        if route_result.confidence_score > 0.8:
            # Find the traditional rail that matches ML recommendation
            for rail in traditional_result["ranked"]:
                if rail["rail_id"] == route_result.recommended_rail:
                    # Calculate ML-adjusted score
                    ml_adjustment = route_result.confidence_score * (1 - self.ml_weight)
                    traditional_score = traditional_best["scores"]["total"] * self.ml_weight
                    combined_score = ml_adjustment + traditional_score
                    
                    # If ML-adjusted score is significantly better, use ML recommendation
                    if combined_score > traditional_best["scores"]["total"] * 1.1:
                        logger.info(f"ML recommendation overrode traditional choice: {rail['rail_id']}")
                        # Update the rail with ML insights
                        enhanced_rail = rail.copy()
                        enhanced_rail["ml_confidence"] = route_result.confidence_score
                        enhanced_rail["expected_cost"] = route_result.expected_cost
                        enhanced_rail["expected_processing_time_hours"] = route_result.expected_processing_time_hours
                        enhanced_rail["expected_success_rate"] = route_result.expected_success_rate
                        enhanced_rail["risk_score"] = route_result.risk_score
                        return enhanced_rail
        
        # Default to traditional best, but add ML insights
        enhanced_rail = traditional_best.copy()
        enhanced_rail["ml_confidence"] = route_result.confidence_score
        enhanced_rail["expected_cost"] = route_result.expected_cost
        enhanced_rail["expected_processing_time_hours"] = route_result.expected_processing_time_hours
        enhanced_rail["expected_success_rate"] = route_result.expected_success_rate
        enhanced_rail["risk_score"] = route_result.risk_score
        return enhanced_rail

    def _generate_ml_enhanced_explanation(
        self, 
        traditional_result: Dict[str, Any], 
        route_result: RouteOptimizationResult,
        cost_predictions: List[CostPredictionResult],
        ml_enhanced_decision: Dict[str, Any]
    ) -> str:
        """Generate enhanced explanation with ML insights."""
        base_explanation = f"Selected {ml_enhanced_decision['rail_id'].upper()} rail for ${ml_enhanced_decision['context']['amount']:.2f} payment."
        
        # Add ML insights
        ml_insights = []
        
        if route_result.confidence_score > 0.7:
            ml_insights.append(f"ML predicted {route_result.recommended_rail.upper()} with {route_result.confidence_score:.1%} confidence")
        
        # Add cost prediction insights
        if cost_predictions:
            avg_predicted_cost = sum(pred.predicted_cost for pred in cost_predictions) / len(cost_predictions)
            ml_insights.append(f"Predicted average cost: ${avg_predicted_cost:.2f}")
        
        # Add risk assessment
        if route_result.risk_score > 0.7:
            ml_insights.append("Higher-risk transaction profile detected")
        elif route_result.risk_score < 0.3:
            ml_insights.append("Low-risk transaction profile")
        
        # Add alternative recommendations
        if route_result.alternative_rails:
            best_alternative = route_result.alternative_rails[0]
            ml_insights.append(f"Alternative: {best_alternative['rail'].upper()} ({best_alternative['confidence']:.1%} confidence)")
        
        if ml_insights:
            enhanced_explanation = f"{base_explanation} ML Insights: {'; '.join(ml_insights)}."
        else:
            enhanced_explanation = base_explanation
        
        return enhanced_explanation


def get_ml_enhanced_optimizer(ml_weight: float = 0.7, use_ml: bool = True) -> MLEnhancedOptimizer:
    """Get an ML-enhanced optimizer instance."""
    return MLEnhancedOptimizer(ml_weight=ml_weight, use_ml=use_ml)
