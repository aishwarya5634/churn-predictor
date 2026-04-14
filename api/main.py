from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import numpy as np
import pandas as pd
import sys
import os

# ─────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.retention_engine import get_strategy, calculate_clv, INTERVENTION_COSTS, RETENTION_LIFT

# ─────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────
app = FastAPI(
    title       = 'Bank Churn Prediction API',
    description = '''
    ## Bank Customer Churn Prediction
    
    Predict which customers are likely to churn and get 
    personalized retention strategy recommendations.
    
    ### Features:
    - Churn probability prediction
    - Risk level classification  
    - Customer Lifetime Value calculation
    - Personalized retention strategy
    - Expected ROI of intervention
    ''',
    version     = '1.0.0',
)

# ─────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────
try:
    model  = joblib.load(os.path.join(BASE_DIR, 'src', 'best_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'src', 'scaler.pkl'))
    print('✅ Model and scaler loaded successfully!')
except Exception as e:
    print(f'❌ Error loading models: {e}')
    raise

# ─────────────────────────────────────
# FEATURE COLUMNS
# must match exact order used in training
# ─────────────────────────────────────
FEATURE_COLS = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    'EstimatedSalary', 'balance_salary_ratio', 'products_per_tenure',
    'is_zero_balance', 'age_x_inactive', 'credit_per_age',
    'high_bal_inactive', 'segment'
]

# ─────────────────────────────────────
# INPUT SCHEMA
# ─────────────────────────────────────
class CustomerData(BaseModel):
    credit_score:      int
    age:               int
    tenure:            int
    balance:           float
    num_products:      int
    has_cr_card:       int
    is_active_member:  int
    estimated_salary:  float
    geography:         str
    gender:            str

    # IMPROVEMENT 1 — input validation
    @field_validator('age')
    def age_must_be_valid(cls, v):
        if v < 18 or v > 100:
            raise ValueError('Age must be between 18 and 100')
        return v

    @field_validator('credit_score')
    def credit_score_must_be_valid(cls, v):
        if v < 300 or v > 850:
            raise ValueError('Credit score must be between 300 and 850')
        return v

    @field_validator('num_products')
    def products_must_be_valid(cls, v):
        if v < 1 or v > 4:
            raise ValueError('Number of products must be between 1 and 4')
        return v

    @field_validator('has_cr_card', 'is_active_member')
    def must_be_binary(cls, v):
        if v not in [0, 1]:
            raise ValueError('Must be 0 or 1')
        return v

    @field_validator('geography')
    def geography_must_be_valid(cls, v):
        if v not in ['France', 'Germany', 'Spain']:
            raise ValueError('Geography must be France, Germany, or Spain')
        return v

    @field_validator('gender')
    def gender_must_be_valid(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be Male or Female')
        return v

    @field_validator('balance', 'estimated_salary')
    def must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Must be a positive number')
        return v

# ─────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_probability:      float
    risk_level:             str
    customer_lifetime_value: float
    recommended_strategy:   str
    intervention_cost:      float
    retention_lift:         float
    expected_roi:           float
    explanation:            dict

# ─────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['balance_salary_ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['products_per_tenure']  = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['is_zero_balance']      = (df['Balance'] == 0).astype(int)
    df['age_x_inactive']       = df['Age'] * (1 - df['IsActiveMember'])
    df['credit_per_age']       = df['CreditScore'] / df['Age']
    df['high_bal_inactive']    = (
        (df['Balance'] > 0) & (df['IsActiveMember'] == 0)
    ).astype(int)
    # IMPROVEMENT 2 — add segment column
    # default to 0 since we don't have clustering here
    df['segment'] = 0
    return df

# ─────────────────────────────────────
# ROUTES
# ─────────────────────────────────────
@app.get('/')
def root():
    return {
        'message': '🏦 Bank Churn Prediction API is running!',
        'version': '1.0.0',
        'endpoints': {
            'predict':  'POST /predict',
            'health':   'GET /health',
            'docs':     'GET /docs'
        }
    }

@app.get('/health')
def health_check():
    # IMPROVEMENT 3 — health check endpoint
    return {
        'status':       'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
    }

@app.post('/predict', response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """
    Predict churn probability and return retention recommendation.
    
    - **credit_score**: Customer credit score (300-850)
    - **age**: Customer age (18-100)
    - **tenure**: Years as customer
    - **balance**: Account balance
    - **num_products**: Number of bank products (1-4)
    - **has_cr_card**: Has credit card (0 or 1)
    - **is_active_member**: Is active (0 or 1)
    - **estimated_salary**: Annual salary estimate
    - **geography**: France, Germany, or Spain
    - **gender**: Male or Female
    """
    try:
        # ── Step 1: Convert to DataFrame ──
        data = customer.model_dump()
        df   = pd.DataFrame([data])

        # ── Step 2: Rename columns to match training ──
        df = df.rename(columns={
            'credit_score':     'CreditScore',
            'age':              'Age',
            'tenure':           'Tenure',
            'balance':          'Balance',
            'num_products':     'NumOfProducts',
            'has_cr_card':      'HasCrCard',
            'is_active_member': 'IsActiveMember',
            'estimated_salary': 'EstimatedSalary',
            'geography':        'Geography',
            'gender':           'Gender',
        })

        # ── Step 3: Encode categoricals ──
        geo_map = {'France': 0, 'Germany': 1, 'Spain': 2}
        gen_map = {'Female': 0, 'Male': 1}
        df['Geography'] = df['Geography'].map(geo_map)
        df['Gender']    = df['Gender'].map(gen_map)

        # ── Step 4: Engineer features ──
        df = engineer_features(df)

        # ── Step 5: Validate all columns present ──
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=500,
                detail=f'Missing features: {missing}'
            )

        # ── Step 6: Scale and predict ──
        X_scaled   = scaler.transform(df[FEATURE_COLS])
        churn_prob = float(model.predict_proba(X_scaled)[0][1])

        # ── Step 7: CLV and strategy ──
        clv      = calculate_clv(
            customer.balance,
            customer.tenure,
            customer.num_products,
            customer.estimated_salary
        )
        strategy = get_strategy(
            churn_prob,
            clv,
            customer.is_active_member,
            customer.num_products
        )

        # ── Step 8: ROI calculation ──
        intervention_cost = INTERVENTION_COSTS.get(strategy, 0)
        retention_lift    = RETENTION_LIFT.get(strategy, 0)
        expected_roi      = (
            clv * churn_prob * retention_lift - intervention_cost
        )

        # ── Step 9: Risk level ──
        if churn_prob >= 0.70:
            risk_level = 'Critical'
        elif churn_prob >= 0.60:
            risk_level = 'High'
        elif churn_prob >= 0.30:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        # ── Step 10: Human readable explanation ──
        explanation = {
            'churn_drivers': [],
            'protective_factors': []
        }
        if customer.num_products == 1:
            explanation['churn_drivers'].append(
                'Only 1 product — cross-sell recommended'
            )
        if customer.age > 45:
            explanation['churn_drivers'].append(
                'Age above 45 — higher churn risk group'
            )
        if customer.is_active_member == 0:
            explanation['churn_drivers'].append(
                'Inactive member — engagement needed'
            )
        if customer.balance == 0:
            explanation['churn_drivers'].append(
                'Zero balance — potential churner signal'
            )
        if customer.num_products >= 3:
            explanation['protective_factors'].append(
                'Multiple products — strong retention signal'
            )
        if customer.tenure > 5:
            explanation['protective_factors'].append(
                'Long tenure — loyal customer'
            )
        if customer.is_active_member == 1:
            explanation['protective_factors'].append(
                'Active member — engaged with bank'
            )

        return PredictionResponse(
            churn_probability       = round(churn_prob, 4),
            risk_level              = risk_level,
            customer_lifetime_value = clv,
            recommended_strategy    = strategy,
            intervention_cost       = intervention_cost,
            retention_lift          = retention_lift,
            expected_roi            = round(expected_roi, 2),
            explanation             = explanation
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# IMPROVEMENT 4 — batch prediction endpoint
@app.post('/predict/batch')
def predict_batch(customers: list[CustomerData]):
    """Predict churn for multiple customers at once."""
    if len(customers) > 1000:
        raise HTTPException(
            status_code=400,
            detail='Maximum 1000 customers per batch request'
        )
    results = []
    for customer in customers:
        try:
            result = predict_churn(customer)
            results.append(result)
        except Exception as e:
            results.append({'error': str(e)})
    return {
        'total':   len(customers),
        'results': results
    }