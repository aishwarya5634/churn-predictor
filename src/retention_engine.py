import pandas as pd

import os

# This automatically finds the correct path no matter where you run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'bank_churn_segmented.csv'))

# Define what each strategy costs the bank
INTERVENTION_COSTS = {
    'Priority Call + Exclusive Offer': 75,   # Agent time + 50 discount
    'Dedicated Relationship Manager':  120,  # High-value service
    'Email + Fee Waiver':              25,   # Small discount + email
    'Loyalty Points Bonus':            30,   # Reward cost
    'Standard Newsletter':              2,   # Just email
    'No Action':                        0,
}
# Expected success rate for each strategy
RETENTION_LIFT = {
    'Priority Call + Exclusive Offer': 0.40,
    'Dedicated Relationship Manager':  0.50,
    'Email + Fee Waiver':              0.20,
    'Loyalty Points Bonus':            0.18,
    'Standard Newsletter':             0.05,
    'No Action':                       0.00,
}
def get_strategy(churn_prob, clv, is_active, num_products):
    """
    Returns the recommended retention strategy.
    Args:
        churn_prob: float 0-1, model's predicted churn probability
        clv: float, customer lifetime value in dollars
        is_active: int 0 or 1, is account active?
        num_products: int, how many bank products do they have?
    """
    # Very high risk, very high value: white-glove treatment
    if churn_prob >= 0.70 and clv >= 15000:
        return 'Dedicated Relationship Manager'
    # High risk, decent value: personal call and incentive
    elif churn_prob >= 0.60 and clv >= 5000:
        return 'Priority Call + Exclusive Offer'
    # Moderate risk, inactive, single product: financial incentive
    elif churn_prob >= 0.45 and is_active == 0:
        return 'Email + Fee Waiver'
    # Moderate risk, engaged: reward them
    elif churn_prob >= 0.40 and num_products >= 2:
        return 'Loyalty Points Bonus'
    # Low risk: just keep them warm
    elif churn_prob >= 0.20:
        return 'Standard Newsletter'
    # Minimal risk: don't waste money
    else:
        return 'No Action'
    

def calculate_clv(balance, tenure, num_products, estimated_salary):
    """
    Simple CLV estimate for banking.
    Real banks use complex models, but this is a reasonable approximation.
    """
    # Average annual revenue per product per dollar of balance
    # Simplified: assume 2% of balance + 500 per product per year
    annual_value = (balance * 0.02) + (num_products * 500)
    # Salary indicates future potential even if current balance is low
    salary_factor = 1 + (estimated_salary / 100000) * 0.1
    # Assume average remaining tenure is proportional to current tenure
    # (sticky customers tend to stay)
    avg_remaining_years = max(3, tenure * 0.5)
    clv = annual_value * salary_factor * avg_remaining_years
    return round(clv, 2)
# Apply to all customers
df['clv'] = df.apply(lambda row: calculate_clv(
    row['Balance'], row['Tenure'],
    row['NumOfProducts'], row['EstimatedSalary']
), axis=1)
print('CLV Summary:')
print(df['clv'].describe())


def apply_retention_strategy(df, model, scaler, feature_cols):
    """Apply the full retention pipeline to a customer dataframe."""
    # Get churn probabilities from model
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    df['churn_prob'] = model.predict_proba(X_scaled)[:, 1]
    # Calculate CLV
    df['clv'] = df.apply(lambda r: calculate_clv(
        r['Balance'], r['Tenure'],
        r['NumOfProducts'], r['EstimatedSalary']
    ), axis=1)
    # Get recommended strategy per customer
    df['strategy'] = df.apply(lambda r: get_strategy(
        r['churn_prob'], r['clv'],
        r['IsActiveMember'], r['NumOfProducts']
    ), axis=1)
    # Calculate expected value of intervention
    df['intervention_cost'] = df['strategy'].map(INTERVENTION_COSTS)
    df['retention_lift']    = df['strategy'].map(RETENTION_LIFT)
    # Expected value = CLV saved * lift probability - intervention cost
    df['expected_value'] = (
        df['clv'] * df['churn_prob'] * df['retention_lift']
        - df['intervention_cost']
    )
    # Priority tier
    df['priority'] = pd.cut(
        df['expected_value'],
        bins=[-float('inf'), 0, 500, float('inf')],
        labels=['Low', 'Medium', 'High']
    )
    # Sort by expected value (highest ROI first)
    return df.sort_values('expected_value', ascending=False)

import joblib

# Load your saved model and scaler
model  = joblib.load(os.path.join(BASE_DIR, 'src', 'best_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'src', 'scaler.pkl'))

# Define feature columns
feature_cols = [c for c in df.columns
                if c not in ['Exited', 'RowNumber', 
                            'CustomerId', 'Surname',
                            'clv', 'strategy',
                            'intervention_cost',
                            'retention_lift',
                            'expected_value',
                            'priority']]

# Run the full pipeline
result = apply_retention_strategy(df, model, scaler, feature_cols)
print(result[['churn_prob', 'clv', 'strategy','expected_value', 'priority']].head(10))

df.to_csv(os.path.join(BASE_DIR, 'data', 'bank_churn_with_predictions.csv'), index=False)
print('✅ Saved to bank_churn_with_predictions.csv!')