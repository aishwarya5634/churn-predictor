import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler 

def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns = ['RowNumber', 'CustomerId', 'Surname'])
    le = LabelEncoder()
    df['Geography'] = le.fit_transform(df['Geography'])
    df['Gender'] = le.fit_transform(df['Gender'])
    
    return df


def engineer_features(df):
    
    df['balance_salary_ratio'] = df['Balance']/(df['EstimatedSalary'] + 1)
    
    df ['products_per_tenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)
    
    df['is_zero_balance'] = (df['Balance'] == 0).astype(int)
    
    df['age_x_inactive'] = df['Age']*(1-df['IsActiveMember'])
    
    df['credit_per_age'] = df['CreditScore']/ df['Age']
    
    df['high_bal_inactive'] = (
        (df['Balance'] > df['Balance'].median()) &
        (df['IsActiveMember'] == 0)
    ).astype(int)
    
    return df

'''Why is this combination so powerful?
This is the most interesting feature in the whole engineering 
step because it captures a very specific type of customer that 
is extremely likely to churn — someone who has a lot of money 
in the bank but is not actively using their account.
Think about what that means in real life:

They have significant savings sitting in the bank ✓
But they are not logging in, not using their card, not engaging ✓
That combination strongly suggests they have mentally checked 
out and are probably comparison shopping at other banks
right now

Neither condition alone is as powerful:

High balance alone → could just be a very loyal wealthy customer
Inactive alone → could be a low value customer with nothing 
to lose

But high balance AND inactive together → that is a very 
specific red flag that a valuable customer is about to walk 
out the door. By creating this feature you are encoding that
business insight directly into your data so the model can act on it.'''

if __name__ == '__main__':
    df = load_and_clean('data/bank_churn.csv')
    print('Cleaned shape: ', df.shape)
    df.to_csv('data/bank_churn_cleaned.csv', index = False)
    print('Saved to data/bank_churn_cleaned.csv')
    
    df = engineer_features(df)
    print('Shape after feature engineering: ', df.shape)
    df.to_csv('data/bank_churn_features.csv', index= False)
    print('Saved to data/bank_churn_features.csv')
    