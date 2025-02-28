import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# Create a sample DataFrame with mixed date types
def create_sample_data():
    # Create dates with different types
    dates = [
        datetime.now(),                      # datetime
        date.today(),                        # date
        datetime.now() - timedelta(days=1),  # datetime
        date.today() - timedelta(days=1),    # date
        np.datetime64('today'),              # numpy datetime64
        pd.Timestamp('today')                # pandas Timestamp
    ]
    
    # Create a DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': range(len(dates))
    })
    
    print("Original DataFrame:")
    print(df)
    print("\nDate column types:")
    for i, d in enumerate(df['date']):
        print(f"Row {i}: {type(d)}")
    
    return df

def demonstrate_issue(df):
    # Try to filter with a datetime
    filter_date = datetime.now()
    
    try:
        # This will fail with mixed types
        filtered = df[df['date'] < filter_date]
        print("\nFiltered successfully!")
    except TypeError as e:
        print(f"\nError when filtering: {e}")

def fix_issue(df):
    # Convert all dates to datetime objects
    print("\nConverting all dates to datetime objects...")
    df['date'] = pd.to_datetime(df['date'])
    
    print("\nAfter conversion:")
    print(df)
    print("\nDate column types after conversion:")
    for i, d in enumerate(df['date']):
        print(f"Row {i}: {type(d)}")
    
    # Try filtering again
    filter_date = datetime.now()
    filtered = df[df['date'] < filter_date]
    print(f"\nFiltered successfully! Got {len(filtered)} rows.")
    
    return df

if __name__ == "__main__":
    print("Demonstrating datetime type mismatch issue and solution")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    
    # Demonstrate the issue
    demonstrate_issue(df)
    
    # Fix the issue
    fixed_df = fix_issue(df)
    
    print("\nConclusion:")
    print("To fix datetime type mismatch issues, convert all date columns to datetime using pd.to_datetime()")
    print("This ensures consistent date comparisons throughout the code.") 