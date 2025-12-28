import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_mmm_data(weeks=156):
    """
    Generates synthetic marketing data with Adstock and Saturation effects.
    weeks=156 represents 3 years of weekly data.
    """
    np.random.seed(42)
    
    # 1. Create a Timeline
    date_range = pd.date_range(start='2021-01-01', periods=weeks, freq='W')
    
    # 2. Simulate Media Spend (The "Inputs")
    # TV: High cost, high reach, seasonal spikes (Gamma distribution)
    tv_spend = np.random.gamma(shape=2, scale=1000, size=weeks)
    tv_spend[weeks//3:weeks//3+10] += 5000 # Simulating a "Big Brand Campaign"
    
    # Search: Lower cost, more consistent (Normal distribution)
    search_spend = np.random.normal(loc=2000, scale=500, size=weeks)
    search_spend = np.clip(search_spend, 500, None) # Ensure spend isn't negative
    
    # 3. Define Marketing Physics Functions
    
    # Adstock: TV ads have 'memory'. Alpha=0.6 means 60% of effect remains next week.
    def apply_adstock(spend, alpha):
        adstocked_spend = np.zeros_like(spend)
        for t in range(1, len(spend)):
            adstocked_spend[t] = spend[t] + alpha * adstocked_spend[t-1]
        return adstocked_spend

    # Saturation (Hill Function): Diminishing returns. 
    # The more you spend, the less 'new' sales you get per dollar.
    def apply_saturation(spend, K):
        return spend / (spend + K)

    # 4. Calculate Sales (The "Target")
    # Baseline sales: What we sell if we spend $0 (Brand Equity)
    baseline_sales = 10000 
    
    # TV Effect: Weight=0.5, Alpha=0.6, Saturation Point (K)=3000
    tv_effect = 0.5 * apply_saturation(apply_adstock(tv_spend, 0.6), 3000)
    
    # Search Effect: Weight=0.3, Alpha=0.1, Saturation Point (K)=500
    search_effect = 0.3 * apply_saturation(apply_adstock(search_spend, 0.1), 500)
    
    # Add seasonality (Sine wave to represent yearly peaks like Black Friday/Holidays)
    seasonality = 1 + 0.2 * np.sin(2 * np.pi * np.arange(weeks) / 52)
    
    # Combine everything + Random Noise (The "Chaos" of the real world)
    # We multiply the effects by a scale (e.g., 50,000) to turn them into sales units
    sales = (baseline_sales + (tv_effect * 50000) + (search_effect * 20000)) * seasonality
    sales += np.random.normal(0, 500, size=weeks)
    
    df = pd.DataFrame({
        'week': date_range,
        'tv_spend': tv_spend,
        'search_spend': search_spend,
        'sales': sales
    })
    
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # Generate the data
    data = generate_mmm_data()
    
    # Save to CSV
    data.to_csv('data/raw_marketing_data.csv', index=False)
    print("✅ Data generated successfully in data/raw_marketing_data.csv")
    
    # Plotting the results to verify
    plt.figure(figsize=(14, 7))
    plt.plot(data['week'], data['sales'], label='Total Sales', color='black', linewidth=2)
    plt.fill_between(data['week'], data['tv_spend'], alpha=0.3, label='TV Spend', color='blue')
    plt.fill_between(data['week'], data['search_spend'], alpha=0.3, label='Search Spend', color='orange')
    plt.title("Simulated Market Environment: Sales vs Media Spend")
    plt.xlabel("Date")
    plt.ylabel("Units / Dollars")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()