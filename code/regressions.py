import pandas as pd 
import numpy as np
import os
import itertools
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings("ignore")

# =============================================================================================================
# PARAMETERS FOR GRID SEARCH
# =============================================================================================================
data_days_list = [7, 14, 21, 28]             
indicator_days_list = [126, 180, 252]             
vol_coeff_list = [0.5, 1.0, 1.5, 2.0]             
targets = ['R_util_debt', 'excess_R_util']   

D_util = 7.5 
D_govt = 8.5
folder = r'/Users/igorbykov/Desktop/Thesis/data'

# =============================================================================================================
# UTILITY FUNCTIONS
# =============================================================================================================
def clean_process(df):
    df = df.where(df != 0, np.nan)
    df = df.dropna()
    df = df.resample('D').asfreq()
    df = df.interpolate('linear')
    return df

def calculate_vif(X):
    """Calculates VIF for each feature in the design matrix."""
    vif_dict = {}
    for i, col in enumerate(X.columns):
        if col != 'const': # Exclude constant from VIF reporting for clarity
            try:
                vif_dict[f'VIF_{col}'] = variance_inflation_factor(X.values, i)
            except Exception:
                vif_dict[f'VIF_{col}'] = np.nan
    return vif_dict

def extract_regression_metrics(model, X, model_name, target_name, params_dict, primary_feature):
    """Extracts R2, specific commodity p-values, and VIF scores."""
    res = params_dict.copy()
    res['Model'] = model_name
    res['Target'] = target_name
    res['Adj_R2'] = model.rsquared_adj
    
    # Store primary p-value for sorting
    if primary_feature in model.pvalues:
        res['Sort_P_Value'] = model.pvalues[primary_feature]
    else:
        res['Sort_P_Value'] = np.nan

    # Extract all relevant feature p-values and coefficients
    features_to_track = ['R_brent', 'R_ttf', 'brent_indic_brent', 'ttf_indic_ttf', 'D_brent', 'D_ttf', 'govt_indic_brent', 'govt_indic_ttf']
    for feat in features_to_track:
        if feat in model.pvalues:
            res[f'p_val_{feat}'] = model.pvalues[feat]
            res[f'coeff_{feat}'] = model.params[feat]
            
    # Calculate and append VIF scores
    vif_scores = calculate_vif(X)
    res.update(vif_scores)
            
    return res

# =============================================================================================================
# DATA LOAD (Execute Once)
# =============================================================================================================
brent = clean_process(pd.read_excel(os.path.join(folder, 'commodities_data_v0.xlsx'), usecols=[0,1], index_col=0))
ttf = clean_process(pd.read_excel(os.path.join(folder, 'ttf_front_month.xlsx'), usecols=[0,1], index_col=0))
utilities_equity = clean_process(pd.read_excel(os.path.join(folder, 'utilities_data_v0.xlsx'), usecols=[0,2], index_col=0))
util_yield_10y = clean_process(pd.read_excel(os.path.join(folder, 'utilities_debt.xlsx'), usecols=[0,3], index_col=0)) / 100
govt_yield_10y = clean_process(pd.read_excel(os.path.join(folder, 'euro_curve.xlsx'), usecols=[0,2], index_col=0)) / 100

df_daily = pd.concat([brent, ttf, utilities_equity, util_yield_10y, govt_yield_10y], axis=1, join='outer')
df_daily.columns = ['CO1_Comdty', 'TTF_front_month', 'SX6P_Index', 'Util_Yield', 'Govt_Yield']
df_daily = df_daily.dropna()

# =============================================================================================================
# ITERATION LOOP
# =============================================================================================================
results = []
parameter_combinations = list(itertools.product(data_days_list, indicator_days_list, vol_coeff_list, targets))

# =============================================================================================================
# ITERATION LOOP
# =============================================================================================================
results = []
parameter_combinations = list(itertools.product(data_days_list, indicator_days_list, vol_coeff_list, targets))

for data_days, indicator_days, vol_coeff, target in parameter_combinations:
    frequency = f'{data_days}D'
    indicator_window = round(indicator_days / data_days)
    
    df = df_daily.resample(frequency).last().dropna()
    
    df['R_util_debt'] = np.log((1 - D_util * df['Util_Yield'].diff() + df['Util_Yield'] * data_days / 365).dropna())
    df['R_govt'] = np.log((1 - D_govt * df['Govt_Yield'].diff() + df['Govt_Yield'] * data_days / 365).dropna())
    df['R_util_eqty'] = np.log((df['SX6P_Index'] / df['SX6P_Index'].shift(1)).dropna())
    df['R_brent'] = np.log(df['CO1_Comdty'] / df['CO1_Comdty'].shift(1)).dropna()
    df['R_ttf'] = np.log(df['TTF_front_month'] / df['TTF_front_month'].shift(1)).dropna()
    
    df['excess_R_util'] = df['R_util_debt'] - df['R_govt']
    df = df.dropna()
    
    # Corrected: Rolling calculations apply to log-returns
    mu_p_6m_brent = df['R_brent'].rolling(window=indicator_window).mean()
    std_p_6m_brent = df['R_brent'].rolling(window=indicator_window).std()
    threshold_brent = mu_p_6m_brent + vol_coeff * std_p_6m_brent
    
    mu_p_6m_ttf = df['R_ttf'].rolling(window=indicator_window).mean()
    std_p_6m_ttf = df['R_ttf'].rolling(window=indicator_window).std()
    threshold_ttf = mu_p_6m_ttf + vol_coeff * std_p_6m_ttf
    
    # Corrected: Binary indicators test log-returns against the threshold
    df['D_brent'] = np.where(df['R_brent'] > threshold_brent, 1, 0)
    df['D_ttf'] = np.where(df['R_ttf'] > threshold_ttf, 1, 0)
    
    df = df.dropna()
    
    df['govt_indic_brent'] = df['D_brent'] * df['R_govt']
    df['eqty_indic_brent'] = df['D_brent'] * df['R_util_eqty']
    df['brent_indic_brent'] = df['D_brent'] * df['R_brent']
    
    df['govt_indic_ttf'] = df['D_ttf'] * df['R_govt']
    df['eqty_indic_ttf'] = df['D_ttf'] * df['R_util_eqty']
    df['ttf_indic_ttf'] = df['D_ttf'] * df['R_ttf']
    
    params_dict = {'data_days': data_days, 'indicator_days': indicator_days, 'vol_coeff': vol_coeff}

    max_lags = int(np.floor(4*(len(df)/100)**(2/9)))
    
    try:
        # Unconditional Brent
        X_unc_brent = sm.add_constant(df[['R_govt', 'R_brent', 'R_util_eqty']])
        unc_brent = sm.OLS(df[target], X_unc_brent).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
        results.append(extract_regression_metrics(unc_brent, X_unc_brent, 'Unconditional Brent', target, params_dict, 'R_brent'))
        
        # Conditional Brent
        X_con_brent = sm.add_constant(df[['R_govt', 'govt_indic_brent', 'R_brent', 'brent_indic_brent', 'D_brent', 'R_util_eqty', 'eqty_indic_brent']])
        con_brent = sm.OLS(df[target], X_con_brent).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
        results.append(extract_regression_metrics(con_brent, X_con_brent, 'Conditional Brent', target, params_dict, 'brent_indic_brent'))
        
        # Unconditional TTF
        X_unc_ttf = sm.add_constant(df[['R_govt', 'R_ttf', 'R_util_eqty']])
        unc_ttf = sm.OLS(df[target], X_unc_ttf).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
        results.append(extract_regression_metrics(unc_ttf, X_unc_ttf, 'Unconditional TTF', target, params_dict, 'R_ttf'))
        
        # Conditional TTF
        X_con_ttf = sm.add_constant(df[['R_govt', 'govt_indic_ttf', 'R_ttf', 'ttf_indic_ttf', 'D_ttf', 'R_util_eqty', 'eqty_indic_ttf']])
        con_ttf = sm.OLS(df[target], X_con_ttf).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
        results.append(extract_regression_metrics(con_ttf, X_con_ttf, 'Conditional TTF', target, params_dict, 'ttf_indic_ttf'))
        
    except ValueError:
        continue

# =============================================================================================================
# EXPORT RESULTS
# =============================================================================================================
results_df = pd.DataFrame(results)

# Sort strictly by the p-value of the primary commodity feature (ascending) to optimize for significance
results_df = results_df.sort_values(by='Sort_P_Value', ascending=True)

output_path = os.path.join(folder, 'regression_optimization_results.xlsx')
results_df.to_excel(output_path, index=False)