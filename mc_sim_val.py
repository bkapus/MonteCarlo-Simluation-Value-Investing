import pandas as pd
# from yahoo_fin import stock_info as si
import yfinance as yf
# import streamlit as st
import numpy as np
from matplotlib import pyplot as plt

import argparse

parser = argparse.ArgumentParser("Valuation with MC")

def comma_format(number):
    if not pd.isna(number) and number != 0:
        return '{:,.0f}'.format(number)

def percentage_format(number):
    if not pd.isna(number) and number != 0:
        return '{:.1%}'.format(number) 

def calculate_value_distribution(parameter_dict_1, parameter_dict_2, parameter_dict_distribution):
    parameter_list = []
    parameter_list.append(parameter_dict_1['latest revenue'])
    for i in parameter_dict_2:
        if parameter_dict_distribution[i] == 'normal':
            parameter_list.append((np.random.normal(parameter_dict_1[i], parameter_dict_2[i]))/100)
        if parameter_dict_distribution[i] == 'triangular':
            lower_bound = parameter_dict_1[i]
            mode = parameter_dict_2[i]
            parameter_list.append((np.random.triangular(lower_bound, mode, 2*mode-lower_bound))/100)
        if parameter_dict_distribution[i] == 'uniform':
            parameter_list.append((np.random.uniform(parameter_dict_1[i], parameter_dict_2[i]))/100)
    parameter_list.append(parameter_dict_1['net debt'])
    return parameter_list

class Company:

    def __init__(self, ticker):
        self.company = yf.Ticker(ticker)
        self.income_statement = self.company.income_stmt
        self.balance_sheet = self.company.balance_sheet
        self.cash_flow_statement = self.company.cashflow
        self.inputs = self.get_inputs_df()

    def get_inputs_df(self):
        income_statement_list = ['Total Revenue', 'EBIT', 
        'Pretax Income', 'Tax Provision'
        ]
        balance_sheet_list = ['Current Assets', 'Cash And Cash Equivalents',
        'Current Liabilities', 'Current Debt',
        'Long Term Debt'
        ]
        balance_sheet_list_truncated = ['Current Assets', 'Cash And Cash Equivalents',
        'Current Liabilities', 'Long Term Debt'
        ]
        balance_sheet_list_no_debt = ['Current Assets', 'Cash And Cash Equivalents',
        'Current Liabilities'
        ]
        cash_flow_statement_list = ['Depreciation And Amortization', 
        'Capital Expenditure'
        ]
        
        income_statement_df = self.income_statement[self.income_statement.index.isin(income_statement_list)]
        try:
            balance_sheet_df = self.balance_sheet[self.balance_sheet.index.isin(balance_sheet_list)]
        except KeyError:
            try:
                balance_sheet_df = self.balance_sheet[self.balance_sheet.index.isin(balance_sheet_list_truncated)]
            except KeyError:
                balance_sheet_df = self.balance_sheet[self.balance_sheet.index.isin(balance_sheet_list_no_debt)]
        cash_flow_statement_df = self.cash_flow_statement[self.cash_flow_statement.index.isin(cash_flow_statement_list)]

        df = pd.concat([income_statement_df, balance_sheet_df])
        df = pd.concat([df, cash_flow_statement_df])
    
        columns_ts = df.columns
        columns_str = [str(i)[:10] for i in columns_ts]
        columns_dict = {}
        for i,f in zip(columns_ts, columns_str):
            columns_dict[i] = f
        df.rename(columns_dict, axis = 'columns', inplace = True)

        columns_str.reverse()
        df = df[columns_str]
        
        prior_revenue_list = [None]
        for i in range(len(df.loc['Total Revenue'])):
            if i != 0 and i != len(df.loc['Total Revenue']):
                prior_revenue_list.append(df.loc['Total Revenue'][i-1])

        df.loc['priorRevenue'] = prior_revenue_list
        df.loc['revenueGrowth'] = (df.loc['Total Revenue'] - df.loc['priorRevenue']) / df.loc['priorRevenue']
        df.loc['ebitMargin'] = df.loc['EBIT']/df.loc['Total Revenue'] 
        df.loc['taxRate'] = df.loc['Tax Provision']/df.loc['Pretax Income'] 
        df.loc['netCapexOverSales'] = (- df.loc['Capital Expenditure'] - df.loc['Depreciation And Amortization']) / df.loc['Total Revenue']
        try:
            df.loc['nwc'] = (df.loc['Current Assets'] - df.loc['Cash And Cash Equivalents']) - (df.loc['Current Liabilities'] - df.loc['Current Debt'])
        except KeyError:
            df.loc['nwc'] = (df.loc['Current Assets'] - df.loc['Cash And Cash Equivalents']) - (df.loc['Current Liabilities'])
        df.loc['nwcOverSales'] = df.loc['nwc']/df.loc['Total Revenue']
        try:
            df.loc['netDebt'] = df.loc['Current Debt'] + df.loc['Long Term Debt'] - df.loc['Cash And Cash Equivalents']
        except KeyError:
            try:
                df.loc['netDebt'] = df.loc['Long Term Debt'] - df.loc['Cash And Cash Equivalents']
            except KeyError:
                df.loc['netDebt'] = - df.loc['Cash And Cash Equivalents']
        df = df[-7:len(df)].drop('nwc')
        df['Historical average'] = [df.iloc[i].mean() for i in range(len(df))]
        return df

    def get_free_cash_flow_forecast(self, parameter_list):
        df = pd.DataFrame(columns = [1, 2, 3, 4, 5])
        revenue_list = []
        for i in range(5):
            revenue_list.append(parameter_list[0] * (1 + parameter_list[1]) ** (i+1))
        df.loc['Revenues'] = revenue_list
        ebit_list = [i * parameter_list[2] for i in df.loc['Revenues']]
        df.loc['EBIT'] = ebit_list
        tax_list = [i * parameter_list[3] for i in df.loc['EBIT']]
        df.loc['Taxes'] = tax_list
        nopat_list = df.loc['EBIT'] - df.loc['Taxes']
        df.loc['NOPAT'] = nopat_list
        net_capex_list = [i * parameter_list[4] for i in df.loc['Revenues']]
        df.loc['Net capital expenditures'] = net_capex_list
        nwc_list = [i * parameter_list[5] for i in df.loc['Revenues']]
        df.loc['Changes in NWC'] = nwc_list
        free_cash_flow_list = df.loc['NOPAT'] - df.loc['Net capital expenditures'] - df.loc['Changes in NWC']
        df.loc['Free cash flow'] = free_cash_flow_list
        return df

    def discount_free_cash_flows(self, free_cash_flow_df, parameter_list, discount_rate, terminal_growth):
        # free_cash_flow_df = self.get_free_cash_flow_forecast(parameter_list)
        df = free_cash_flow_df.copy()
        discount_factor_list = [(1 + discount_rate) ** i for i in free_cash_flow_df.columns]
        df.loc['Discount factor'] = discount_factor_list
        present_value_list = df.loc['Free cash flow'] / df.loc['Discount factor']
        df.loc['PV free cash flow'] = present_value_list
        df[0] = [0 for i in range(len(df))]
        df.loc['Sum PVs', 0] = df.loc['PV free cash flow', 1:5].sum()
        df.loc['Terminal value', 5] = df.loc['Free cash flow', 5] * (1 + terminal_growth) / (discount_rate - terminal_growth)
        df.loc['PV terminal value', 0] = df.loc['Terminal value', 5] / df.loc['Discount factor', 5]
        df.loc['Company value (enterprise value)', 0] = df.loc['Sum PVs', 0] + df.loc['PV terminal value', 0]
        df.loc['Net debt', 0] = parameter_list[-1]
        df.loc['Equity value', 0] = df.loc['Company value (enterprise value)', 0] - df.loc['Net debt', 0]
        equity_value = df.loc['Equity value', 0] 
        df = df.map(lambda x: comma_format(x))
        df = df.fillna('')
        column_name_list = range(6)
        df = df[column_name_list]
        return df, equity_value


parser.add_argument("ticker_input", help="Ticker label.", type=str)
# parser.add_argument("a1", help="Lower Bound / Mean.", type=int)
# parser.add_argument("a2", help="Upper Bound / Std.", type=int)
parser.add_argument('-l', '--list', help='delimited list input, [revenue growth, ebit margin, tax rate, capex ratio, NWC ratio]', 
                    type=str, nargs='?', default="29,33, 13,15, 21,22, 0.6,0.7, -0.7,-0.6")
parser.add_argument("discount_rate", help="Discount rate", type=float, nargs='?', default=7)
parser.add_argument("terminal_growth", help="Terminal growth rate", type=float, nargs='?', default=3)
parser.add_argument("simulation_iterations", help="Number of Monte Carlo simulation iterations (must be less than 1000)", type=int, nargs='?', default=100)

args = parser.parse_args()

my_list = [float(item) for item in args.list.split(',')]
a1 = my_list[0]
a2 = my_list[1]
b1 = my_list[2]
b2 = my_list[3]
c1 = my_list[4]
c2 = my_list[5]
d1 = my_list[6]
d2 = my_list[7]
e1 = my_list[8]
e2 = my_list[9]

print(a1,a2,b1,b2)

ticker_input = args.ticker_input
discount_rate = args.discount_rate
terminal_growth = args.terminal_growth
simulation_iterations = args.simulation_iterations

discount_rate = float(discount_rate/100)
terminal_growth = terminal_growth/100
simulation_iterations = simulation_iterations

print("Discount Rate: ", discount_rate)
print("Terminal Growth Rate: ", terminal_growth)
print("# of MC simulations: ", simulation_iterations)


def get_company_data():
    company = Company(ticker_input)
    return company

company = get_company_data()
print(company.inputs)
# pd.DataFrame(company.inputs)

parameter_dict_1 = {
    'latest revenue' : 0,
    'revenue growth': 5,
    'ebit margin' : 5,
    'tax rate' : 10,
    'capex ratio' : 5,
    'NWC ratio' : 5,
    'net debt' : 0
}

parameter_dict_2 = {
    'latest revenue' : 0,
    'revenue growth': 20,
    'ebit margin' : 20,
    'tax rate' : 20,
    'capex ratio' : 20,
    'NWC ratio' : 18
}

parameter_dict_distribution = {
    'latest revenue' : '',
    'revenue growth': '',
    'ebit margin' : '',
    'tax rate' : '',
    'capex ratio' : '',
    'NWC ratio' : ''
}

dist_rev = 'Uniform'
if dist_rev == 'Normal':
    mean_input = a1
    stddev_input = a2
    parameter_dict_1['revenue growth'] = mean_input
    parameter_dict_2['revenue growth'] = stddev_input
    parameter_dict_distribution['revenue growth'] = 'normal'
elif dist_rev == 'Triangular':
    lower_input = a1
    mode_input = a2
    parameter_dict_1['revenue growth'] = lower_input
    parameter_dict_2['revenue growth'] = mode_input
    parameter_dict_distribution['revenue growth'] = 'triangular'
elif dist_rev == 'Uniform':
    lower_input = a1
    upper_input = a2
    parameter_dict_1['revenue growth'] = lower_input
    parameter_dict_2['revenue growth'] = upper_input
    parameter_dict_distribution['revenue growth'] = 'uniform'

dist_ebit = 'Uniform'
if dist_ebit == 'Normal':
    mean_input = b1
    stddev_input = b2
    parameter_dict_1['ebit margin'] = mean_input
    parameter_dict_2['ebit margin'] = stddev_input
    parameter_dict_distribution['ebit margin'] = 'normal'
elif dist_ebit == 'Triangular':
    lower_input = b1
    mode_input = b2
    parameter_dict_1['ebit margin'] = lower_input
    parameter_dict_2['ebit margin'] = mode_input
    parameter_dict_distribution['ebit margin'] = 'triangular'
elif dist_ebit == 'Uniform':
    lower_input = b1
    upper_input = b2
    parameter_dict_1['ebit margin'] = lower_input
    parameter_dict_2['ebit margin'] = upper_input
    parameter_dict_distribution['ebit margin'] = 'uniform'

dist_tax = 'Uniform'
if dist_tax == 'Normal':
    mean_input = c1
    stddev_input = c2
    parameter_dict_1[''] = mean_input
    parameter_dict_2['tax rate'] = stddev_input
    parameter_dict_distribution['tax rate'] = 'normal'
elif dist_tax == 'Triangular':
    lower_input = c1
    mode_input = c2
    parameter_dict_1['tax rate'] = lower_input
    parameter_dict_2['tax rate'] = mode_input
    parameter_dict_distribution['tax rate'] = 'triangular'
elif dist_tax == 'Uniform':
    lower_input = c1
    upper_input = c2
    parameter_dict_1['tax rate'] = lower_input
    parameter_dict_2['tax rate'] = upper_input
    parameter_dict_distribution['tax rate'] = 'uniform'

dist_capex = 'Uniform'
if dist_capex == 'Normal':
    mean_input = d1
    stddev_input = d2
    parameter_dict_1['capex ratio'] = mean_input
    parameter_dict_2['capex ratio'] = stddev_input
    parameter_dict_distribution['capex ratio'] = 'normal'
elif dist_capex == 'Triangular':
    lower_input = d1
    mode_input = d2
    parameter_dict_1['capex ratio'] = lower_input
    parameter_dict_2['capex ratio'] = mode_input
    parameter_dict_distribution['capex ratio'] = 'triangular'
elif dist_capex == 'Uniform':
    lower_input = d1
    upper_input = d2
    parameter_dict_1['capex ratio'] = lower_input
    parameter_dict_2['capex ratio'] = upper_input
    parameter_dict_distribution['capex ratio'] = 'uniform'

dist_nwc = 'Uniform'
if dist_nwc == 'Normal':
    mean_input = e1
    stddev_input = e2
    parameter_dict_1['NWC ratio'] = mean_input
    parameter_dict_2['NWC ratio'] = stddev_input
    parameter_dict_distribution['NWC ratio'] = 'normal'
elif dist_nwc == 'Triangular':
    lower_input = e1
    mode_input = e2
    parameter_dict_1['NWC ratio'] = lower_input
    parameter_dict_2['NWC ratio'] = mode_input
    parameter_dict_distribution['NWC ratio'] = 'triangular'
elif dist_nwc == 'Uniform':
    lower_input = e1
    upper_input = e2
    parameter_dict_1['NWC ratio'] = lower_input
    parameter_dict_2['NWC ratio'] = upper_input
    parameter_dict_distribution['NWC ratio'] = 'uniform'


equity_value_list = []
revenue_list_of_lists = []
ebit_list_of_lists = []
parameter_dict_1['latest revenue'] = company.income_statement.loc['Total Revenue', company.income_statement.columns[-1]]
parameter_dict_1['net debt'] = company.inputs.loc['netDebt', 'Historical average']
if simulation_iterations > 1000:
    simulation_iterations = 1000
elif simulation_iterations < 0:
    simulation_iterations = 100
for i in range(int(simulation_iterations)):
    model_input = calculate_value_distribution(parameter_dict_1, parameter_dict_2, parameter_dict_distribution)
    forecast_df = company.get_free_cash_flow_forecast(model_input)
    revenue_list_of_lists.append(forecast_df.loc['Revenues'])
    ebit_list_of_lists.append(forecast_df.loc['EBIT'])
    model_output, equity_value = company.discount_free_cash_flows(forecast_df, model_input, discount_rate, terminal_growth)
    equity_value_list.append(equity_value)



mean_equity_value = np.mean(equity_value_list)
stddev_equity_value = np.std(equity_value_list)
print()

try:
    info = yf.Ticker(ticker_input).info
    if info:
        if "marketCap" in info:
            market_cap = info['marketCap']
            market_cap_str = str(round(market_cap/1000000000, 2)) + " blns$"
except Exception as e:
    print(e)
    
print('Market Cap: ', market_cap_str)
print('Mean equity value: $' + str(comma_format(mean_equity_value)))
print(f'Margin of Safety: {mean_equity_value/market_cap-1:.2f}')
print()
print('Equity value std. deviation: $' + str(comma_format(stddev_equity_value)))
print(f'Std Margin of Safety: {market_cap/stddev_equity_value-1:.2f}')
print()



font_1 = {
    'family' : 'Arial',
        'size' : 12
}
font_2 = {
    'family' : 'Arial',
        'size' : 14
}

fig1 = plt.figure()
# plt.style.use('seaborn-whitegrid')
plt.title(ticker_input + ' Monte Carlo Simulation', fontdict = font_1)
plt.xlabel('Equity value (in $)', fontdict = font_1)
plt.ylabel('Number of occurences', fontdict = font_1)
plt.hist(equity_value_list, bins = 50, color = '#006699', edgecolor = 'black')
# fig1.show()



fig2 = plt.figure()
x = range(6)[1:6]
# plt.style.use('seaborn-whitegrid')
plt.title('Revenue Forecast Monte Carlo Simulation', fontdict = font_2)
plt.xticks(ticks = x)
plt.xlabel('Year', fontdict = font_2)
plt.ylabel('Revenue (in $)', fontdict = font_2)
for i in revenue_list_of_lists:
    plt.plot(x, i)
# fig2.show()


fig3 = plt.figure()
x = range(6)[1:6]
# plt.style.use('seaborn-whitegrid')
plt.title('EBIT Forecast Monte Carlo Simulation', fontdict = font_2)
plt.xticks(ticks = x)
plt.xlabel('Year', fontdict = font_2)
plt.ylabel('EBIT (in $)', fontdict = font_2)
for i in ebit_list_of_lists:
    plt.plot(x, i)
plt.show()
