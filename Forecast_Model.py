#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install matplotlib seaborn


# In[3]:


pip install Streamlit


# In[5]:


import streamlit as st
import pandas as pd
import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt
import seaborn as sns

# 设置页面标题
st.title("Forecast Model")

# 目标人群模块
st.header("Target Pop")
total_population = st.number_input("local Pop", value=1000000)
problem_occurrence_rate = st.slider("Problem %", min_value=0.0, max_value=1.0, value=0.2)
problem_occurrence_growth_rate = st.slider("Problem Growth %", min_value=-0.1, max_value=0.1, value=0.05)
theory_conformity_rate = st.slider("Solution Bought %", min_value=0.0, max_value=1.0, value=0.8)
technology_theory_iteration = st.number_input("Solution Iteration (SI)", value=0)
technology_theory_probability = st.slider("SI probability", min_value=0.0, max_value=1.0, value=0.5)
growth_type = st.selectbox("type of growth", ["Lin", "Geo"])

# 蒙特卡洛模拟次数
monte_carlo_simulations = st.number_input("times to sim", value=1000, min_value=1)

# 计算目标人群人口
@lru_cache(maxsize=128)
def simulate_target_population(total_population, problem_occurrence_rate, problem_occurrence_growth_rate, theory_conformity_rate, technology_theory_probability, growth_type, monte_carlo_simulations):
    target_populations = []
    for _ in range(monte_carlo_simulations):
        if np.random.rand() < technology_theory_probability:
            if growth_type == "Lin":
                theory_conformity_rate_sim = theory_conformity_rate * (1 + np.random.uniform(0, 1))
            elif growth_type == "Geo":
                theory_conformity_rate_sim = theory_conformity_rate * (1 + np.random.exponential(scale=0.5))
        else:
            theory_conformity_rate_sim = theory_conformity_rate
        
        target_population_sim = total_population * problem_occurrence_rate * (1 + problem_occurrence_growth_rate) * theory_conformity_rate_sim
        target_populations.append(target_population_sim)
    
    return np.array(target_populations)

target_populations = simulate_target_population(total_population, problem_occurrence_rate, problem_occurrence_growth_rate, theory_conformity_rate, technology_theory_probability, growth_type, monte_carlo_simulations)

# 输出目标人群人口区间
mean_target_population = np.mean(target_populations)
std_target_population = np.std(target_populations)
lower_bound = max(0, mean_target_population - 2 * std_target_population)
upper_bound = mean_target_population + 2 * std_target_population
st.write(f"Target Pop: {lower_bound:.2f} - {upper_bound:.2f}")

# 市场份额模块
st.header("Mkt Share")
existing_products_count = st.number_input("# of On Mkt Sub", value=0)
future_products_count = st.number_input("# of Potential Sub", value=0)

# 已上市产品部分
existing_product_sales = []
price_discount_intervals = []
volume_increase_intervals = []
for i in range(existing_products_count):
    sales = st.number_input(f"On Mkt Sub {i+1} Quantity", value=0)
    use_discount = st.checkbox(f"On Mkt Sub {i+1} Price Give to Access", value=True)
    if use_discount:
        price_discount_min = st.slider(f"On Mkt Sub {i+1} Min Off for Medicare", min_value=0.0, max_value=1.0, value=0.4)
        price_discount_max = st.slider(f"On Mkt Sub {i+1} Max Off for Medicare", min_value=0.0, max_value=1.0, value=0.6)
    else:
        price_discount_min = 0.0
        price_discount_max = 0.0
    use_volume_increase = st.checkbox(f"On Mkt Sub {i+1} Medicare Boost", value=True)
    if use_volume_increase:
        volume_increase_min = st.slider(f"On Mkt Sub {i+1} Min Medicare Boost", min_value=0.0, max_value=5.0, value=2.0)
        volume_increase_max = st.slider(f"On Mkt Sub {i+1} Max Medicare Boost", min_value=0.0, max_value=5.0, value=4.0)
    else:
        volume_increase_min = 0.0
        volume_increase_max = 0.0
    existing_product_sales.append(sales)
    price_discount_intervals.append((price_discount_min, price_discount_max))
    volume_increase_intervals.append((volume_increase_min, volume_increase_max))

# 待上市产品部分
future_product_probabilities = []
future_product_years = []
for i in range(future_products_count):
    probability = st.slider(f"Potential Sub {i+1} LOA", min_value=0.0, max_value=1.0, value=0.5)
    year = st.number_input(f"Potential Sub {i+1} Time to Launch(Year)", value=1)
    future_product_probabilities.append(probability)
    future_product_years.append(year)

# 预测时长
forecast_years = st.number_input("Period(Year)", value=5, min_value=1)

# 特定大促效应
promotion_years = st.multiselect("Year with Special Promotion", list(range(1, forecast_years + 1)))
promotion_effect = st.slider("Effect of Special Promotion", min_value=0.0, max_value=1.0, value=0.3)

# 计算市场份额
def simulate_market_share(target_populations, existing_sales, price_discount_intervals, volume_increase_intervals, future_probabilities, future_years, forecast_years, promotion_years, promotion_effect, monte_carlo_simulations):
    market_shares = []
    for target_population in target_populations:
        current_market_shares = [0] * forecast_years
        existing_sales_discounted = []
        for sales, (discount_min, discount_max) in zip(existing_sales, price_discount_intervals):
            if discount_min == 0.0 and discount_max == 0.0:
                discount = 0.0
            else:
                discount = np.random.uniform(discount_min, discount_max)
            existing_sales_discounted.append(sales * (1 - discount))
        total_existing_sales = sum(existing_sales_discounted)
        
        # 已上市产品的市场份额
        for year in range(forecast_years):
            existing_sales_growth = []
            for sales, (increase_min, increase_max) in zip(existing_sales_discounted, volume_increase_intervals):
                if increase_min == 0.0 and increase_max == 0.0:
                    increase = 0.0
                else:
                    increase = np.random.uniform(increase_min, increase_max)
                existing_sales_growth.append(sales * (1 + increase) * (1 + np.random.normal(0, 0.1)))  # 引入随机波动
            total_existing_sales_growth = sum(existing_sales_growth)
            current_market_shares[year] += total_existing_sales_growth / (total_existing_sales_growth + target_population * 0.05)  # 假设剩余5%为新进入者的空间
            
            # 特定大促效应
            if year + 1 in promotion_years:
                current_market_shares[year] *= (1 + promotion_effect)
        
        # 待上市产品的市场份额
        for i in range(future_products_count):
            if np.random.rand() < future_probabilities[i]:
                for year in range(forecast_years):
                    if year >= future_years[i]:
                        current_market_shares[year] -= total_existing_sales_growth * 0.05 / (future_products_count + 1)  # 新产品抢占5%的市场份额
        
        market_shares.append(current_market_shares)
    
    # 计算市场份额的平均值和标准差
    market_shares_array = np.array(market_shares)
    mean_market_shares = np.mean(market_shares_array, axis=0)
    std_market_shares = np.std(market_shares_array, axis=0)
    
    return mean_market_shares, std_market_shares

mean_market_shares, std_market_shares = simulate_market_share(target_populations, existing_product_sales, price_discount_intervals, volume_increase_intervals, future_product_probabilities, future_product_years, forecast_years, promotion_years, promotion_effect, monte_carlo_simulations)

# 输出市场份额区间
st.write("Range of Mkt Share in Period")
for year, mean_share, std_share in zip(range(1, forecast_years + 1), mean_market_shares, std_market_shares):
    lower_bound = max(0, mean_share - 2 * std_share)
    upper_bound = min(1, mean_share + 2 * std_share)
    st.write(f"Year{year} Mkt Share: {lower_bound:.2%} - {upper_bound:.2%}")

# 销售预测模块
st.header("Sales Forecast")
usage_frequency = st.number_input("Freq & Period of Usage (to pay)", value=1)
product_retail_price = st.number_input("Retail Price", value=100)
wholesale_discount_rate = st.slider("Wholesale Discount", min_value=0.0, max_value=1.0, value=0.1)
sales_volume_growth_rate = st.slider("Vol Growth %", min_value=-0.1, max_value=0.1, value=0.05)

# 计算逐年销量、销量增长率、市场规模、收入和收入增长率
years = list(range(1, forecast_years + 1))
sales_volumes = []
market_sizes = []
incomes = []
income_growths = []

for year, mean_share, std_share in zip(years, mean_market_shares, std_market_shares):
    lower_bound = max(0, mean_share - 2 * std_share)
    upper_bound = min(1, mean_share + 2 * std_share)
    sales_volume = np.mean([tp * (lower_bound + upper_bound) / 2 * usage_frequency for tp in target_populations])
    sales_volumes.append(sales_volume)
    
    # 销量增长
    if year > 1:
        sales_growth = (sales_volume - sales_volumes[year - 2]) / sales_volumes[year - 2]
    else:
        sales_growth = 0
    
    # 零售价计算
    competition_discount = 0.05 * (1 + 0.2 * (future_products_count - 1))  # 每增加一家竞争者，降价均值增加0.2
    retail_price = product_retail_price * (1 - np.random.uniform(0.03, 0.07) - competition_discount)
    
    # 市场规模
    market_size = sales_volume * retail_price
    market_sizes.append(market_size)
    
    # 收入
    income = market_size * (1 - wholesale_discount_rate)
    incomes.append(income)
    
    # 特定大促效应
    if year in promotion_years:
        income *= (1 + promotion_effect)
    
    income_growths.append(sales_growth)

# 将结果存储到DataFrame中
sales_data = pd.DataFrame({
    "Year": years,
    "Vol": sales_volumes,
    "Vol Growth": [f"{g:.2%}" for g in income_growths],
    "Mkt Size": market_sizes,
    "Rev": incomes,
    "Rev Growth": [f"{g:.2%}" for g in income_growths]
})

st.write("Result of Sales Forecast:")
st.dataframe(sales_data)

# 损益测算模块
st.header("Finance")
unit_cost = st.number_input("Unit Cost", value=50)
special_promotion_effect = st.slider("特别大促效果", min_value=0.0, max_value=1.0, value=0.1)
sales_expense_rate = st.slider("Sales Exp", min_value=0.0, max_value=1.0, value=0.1)
management_expense_rate = st.slider("Mgmt Exp", min_value=0.0, max_value=1.0, value=0.1)
financial_expense_rate = st.slider("Fin Exp", min_value=0.0, max_value=1.0, value=0.1)

# 计算逐年总成本、总毛利和利润
total_costs = [sales_volume * unit_cost for sales_volume in sales_volumes]
gross_profits = [income - total_cost for income, total_cost in zip(incomes, total_costs)]
operating_expenses = [income * (sales_expense_rate + management_expense_rate + financial_expense_rate) for income in incomes]

# 特定大促产生的让利预算
promotion_budgets = [income * special_promotion_effect if year in promotion_years else 0 for year, income in zip(years, incomes)]
operating_expenses_with_promotion = [oe + pb for oe, pb in zip(operating_expenses, promotion_budgets)]
profits = [gross_profit - operating_expense for gross_profit, operating_expense in zip(gross_profits, operating_expenses_with_promotion)]

# 将结果存储到DataFrame中
profit_data = pd.DataFrame({
    "Year": years,
    "Cost": total_costs,
    "Gross Margin": gross_profits,
    "Mgmt Budgets": [oe - pb for oe, pb in zip(operating_expenses, promotion_budgets)],
    "Special Promotion Exp": promotion_budgets,
    "Total Mgmt Exp": operating_expenses_with_promotion,
    "Profits": profits
})

st.write("Result of Finance:")
st.dataframe(profit_data)

# 可视化部分
st.header("Figures")

# 用户选择要可视化的指标
selected_metric = st.selectbox("Choose Your Scale", ["Vol", "Rev", "Mkt Size", "Gross Margin", "Profits"])

# 获取选中的指标数据
metric_data = profit_data if selected_metric in ["Cost", "Gross Margin", "Mgmt Budgets", "Special Promotion Exp", "Total Mgmt Exp", "Profts"] else sales_data
metric_values = metric_data[selected_metric].values

# 创建箱型图
plt.figure(figsize=(12, 6))
sns.boxplot(x="Year", y=selected_metric, data=metric_data)
plt.title(f"{selected_metric}Box")
plt.xlabel("Year")
plt.ylabel(selected_metric)
st.pyplot(plt)

# 创建直方图
plt.figure(figsize=(12, 6))
sns.histplot(metric_values, kde=True, bins=30)
plt.title(f"{selected_metric}Hist")
plt.xlabel(selected_metric)
plt.ylabel("Freq")
st.pyplot(plt)

# 运行应用
if __name__ == "__main__":
    st.write("Forecast Model Launched")


# In[ ]:




