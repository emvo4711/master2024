import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Test engagement data
engagement_data = {
    'user1': {'fund1': {'search': np.random.randint(0, 8), 'view': np.random.randint(0, 15), 'performance': np.random.randint(0, 12), 'terms': np.random.randint(0, 6), 'contact': np.random.choice([0, 1]), 'preference': 2},
              'fund2': {'search': 0, 'view': 0, 'performance': 0, 'terms': 0, 'contact': 0, 'preference': 0},
              'fund3': {'search': np.random.randint(0, 12), 'view': np.random.randint(0, 20), 'performance': np.random.randint(0, 15), 'terms': np.random.randint(0, 8), 'contact': np.random.choice([0, 1]), 'preference': -1},
              'fund4': {'search': np.random.randint(0, 10), 'view': np.random.randint(0, 18), 'performance': np.random.randint(0, 12), 'terms': np.random.randint(0, 6), 'contact': np.random.choice([0, 1]), 'preference': 1},
              'fund5': {'search': 0, 'view': 0, 'performance': 0, 'terms': 0, 'contact': 0, 'preference': 0}},

    'user2': {'fund1': {'search': np.random.randint(0, 15), 'view': np.random.randint(0, 20), 'performance': np.random.randint(0, 18), 'terms': np.random.randint(0, 10), 'contact': np.random.choice([0, 1]), 'preference': 1},
              'fund2': {'search': np.random.randint(0, 12), 'view': np.random.randint(0, 18), 'performance': np.random.randint(0, 15), 'terms': np.random.randint(0, 8), 'contact': np.random.choice([0, 1]), 'preference': 2},
              'fund3': {'search': 0, 'view': 0, 'performance': 0, 'terms': 0, 'contact': 0, 'preference': 0},
              'fund4': {'search': np.random.randint(0, 12), 'view': np.random.randint(0, 20), 'performance': np.random.randint(0, 15), 'terms': np.random.randint(0, 8), 'contact': np.random.choice([0, 1]), 'preference': 1},
              'fund5': {'search': 0, 'view': 0, 'performance': 0, 'terms': 0, 'contact': 0, 'preference': 0}},

    'user3': {'fund1': {'search': 0, 'view': 0, 'performance': 0, 'terms': 0, 'contact': 0, 'preference': 0},
              'fund2': {'search': np.random.randint(0, 12), 'view': np.random.randint(0, 20), 'performance': np.random.randint(0, 15), 'terms': np.random.randint(0, 8), 'contact': np.random.choice([0, 1]), 'preference': 2},
              'fund3': {'search': np.random.randint(0, 25), 'view': np.random.randint(0, 30), 'performance': np.random.randint(0, 25), 'terms': np.random.randint(0, 18), 'contact': np.random.choice([0, 1]), 'preference': 0},
              'fund4': {'search': 0, 'view': 0, 'performance': 0, 'terms': 0, 'contact': 0, 'preference': 0},
              'fund5': {'search': np.random.randint(0, 30), 'view': np.random.randint(0, 72), 'performance': np.random.randint(0, 70), 'terms': np.random.randint(0, 50), 'contact': np.random.choice([0, 1]), 'preference': 2}},
              
    'user4': {'fund1': {'search': np.random.randint(0, 10), 'view': np.random.randint(0, 18), 'performance': np.random.randint(0, 10), 'terms': np.random.randint(0, 5), 'contact': np.random.choice([0, 1]), 'preference': 2},
              'fund2': {'search': np.random.randint(0, 50), 'view': np.random.randint(0, 100), 'performance': np.random.randint(0, 80), 'terms': np.random.randint(0, 30), 'contact': np.random.choice([0, 1]), 'preference': 1},
              'fund3': {'search': 0, 'view': 0, 'performance': 0, 'terms': 0, 'contact': 0, 'preference': 0},
              'fund4': {'search': np.random.randint(0, 12), 'view': np.random.randint(0, 20), 'performance': np.random.randint(0, 15), 'terms': np.random.randint(0, 8), 'contact': np.random.choice([0, 1]), 'preference': 2},
              'fund5': {'search': 0, 'view': 0, 'performance': 0, 'terms': 0, 'contact': 0, 'preference': 0}}
}


flattened_data = {}
for user, user_data in engagement_data.items():
    for fund, metrics in user_data.items():
        for metric, value in metrics.items():
            flattened_data[(user, fund, metric)] = value

flattened_df = pd.DataFrame(flattened_data.values(), index=pd.MultiIndex.from_tuples(flattened_data.keys())).unstack()

scaler = MinMaxScaler()
normalized_engagement = scaler.fit_transform(flattened_df)

# Weights for engagement metrics
weights = {
    'search': 0.1,
    'view': 0.15,
    'performance': 0.2,
    'terms': 0.3,
    'contact': 0.8,
    'preference': 0.6
}

weighted_sum = normalized_engagement.dot([weights[metric] for metric in flattened_df.columns.levels[1]])

min_score = weighted_sum.min()
max_score = weighted_sum.max()
normalized_score = (weighted_sum - min_score) / (max_score - min_score)

num_users = len(engagement_data)
num_funds = len(next(iter(engagement_data.values())))


ratings_df = pd.DataFrame(normalized_score.reshape(num_users, num_funds),
                          index=engagement_data.keys(),
                          columns=engagement_data[next(iter(engagement_data))].keys())

for fund in engagement_data[next(iter(engagement_data))].keys():
    ratings_df[fund].fillna(0, inplace=True)

# User-item matrix
print("User-Item Matrix (Normalized Ratings):\n")
print(ratings_df)

# Test fund data
fund_properties = {
    'Fund': ['fund1', 'fund2', 'fund3', 'fund4', 'fund5'],
    'Asset Class': ['Private Equity', 'Private Equity', 'Infrastructure', 'Infrastructure','Private Equity'],
    'Strategy': ['Buyout', 'Buyout', 'Venture', 'Venture', 'Venture'],
    'Geography': ['North America', 'Europe', 'Europe', 'Asia', 'North America'],
    'Sector': ['Healthcare', 'Technology', 'Financials', 'Healthcare', 'Technology'],
    'Stage': ['Large', 'Small', 'Large', 'Large', 'Mid'],
    'ESG': ['Article 6, UNPRI', 'Article 9, UNPRI', 'Article 8, UNPRI', 'Article 6, UNPRI', 'Article 6, UNPRI'],
    'Ownership': ['Private', 'Private', 'Public', 'Private', 'Private'],
    'Type': ['Secondary', 'Secondary', 'Co-Invest', 'Co-Invest', 'Secondary']
}

fund_properties_df = pd.DataFrame(fund_properties)

fund_properties_df.set_index('Fund', inplace=True)

fund_properties_encoded = pd.get_dummies(fund_properties_df)


# Weights for fund property classes
property_weights = {
    'Asset Class': 0.125,
    'Strategy': 0.125,
    'Geography': 0.125,
    'Sector': 0.125,
    'Stage': 0.125,
    'ESG': 0.125,
    'Ownership': 0.125,
    'Type': 0.125
}

encoded_columns = []
for column, weight in property_weights.items():
    encoded_column = pd.get_dummies(fund_properties_df[column])
    encoded_column *= weight
    encoded_columns.append(encoded_column)

weighted_encoded_df = pd.concat(encoded_columns, axis=1)

fund_similarity = cosine_similarity(weighted_encoded_df)

fund_similarity_df = pd.DataFrame(fund_similarity, index=fund_properties_df.index, columns=fund_properties_df.index)

# Fund similarity matrix
print("\nFund Similarity Matrix (Cosine Similarity):\n")
print(fund_similarity_df)

# Fund similarity threshold

fund_similarity_threshold = 0.5

# User similarity threshold
user_similarity_threshold = 0.0

user_similarity = pd.DataFrame(index=ratings_df.index, columns=ratings_df.index)


for user1 in ratings_df.index:
    for user2 in ratings_df.index:
        if user1 != user2: 
            common_funds = ratings_df.loc[user1].notna() & ratings_df.loc[user2].notna()
            if common_funds.any(): 
                similar_funds_indices = [fund for fund in common_funds.index if fund_similarity_df.loc[fund, fund] > fund_similarity_threshold]
                if similar_funds_indices: 
                    user1_engagement = ratings_df.loc[user1, similar_funds_indices]
                    user2_engagement = ratings_df.loc[user2, similar_funds_indices]
                    similarity_score = cosine_similarity([user1_engagement.values], [user2_engagement.values])[0][0]
                    if similarity_score > user_similarity_threshold:
                        user_similarity.loc[user1, user2] = similarity_score

np.fill_diagonal(user_similarity.values, 1)

# User similarity matrix
print("User Similarity Matrix:")
print(user_similarity)


# Recommendation
recommendations = {}

for user in ratings_df.index:
    similar_users = user_similarity[user].sort_values(ascending=False)[1:]

    similar_user_engagement = ratings_df.loc[similar_users.index].mean(axis=0)
    
    recommended_funds = similar_user_engagement.sort_values(ascending=False).index

    recommended_funds = recommended_funds.difference(ratings_df.loc[user][ratings_df.loc[user].notna()].index)
  
    recommended_funds = recommended_funds.union(ratings_df.loc[user][ratings_df.loc[user] == 0].index)

    recommended_funds = [fund for fund in recommended_funds if engagement_data[user][fund]['preference'] != -1]
    
    recommendations[user] = recommended_funds


user_preferences = {'Asset Class': 'Private Equity'}  # Test user filter preferences

for user, recommended_funds in recommendations.items():
    recommended_funds_filtered = [fund for fund in recommended_funds if fund_properties_df.loc[fund, 'Asset Class'] == user_preferences['Asset Class']]
    recommendations[user] = recommended_funds_filtered

# Final recommendations for each user after filtering
for user, recommended_funds in recommendations.items():
    print(f"Recommendations for {user}: {recommended_funds}")
