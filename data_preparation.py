def prepare_data_test(x):
    x['groupe'] = x['PassengerId'].str[:4]
    counts= x['groupe'].value_counts()
    counts  
    x['is_travel_group']=x['groupe'].apply(lambda x: 1 if counts[x]>1 else 0 )
    x['family']=x['Name'].str.split(" ").str[-1]
    x['family_journey'] = (x['is_travel_group'] == 1) & (x['family'].map(x['family'].value_counts()) >= 2)
    x['family_journey'] = x['family_journey'].astype(int)
    x['deck'] = x['Cabin'].str[0]
    x['number_cabin'] = x['Cabin'].str[1:].str.extract('(\d+)').astype(float)
    x['side']=x['Cabin'].str[-1]
    x['luxury'] = 0
    columns_to_check = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for index, row in x.iterrows():
        count = 0
        for column in columns_to_check:
            if row[column] > 1:
                count += 1
        x.at[index, 'luxury'] = count
    x.drop(['PassengerId','Cabin','Name','groupe','family','number_cabin'], axis=1,inplace=True)
    x['is_travel_group']=x['is_travel_group'].astype('object')
    x['family_journey']=x['family_journey'].astype('object')
    ordinal_features = ['luxury']
    categorical_features=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'deck', 'side','is_travel_group','family_journey']
    numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    return x


def prediction_instance(model, dataf):
    # Transform dataf using the pipeline
    tempo = full_pipeline_loaded.transform(dataf)
    
    # Get feature names from the model
    feature_names = model.get_booster().feature_names
    
    # Create DataFrame with features as columns
    data = pd.DataFrame(columns=feature_names, data=tempo)
    # fill with NA for value not present
    data.fillna(0, inplace=True)
    # Use model to predict outcome and probability
    outcome = model.predict(data)
    probability = model.predict_proba(data)[:, 1]  
    
    # Return outcome and probability
    return outcome[0], probability[0]


# Create a fonction to predict the outcome and the probability
def predict_outcome_and_probability(model, features):
    # DataFrame with features as columns
    data = pd.DataFrame(columns=model.get_booster().feature_names, data=[features])
    
    # fill with NA for value not present
    data.fillna(0, inplace=True)
    
    # Use model to predict outcome and probability
    outcome = model.predict(data)
    probability = model.predict_proba(data)[:, 1]  
    
    # Return outcome and probability
    return outcome[0], probability[0]
