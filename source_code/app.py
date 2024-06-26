from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
from keras.models import load_model
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA

app = Flask(__name__)

# Cargar modelos y escaladores al iniciar la aplicación
modelos = {}
scalers = {}
crypto_list = pd.read_csv('data/clasificacion.csv')['symbol'].tolist()

# Cargar modelos LSTM y escaladores
for symbol in crypto_list:
    modelos[symbol] = load_model(f'data/modelos_guardados/modelo_{symbol}.keras')
    scalers[symbol] = {}
    for col in ['volume_24h', 'close', 'open', 'high', 'low', 'market_cap']:
        scalers[symbol][col] = joblib.load(f'data/escaladores_guardados/scaler_{symbol}_{col}.pkl')

# Cargar modelo de clasificación
clf = joblib.load('data/RandomForest.pkl')

# Cargar datos históricos de criptomonedas
crypto_data = pd.read_csv('data/cmcHistorical(2024)(Model).csv')


# Load cryptocurrencies from a CSV file
cryptocurrencies = pd.read_csv('data/criptomonedas.csv')['symbol'].tolist()

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])

def predict():
    symbol = request.form['symbol']
    symbol_data = crypto_data[crypto_data['symbol'] == symbol]
    
    avg_volatility_daily = symbol_data['log_return'].std()
    total_volatility = avg_volatility_daily * np.sqrt(len(symbol_data))
    price_variation_multiplier = (symbol_data["close"].max() - symbol_data["close"].iloc[0]) / symbol_data["close"].iloc[0]
    avg_daily_market_cap = symbol_data["market_cap"].mean()
    max_drawdown = symbol_data["drawdown"].min()
    avg_volume_24h = symbol_data["volume_24h"].mean()
    
    dictCrypto = {
        "symbol": symbol,
        "total_volatility": total_volatility,
        "price_variation_multiplier": price_variation_multiplier,
        "avg_daily_market_cap": avg_daily_market_cap,
        "max_drawdown": max_drawdown,
        "avg_volume_24h": avg_volume_24h
    }
    dfClasTest = pd.DataFrame([dictCrypto])

    X_test = dfClasTest.drop(columns=['symbol', 'price_variation_multiplier'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_test_scaled = scaler.fit_transform(X_test)
    clf_prediction = clf.predict(X_test_scaled)

    clases = pd.read_csv('data/clasificacion.csv')
    symbol_list = clases[clases["type"] == clf_prediction[0]]["symbol"].tolist()
    #if clf_prediction[0] == "Estable":
        #symbol_list = ["ADA","AMB"]
    #elif clf_prediction[0] == "Especulativa":
        #symbol_list = ["BLZ","BNT"]
    #elif clf_prediction[0] == "Oportunidad":
        #symbol_list = ["GARD","STX"]   
    predictionIncrements = []

    for s in symbol_list:
        lstm_features = symbol_data[['open', 'high', 'low', 'close', 'volume_24h', 'market_cap']].copy()
        
        # Normalizar las características usando los escaladores específicos
        for col in lstm_features.columns:
            lstm_features[col] = scalers[s][col].transform(lstm_features[[col]])

        pca = PCA(n_components=3)
        pca_data = lstm_features.values
        pca_result = pca.fit_transform(pca_data)

        pca_columns = [f'pca_{i+1}' for i in range(pca.n_components_)]
        df_pca = pd.DataFrame(pca_result, columns=pca_columns, index=lstm_features.index)
        lstm_features = df_pca.copy()
        volatility_scaler = MinMaxScaler(feature_range=(0, 1))
        volatility_df = pd.DataFrame(symbol_data['volatility'])  # Asegúrate de pasar un DataFrame
        lstm_features['volatility'] = volatility_scaler.fit_transform(volatility_df)
        lstm_features.reset_index(drop=True, inplace=True)
        
        features = ['pca_1', 'pca_2', 'pca_3', 'volatility']    
        
        recent_data = lstm_features[features].values[-120:]  # Obtener los últimos 90 días de datos
        recent_data = recent_data.reshape((1, 120, len(features)))

        # Predicción a 90 días en el futuro
        predictions = []
        for _ in range(210):
            next_prediction = modelos[s].predict(recent_data)
            predictions.append(next_prediction[0][0])

            # Crear un array con los valores necesarios para actualizar recent_data
            next_prediction_scaled = np.zeros(len(features))
            next_prediction_scaled[0] = next_prediction[0][0]  # Asumiendo que la predicción es del primer PCA

            # Actualizar recent_data para incluir la nueva predicción y eliminar el día más antiguo
            recent_data = np.append(recent_data[:, 1:, :], next_prediction_scaled.reshape((1, 1, len(features))), axis=1)

        # Desnormalizar las predicciones
        predicted_prices = scalers[s]['close'].inverse_transform(np.array(predictions).reshape(-1, 1))
        predictionIncrements.append(((predicted_prices.max() - symbol_data['close'].iloc[0]) / symbol_data['close'].iloc[0]) * 100)

    worst_result = min(predictionIncrements)
    best_result = max(predictionIncrements)
    return jsonify({
        'symbol': symbol,
        'worst_result': worst_result,
        'best_result': best_result,
        'type': clf_prediction[0]
    })


@app.route('/')
def main():
    return render_template("index.html")

@app.route('/recommender')
def recommender():
    return render_template('recommender.html', cryptos=cryptocurrencies)    