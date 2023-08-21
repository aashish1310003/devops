
import numpy as np

from keras.layers import Dense #Activation, BatchNormalization #Dropout
#from keras import regularizers
import pandas as pd
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://user:user@solarcluster0.lbfnszw.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your Predition and working on it. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

#client = pymongo.MongoClient('mongodb://127.0.0.1:27017/')
class Predict:

    def request_train_data_db(self):


        mydb = client['solar']
        collection = mydb["weather-dataset"]

        field_order = [
            "ghi",
            "ghi10",
            "ghi90",
            "ebh",
            "dni",
            "dni10",
            "dni90",
            "dhi",
            "air_temp",
            "zenith",
            "azimuth",
            "cloud_opacity",
            "period_end"
        ]

        # Retrieve documents
        data = list(collection.find())

        # Create a sorted list of dictionaries based on the custom order
        sorted_data = [{field: document[field] for field in field_order} for document in data]

        # cursor = collection.find()
        # Retrieve data from the collection

        df_sheet1 = pd.DataFrame(sorted_data)  # Convert cursor data to a list and then to a DataFrame

        collection1 = mydb["solar-dataset"]

        field_order = [

            "pv_estimate", "period_end", "period"
        ]

        # Retrieve documents
        data1 = list(collection1.find())

        # Create a sorted list of dictionaries based on the custom order
        sorted_data1 = [{field: document[field] for field in field_order} for document in data1]

        #cursor1 = collection1.find()  # Retrieve data from the collection
        df_sheet2 = pd.DataFrame(sorted_data1)
        # common_periods = set(df_sheet1['period_end']).intersection(df_sheet2['period_end'])
        #
        # # Filter data based on common 'period_end' values
        # common_data_sheet1 = df_sheet1[df_sheet1['period_end'].isin(common_periods)]
        # common_data_sheet2 = df_sheet2[df_sheet2['period_end'].isin(common_periods)]
        print("inside the Prediction Class: ", df_sheet1.head())
        return df_sheet1, df_sheet2
    def request_test_data_db(self):


        mydb = client['test']
        collection = mydb["weather-dataset"]

        cursor = collection.find()
        field_order = [
            "ghi",
            "ghi10",
            "ghi90",
            "ebh",
            "dni",
            "dni10",
            "dni90",
            "dhi",
            "air_temp",
            "zenith",
            "azimuth",
            "cloud_opacity",
            "period_end"
        ]

        # Retrieve documents
        data = list(collection.find())

        # Create a sorted list of dictionaries based on the custom order
        sorted_data = [{field: document[field] for field in field_order} for document in data]
        # Retrieve data from the collection
        df_sheet1 = pd.DataFrame(sorted_data)  # Convert cursor data to a list and then to a DataFrame

        collection1 = mydb["solar-dataset"]
        field_order = [

            "pv_estimate", "period_end", "period"
        ]

        # Retrieve documents
        data1 = list(collection1.find())

        # Create a sorted list of dictionaries based on the custom order
        sorted_data1 = [{field: document[field] for field in field_order} for document in data1]

        df_sheet2 = pd.DataFrame(sorted_data1)

        return df_sheet1, df_sheet2

    def predection_power(self,file):
        import datetime
        import matplotlib.pyplot as plt
        #import seaborn as sns

        # dts = pd.read_csv(
        #     '/home/ec2-user/solar/{}.csv'.format(file[1]))

        # dts = pd.read_csv(
        #     "D:/projects/augproj/{}.csv".format(file[1]))
        #
        # # dt = pd.read_csv(
        # #     '/home/ec2-user/solar/{}.csv'.format(file[0]))
        #
        # dt = pd.read_csv(
        #      "D:/projects/augproj/{}.csv".format(file[0]))
        dts, dt = self.request_test_data_db()

        dts.head(10)

        X = dts.iloc[:, :-2].values
        y = dt.iloc[:, :-2].values
        print("test = ",X.shape, y.shape)
        y = np.reshape(y, (-1, 1))

        # from sklearn.model_selection import train_test_split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        # print("Train Shape: {} {} \nTest Shape: {} {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
        X_trai, y_trai = self.request_train_data_db()
        # X_trai = pd.read_csv(
        #     "C:/Users/91875/PycharmProjects/pythonProject/weather_common_data_sheet1.csv")
        X_train = X_trai.iloc[:, :-2].values
        # y_trai = pd.read_csv(
        #     "C:/Users/91875/PycharmProjects/pythonProject/solar_common_data_sheet2.csv")
        y_train = y_trai.iloc[:, :-2].values
        y_train = np.reshape(y_train, (-1, 1))
        print("train = ",X_train.shape, y_train.shape)
        print(X_trai.head)
        X_test = X
        y_test = y
        from sklearn.preprocessing import StandardScaler
        import tensorflow as tf
        #from tensorflow.keras.layers import Dense

        # Assuming you have X_train and y_train data loaded

        # Input scaling
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)

        # Outcome scaling
        sc_y = StandardScaler()
        y_train = sc_y.fit_transform(y_train)

        def create_spfnet(n_layers, n_activation, kernels):
            model = tf.keras.models.Sequential()
            for i, nodes in enumerate(n_layers):
                if i == 0:
                    model.add(
                        Dense(nodes, kernel_initializer=kernels, activation=n_activation, input_dim=X_train.shape[1]))
                else:
                    model.add(Dense(nodes, activation=n_activation, kernel_initializer=kernels))

            model.add(Dense(1))
            model.compile(loss='mse',
                          optimizer='adam',
                          metrics=[tf.keras.metrics.RootMeanSquaredError()])
            return model

        spfnet = create_spfnet([32, 64], 'relu', 'normal')
        spfnet.summary()

        hist = spfnet.fit(X_train, y_train, batch_size=32, epochs=150, verbose=2)

        # Preprocess new input data
        new_data_scaled = sc_X.transform(X_test)

        # Make predictions
        predictions_scaled = spfnet.predict(new_data_scaled)

        # Inverse transform predictions to get original scale
        predictions_original_scale = sc_y.inverse_transform(predictions_scaled)

        # Print the predictions

        predictions_original_scale = sc_y.inverse_transform(predictions_scaled)

        # Round each value to two decimal places
        y_pred_orig = np.round(predictions_original_scale, 2)

        #end of model

        plt.plot(hist.history['root_mean_squared_error'])
        # plt.plot(hist.history['val_root_mean_squared_error'])
        plt.title('Root Mean Squares Error')
        plt.xlabel('Epochs')
        plt.ylabel('error')
        plt.show()

        spfnet.evaluate(X_train, y_train)

        #from sklearn.metrics import mean_squared_error

        # y_pred = spfnet.predict(X_test)  # get model predictions (scaled inputs here)
        # y_pred_orig = sc_y.inverse_transform(y_pred)  # unscale the predictions
        #y_test_orig = sc_y.inverse_transform(y_test)

        train_pred = spfnet.predict(X_train)  # get model predictions (scaled inputs here)
        train_pred_orig = sc_y.inverse_transform(train_pred)  # unscale the predictions
        y_train_orig = sc_y.inverse_transform(y_train)

        np.concatenate((train_pred_orig, y_train_orig), 1)

        np.concatenate((y_pred_orig, y_test), 1)

        results = np.concatenate((dts.iloc[:,:].values,np.concatenate((y_test, y_pred_orig), 1)),1)
        results = pd.DataFrame(data=results)
        new_order = [
            "ghi",
            "ghi10",
            "ghi90",
            "ebh",
            "dni",
            "dni10",
            "dni90",
            "dhi",
            "air_temp",
            "zenith",
            "azimuth",
            "cloud_opacity",
            "period_end",'Real Solar Power Produced', 'Predicted Solar Power'
        ]
        results.columns = new_order
        # results = results.sort_values(by=['Real Solar Power Produced'])
        pd.options.display.float_format = "{:,.2f}".format
        # results[800:820]
        print(results[1:10])
        # updating the data in csv file for future reference

        from datetime import date

        today ="predected_power-"+str(date.today())+".csv"
        #results.to_csv(today, index=False, float_format='%.2f')
        uri = "mongodb+srv://root:root@solarcluster0.lbfnszw.mongodb.net/?retryWrites=true&w=majority"

        # Create a new client and connect to the server
        client = MongoClient(uri, server_api=ServerApi('1'))

        # Send a ping to confirm a successful connection
        try:
            client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)
        mydb = client['solar']

        col = mydb["prediction"]

        for record in results.to_dict('records'):
            predicted_power = float("{:.2f}".format(record["Predicted Solar Power"]))
            filter_query = {"period_end": record["period_end"]}

            update_query = {
                "$set":
                    record
            }
            col.update_many(filter_query, update_query, upsert=True)

        results.to_csv(today, index=False, float_format='%.2f')


        sc = StandardScaler()
        pred_whole = spfnet.predict(sc.fit_transform(X))
        pred_whole_orig = sc_y.inverse_transform(pred_whole)
       # print(pred_whole_orig)

        """# New Section"""
        import datetime
        import io
        import base64
        from PIL import Image
        mydb = client["Images"]
        x = str(datetime.datetime.now())
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 2)
        plt.scatter(y_pred_orig, y_test)
        plt.xlabel('Predicted Generated Power on Test Data')
        plt.ylabel('Real Generated Power on Test Data')
        plt.title('Test Predictions vs Real Data')
        # Convert the figure to bytes and encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        figure_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.savefig('Test Predictions vs Real Data'+"<-"+x)
        # Close the figure
        plt.close()

        # Create a document with the image data
        image_document = {
            "image": figure_data,
            "title": "Test Predictions vs Real Data"+"<-"+x
        }

        # Insert the document into the MongoDB collection
        col = mydb["Test images"]  # Replace with your desired collection name
        col.insert_one(image_document)

        # plt.scatter(y_test_orig, sc_X.inverse_transform(X_test)[:,2], color='green')
        plt.subplot(1, 2, 1)
        plt.scatter(train_pred_orig, y_train_orig)
        plt.xlabel('Predicted Generated Power on Training Data')
        plt.ylabel('Real Generated Power on Training Data')
        plt.title('Training Predictions vs Real Data')
        # Convert the figure to bytes and encode as base64
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png')
        buf.seek(0)
        figure_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.savefig('Training Predictions vs Real Data' + "<-" + x)
        # Close the figure
        plt.close()

        # ... your MongoDB connection setup ...

        # Create a document with the image data
        image_document = {
            "image": figure_data,
            "title": "Training Predictions vs Real Data"+"<-"+x
        }

        # Insert the document into the MongoDB collection
        col = mydb["Train images"]  # Replace with your desired collection name
        col.insert_one(image_document)
        plt.show()
        return today
