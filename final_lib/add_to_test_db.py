import pandas as pd
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
class Add_to_Test_Db:



    def add_to_db(self, lis):
        df_sheet1 = pd.DataFrame(lis[0]['forecasts'])  # Convert to DataFrame
        df_sheet2 = pd.DataFrame(lis[1]['forecasts'])   #client = pymongo.MongoClient('mongodb://127.0.0.1:27017/')
        uri = "mongodb+srv://root:root@solarcluster0.lbfnszw.mongodb.net/?retryWrites=true&w=majority"

        # Create a new client and connect to the server
        client = MongoClient(uri, server_api=ServerApi('1'))

        # Send a ping to confirm a successful connection
        try:
            client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB! in test class")
        except Exception as e:
            print(e)

        mydb = client['test']
        collection = mydb["weather-dataset"]

        merged_df = pd.merge(df_sheet1, df_sheet2, on='period_end', how='inner')
        matching_period_end = merged_df['period_end']
        wea_dict = df_sheet1[df_sheet1['period_end'].isin(matching_period_end)]
        sol_dict = df_sheet2[df_sheet2['period_end'].isin(matching_period_end)]

        try:
                # Assuming you have imported necessary modules and initialized your MongoDB collection as 'collection'
                for record in wea_dict.to_dict('records'):
                    filter_query = {"period_end": record["period_end"]}  # Replace "key_field" with the actual field name
                    update_query = {"$set": record}

                    collection.update_many(filter_query, update_query, upsert=True)


                collection1 = mydb["solar-dataset"]
                for record in sol_dict.to_dict('records'):
                    filter_query = {"period_end": record["period_end"]}  # Replace "key_field" with the actual field name
                    update_query = {"$set": record}

                    collection1.update_many(filter_query, update_query, upsert=True)

        except Exception as e:
            print("An error occurred:", str(e))
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$      Test Data is not added to the database        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            return 0
        print("\nhi data added to test Database\n")
        return 1
