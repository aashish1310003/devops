import pymongo
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
class Add_to_model:
    import pandas as pd
    import pymongo
    def check(self):
        import datetime
        x = datetime.datetime.now()
        print(type(x))
        print(x.strftime("%w"))
        return x.strftime("%w")

    def date_reached(self):
        uri = "mongodb+srv://root:root@solarcluster0.lbfnszw.mongodb.net/?retryWrites=true&w=majority"

        # Create a new client and connect to the server
        client = MongoClient(uri, server_api=ServerApi('1'))

        # Send a ping to confirm a successful connection
        try:
            client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)
        #client1 = pymongo.MongoClient('mongodb://127.0.0.1:27017/')

        mydb = client['test']
        collection = mydb["weather-dataset"]

        cursor = list(collection.find()) # Retrieve data from the collection
        df_sheet1 = pd.DataFrame(list(cursor))  # Convert cursor data to a list and then to a DataFrame

        collection1 = mydb["solar-dataset"]

        cursor1 = collection1.find()  # Retrieve data from the collection
        df_sheet2 = pd.DataFrame(list(cursor1))

        db = client['solar']
        # db.create_collection("weather-dataset")
        # db.create_collection("solar-dataset")
        col = db["weather-dataset"]
        col1 = db["solar-dataset"]
        try:
            for record in df_sheet1.to_dict('records'):
                filter_query = {"period_end": record["period_end"]}  # Replace "key_field" with the actual field name
                update_query = {"$set": record}

                col.update_many(filter_query, update_query, upsert=True)

            for record in df_sheet2.to_dict('records'):
                filter_query = {"period_end": record["period_end"]}  # Replace "key_field" with the actual field name
                update_query = {"$set": record}

                col1.update_many(filter_query, update_query, upsert=True)
        except Exception as e:
            print("An error occurred:", str(e))
            return 0
        return 1

    # client = pymongo.MongoClient('mongodb://127.0.0.1:27017/')
    #
    # mydb = client['solar']
    # collection = mydb["weather-dataset"]
    #
    # cursor = collection.find()  # Retrieve data from the collection
    # df_sheet1 = pd.DataFrame(list(cursor))  # Convert cursor data to a list and then to a DataFrame
    #
    # collection1 = mydb["solar-dataset"]
    # print("came inside")
    # cursor1 = collection1.find()  # Retrieve data from the collection
    # df_sheet2 = pd.DataFrame(list(cursor1))

    print(check(1))

    if ((check(1) == '1')):
      if(date_reached(1)):
        uri = "mongodb+srv://root:root@solarcluster0.lbfnszw.mongodb.net/?retryWrites=true&w=majority"

        # Create a neec-user
        # w client and connect to the server
        client = MongoClient(uri, server_api=ServerApi('1'))

        # Send a ping to confirm a successful connection
        try:
            client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)
        mydb = client['test']
        mydb.drop_collection("weather-dataset")
        mydb.drop_collection("solar-dataset")







