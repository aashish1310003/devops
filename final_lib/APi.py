import csv
import requests
my_weather = {}
my_power = {}

class Api:

    def obtaining_api(self):

        apikey_solcast ="0WvZeaEuERLydgQZ9sKuR-Xu8uiC-EA4"
         # "hnU9uUoxOr3t9miLEwC_Ap9q8OYImXtK"
        # nithish#hnU9uUoxOr3t9miLEwC_Ap9q8OYImXtK
        # dhinesh#kxASosF5BNyVd-7CxDSTRYuCPLderLoV
        # aashish#0WvZeaEuERLydgQZ9sKuR-Xu8uiC-EA4
        #############################################################################
        #####LOCATION
        endpoint = "http://dataservice.accuweather.com/locations/v1/cities/IN/search"
        apikey = "mUZP780gNd6rN6WttTdSGQXqbuLl5Wux"
        city = "erode"
        params = {
            "apikey": apikey,
            "q": city,

            "details": "true"
        }
        response = requests.request("GET", endpoint, params=params)
        myjson = response.json()
        # print(myjson)
        lat = myjson[0]["GeoPosition"]["Latitude"]
        lon = myjson[0]["GeoPosition"]["Longitude"]
        print(lat, lon)

        #############################################################################
        # OPTANING API FOR WEATHER FORCAST
        url = "https://solcast.p.rapidapi.com/radiation/forecasts"

        querystring = {"api_key": apikey_solcast, "latitude": lat, "longitude": lon, "format": "json"}

        headers = {
            "X-RapidAPI-Key": "ab3dbf3a9bmsh4b3c3f5a60c89d5p121ce2jsnac7356643959",
            "X-RapidAPI-Host": "solcast.p.rapidapi.com"
        }

        mylist = []
        response = requests.request("GET", url, headers=headers, params=querystring)

        print(response.text)
        mydata = response.json()
        global my_weather
        my_weather= mydata

        csv_head = ['ghi', 'ghi10', 'ghi90', 'ebh', 'dni', 'dni10', 'dni90', 'dhi', 'air_temp', 'zenith', 'azimuth',
                    'cloud_opucity', 'period_end']
        name_we = "weather_output_data" + mydata["forecasts"][0]['period_end'].replace(':', '-').split('.')[0]
        for i in mydata["forecasts"]:
            lis = [i['ghi'], i['ghi90'], i['ghi10'], i['ebh'], i['dni'], i['dni10'], i['dni90'], i['dhi'],
                   i['air_temp'], i['zenith'], i['azimuth'], i['cloud_opacity'], i['period_end']]
            mylist.append(lis)
        # with open("{}.csv".format(name_we), 'w+', encoding="UTF8", newline='') as F:
        #     writer = csv.writer(F)
        #     writer.writerow(csv_head)
        #     writer.writerows(mylist)
        #F.close()
        print("done")




        ##############################################################################
        # OPTAINING API FOR SOLAR

        url = "https://solcast.p.rapidapi.com/pv_power/forecasts"

        querystring = {"api_key": apikey_solcast, "capacity": '5', "latitude": lat, "longitude": lon, "tilt": "23",
                       "format": "json"}

        headers = {
            "X-RapidAPI-Key": "ab3dbf3a9bmsh4b3c3f5a60c89d5p121ce2jsnac7356643959",
            "X-RapidAPI-Host": "solcast.p.rapidapi.com"
        }

        response_power = requests.request("GET", url, headers=headers, params=querystring)

#        print(response_power.text)

        mypower = response_power.json()
        global my_power
        my_power = mypower
        power_list = []
        csv_head_power = ['pv_estimate', 'period_end', 'period']
        for j in mypower["forecasts"]:
            lis_power = [j['pv_estimate'], j['period_end'], j['period']]
            power_list.append(lis_power)
        name = "solarpower_output_data" + mypower["forecasts"][0]['period_end'].replace(':', '-').split('.')[0]
        # with open("{}.csv".format(name), 'w', encoding="UTF8", newline='') as Fa:
        #     writers = csv.writer(Fa)
        #     writers.writerow(csv_head_power)
        #     writers.writerows(power_list)

        #Fa.close()
        print("done")
        return [name, name_we]
    def wea_data(self):
        return my_weather
    def sol_data(self):
        return my_power

