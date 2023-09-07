import requests
import json
import pyttsx3
import speech_recognition as sr
import re
import threading
import time


API_KEY = "tpp4MuRNoovx"
PROJECT_TOKEN1 = "teHgN4Zrvc3e"
PROJECT_TOKEN2 = "tAO1qvgOyeT6"
RUN_TOKEN1 = "t03p6K4TCcje"
RUN_TOKEN2 = "tLXxfJL_kygB"

class Data:
    def __init__(self, api_key, project_token1, project_token2 ):
        self.api_key = api_key
        self.project_token1 = project_token1
        self.project_token2 = project_token2
        self.params = {
            "api_key":self.api_key
        }
        self.data, self.data2 =self.get_data()
    
    def get_data(self):
        response  = requests.get(f'https://www.parsehub.com/api/v2/projects/{self.project_token1}/last_ready_run/data', params=self.params)
        response2 = requests.get(f'https://www.parsehub.com/api/v2/projects/{self.project_token2}/last_ready_run/data', params=self.params)
        data = json.loads(response.text)
        data2 = json.loads(response2.text)
        return data, data2
    
    def get_total_population(self):
        data = self.data['value']
        return data

    def get_totals(self, string_1):
        data = self.data['totals']

        for content in data:
            if content['name'] == string_1:
                return content['value']
        return "None" 
    
    def get_country_data(self, country):
        data = self.data2['country']

        for content in data:
            if content['name'].lower() == country.lower():
                return content
        return "None"

    def get_list_countries(self):
        countries = []
        for country in self.data2['country']:
            countries.append(country['name'].lower())
        
        return countries

    def update_data(self):
        response = requests.post(f'https://www.parsehub.com/api/v2/projects/{self.project_token1}/run', params=self.params)
        response2 = requests.post(f'https://www.parsehub.com/api/v2/projects/{self.project_token2}/run', params=self.params)

        def pull():
            time.sleep(0.1)
            old_data = self.data
            old_data2 = self.data2
            while True:
                new_data, new_data2 = self.get_data()
                if new_data != old_data:
                    self.data=new_data
                    self.data2=new_data2
                    print("Data updated")
                    break
                time.sleep(5)


        t=threading.Thread(target=pull)
        t.start()

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio)
        except Exception as e:
            print("Exception:", str(e))
    return said.lower()

def main():
    print("Started Program")
    data = Data(API_KEY, PROJECT_TOKEN1, PROJECT_TOKEN2)
    END_PHRASE ="stop"
    country_list = data.get_list_countries()

    TOTAL_PATTERNS = {
        re.compile("[\w\s]+ total [\w\s]+ population"):data.get_total_population(),
        re.compile("[\w\s]+ total population"):data.get_total_population(),
        re.compile("[\w\s]+ born today"):data.get_totals(string_1="Births today"),
        re.compile("[\w\s]+ born [\w\s] + today"):data.get_totals(string_1="Births today"),
        re.compile("[\w\s]+ born this year"):data.get_totals(string_1="Births this year"),
        re.compile("[\w\s]+ passed on today"):data.get_totals(string_1="Deaths today"),
        re.compile("[\w\s]+ passed on this year"):data.get_totals(string_1="Deaths this year"),
        re.compile("[\w\s]+ population growth today"):data.get_totals(string_1="Population Growth today"),
        re.compile("[\w\s]+ population growth this year"):data.get_totals(string_1="Population Growth this year"),
        re.compile("[\w\s]+ born [\w\s] + this year"):data.get_totals(string_1="Births this year"),
        re.compile("[\w\s]+ passed on [\w\s] + today"):data.get_totals(string_1="Deaths today"),
        re.compile("[\w\s]+ passed on [\w\s] + this year"):data.get_totals(string_1="Deaths this year"),
        re.compile("[\w\s]+ population growth [\w\s] + today"):data.get_totals(string_1="Population Growth today"),
        re.compile("[\w\s]+ population growth [\w\s] + this year"):data.get_totals(string_1="Population Growth this year")
    }

    COUNTRY_PATTERNS = {
        re.compile("[\w\s] + total population [\w\s]+"): lambda country: data.get_country_data(country)['population'] 
    }

    UPDATE_PATTERN = "update"

    while True:
        print("Listening...")
        text = get_audio()
        print(text)
        result=None

        for pattern, func in COUNTRY_PATTERNS.items():
            if pattern.match(text):
                words = set(text.split(" "))
                for country in country_list:
                    if country in words:
                        result = func(country)
                break

        for pattern, func in TOTAL_PATTERNS.items():
            if pattern.match(text):
                result = func
                break
        
        if text ==UPDATE_PATTERN:
            result = "Data is being updated. This may take a moment!"
            data.update_data()

        if result:
            speak(result)

        if text.find(END_PHRASE) != -1:
            speak("Goodbye")
            break

main()


#####################################################################
# Test code
#s = "Births today"
#print(s + ": " + data.get_totals(string_1=s))
#s = "Births this year"
#print(s + ': ' + data.get_totals(string_1=s))
#s = "Deaths today"
#print(s + ': ' + data.get_totals(string_1=s))
#s = "Deaths this year"
#print(s + ': ' + data.get_totals(string_1=s))
#s = "Population Growth today"
#print(s + ': ' + data.get_totals(string_1=s))
#s = "Population Growth this year"
#print(s + ': ' + data.get_totals(string_1=s))
#country = "China"
#print(data.get_country_data("China"))
#speak("Hello")
#print(data.get_audio())
#print(data.get_list_countries())


        