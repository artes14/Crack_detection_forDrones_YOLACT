from __future__ import print_function
import os, time, sys, datetime
import shutil
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import wget

#reference : https://blog.naver.com/eziya76/221340903346

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/photoslibrary",
          "https://www.googleapis.com/auth/photoslibrary.readonly",
          "https://www.googleapis.com/auth/photoslibrary.readonly.appcreateddata"]
def getauth():
    #OAuth2 authentication process
    print("OAuth2 authentication process...")

    store = file.Storage("token.json")
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets("credentials.json", SCOPES)
        creds = tools.run_flow(flow, store)
    service = build("photoslibrary", "v1", http=creds.authorize(Http()),static_discovery=False)
    return service

def set_cloud_service(startDatetime:datetime.datetime, endDatetime:datetime.datetime):
    service = getauth()
    # Search image files
    print("Search image files...")
    # dtToday = datetime.date.today()
    # dt7DaysAgo = dtToday - datetime.timedelta(days=dtDates, hours=dtHours)

    results = service.mediaItems().search(
        body={
            "filters": {
                "dateFilter": {
                    "ranges": [{
                            # "startDate": {"year": dt7DaysAgo.year, "month": dt7DaysAgo.month, "day": dt7DaysAgo.day},
                            # "startDate": {"year": dtToday.year, "month": dtToday.month, "day": dtToday.day},
                        "startDate": {"year": startDatetime.year, "month": startDatetime.month, "day": startDatetime.day},
                            # "endDate": {"year": dtToday.year, "month": dtToday.month, "day": dtToday.day}
                        "endDate": {"year": endDatetime.year, "month": endDatetime.month, "day": endDatetime.day}
                        }
                    ]
                },
                "mediaTypeFilter": { # mediaTypes: ALL_MEDIA, VIDEO, PHOTO
                    "mediaTypes": ["PHOTO"]
                }
            }
        }
    ).execute()
    # localtz=pytz.timezone('Asia/Seoul')
    items = results.get("mediaItems", [])
    # print(items)
    if not items:
        print("No media found.")
        return None
    else:
        return items

def downloadfile_fromcloud(startDatetime:datetime.datetime, endDatetime:datetime.datetime, savefolder:str="cloudImages"):
    """dtDates: get images from (int) dates ago
    savefolder: folder to save images, default->cloudImages"""
    # first clear all data inside folder
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    else:
        shutil.rmtree(savefolder)
        os.mkdir(savefolder)

    items = set_cloud_service(startDatetime, endDatetime)
    if items is None:
        return
    for index,item in enumerate(items):
        filename = item["filename"].encode("utf8")
        baseUrl = item["baseUrl"].encode("utf8")
        # print(baseUrl)
        baseUrl_str=baseUrl.decode('utf-8')+'=d' #'=d' means with metadata & full size
        #baseUrl parameter reference :  https://developers.google.com/photos/library/guides/access-media-items#base-urls
        filename_str=filename.decode('utf-8')

        # check duplication & download
        file_full_path = os.path.join(savefolder, filename_str)
        if not os.path.isfile(file_full_path):
            wget.download(baseUrl_str, savefolder + '/' + filename_str)



def full():
    #OAuth2 authentication process
    print("OAuth2 authentication process...")

    store = file.Storage("token.json")
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets("credentials.json", SCOPES)
        creds = tools.run_flow(flow, store)
    service = build("photoslibrary", "v1", http=creds.authorize(Http()),static_discovery=False)
    #reference : https://stackoverflow.com/questions/66689941/google-photos-api-new-version

    # Search image files
    print("Search image files...")
    dtToday = datetime.date.today()
    print(dtToday)
    dt7DaysAgo = dtToday - datetime.timedelta(days=10)

    results = service.mediaItems().search(
        body={
            "filters": {
                "dateFilter": {
                    "ranges": [
                        {
                            "startDate": {"year": dt7DaysAgo.year, "month": dt7DaysAgo.month, "day": dt7DaysAgo.day},
                             # "startDate": {"year": dtToday.year, "month": dtToday.month, "day": dtToday.day},
                            "endDate": {"year": dtToday.year, "month": dtToday.month, "day": dtToday.day}
                         }
                    ]
                },
                "mediaTypeFilter": {
                    "mediaTypes": ["PHOTO"]
                }
            }
        }
    ).execute()

    items = results.get("mediaItems", [])
    print(items)
    if not items:
        print("No media found.")
        quit()


    # Download & conver image files
    for index,item in enumerate(items):
        filename = item["filename"].encode("utf8")
        baseUrl = item["baseUrl"].encode("utf8")
        bmpname = filename.decode().split('.')[0] + ".bmp"

        print(baseUrl)

        baseUrl_str=baseUrl.decode('utf-8')+'=d' #'=d' means with metadata & full size
        #baseUrl parameter reference :  https://developers.google.com/photos/library/guides/access-media-items#base-urls
        filename_str=filename.decode('utf-8')

        # check duplication & download
        file_full_path = os.path.join("cloudImages", bmpname)
        if not os.path.isfile(file_full_path):
            wget.download(baseUrl_str, "images/" + filename_str)

# downloadfile_fromcloud(datetime.datetime(2022,7,21), datetime.datetime(2022,7,22))