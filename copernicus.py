# It seems that if you pick a date that is far too back the request gets denied,
# certain requests also produce 500 internal server error on their server side.

# from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import sentinelsat
import glob
import os
from datetime import date
import zipfile
import operator

# connect to the API
api = sentinelsat.SentinelAPI('kfilyk', 'wet_garbage')

tiles = ['10UDV']
#  ghana : 30VNL, 30NUL, 30NTL
# vancouver/victoria 10UDV, 10UEV, 10UDU, 10UEU
# taiwan: 51RUH, 51RTH, 51RUJ, 51RTJ
# vietnam (hanoi): 48QXH, 48QYJ

query_kwargs = {
    'platformname': 'Sentinel-2',
    'producttype': 'S2MSI2A',  # sentinel 2 multispectrum instrument level 2a
    'date': ('20210801', '20210813')}  # < - this scans 30 days into the past

# For each area we we need to request grab N number of least cloudy days for it
for tile in tiles:
    print("TILE: ", tile)
    kw = query_kwargs.copy()
    kw['filename'] = f'*_T{tile}_*'
    pp = api.query(**kw) # Ordered dictionary
    # print("LEN PP: ", len(pp))
    # print("TYPE PP: ", type(pp))

    tilesToDownload = {} # Dictionary of (tileId, cloud cover percentage)
    numTilesToSave = 3

    for p in pp:
        # print(pp[p]['cloudcoverpercentage'])
        if len(tilesToDownload) < numTilesToSave:
            tilesToDownload.update({p: pp[p]['cloudcoverpercentage']})
        else:
            # Replace if current cloud coverage is lower than highest in the list
            currentMaxCCId = max(tilesToDownload, key=tilesToDownload.get)
            if pp[p]['cloudcoverpercentage'] < tilesToDownload[currentMaxCCId]:
                # print (pp[p]['cloudcoverpercentage'], " smaller than ", tilesToDownload[currentMaxCCId])
                tilesToDownload.pop(currentMaxCCId)
                tilesToDownload.update({p: pp[p]['cloudcoverpercentage']})

    tilesSortedByCC = sorted(tilesToDownload.items(), key=operator.itemgetter(1)) #  Sort dictionary from lowest to highest
    print("Tile (", tile,")'s least cloud tiles: ", tilesSortedByCC)

    for sortedTile in tilesSortedByCC:
        if not os.path.exists("./maps/" + tile + "/"+pp[sortedTile[0]]['title'] + ".SAFE"):
            print(pp[sortedTile[0]]['title'] + " doesn't exist yet. Downloading...")
            try:
                api.download(sortedTile[0], "./maps/"+tile)
            except:
                pass
        else:
            print(pp[sortedTile[0]]['title'] + " already exists.")

    # extract all previously downloaded and unzipped
    # for file in os.listdir('./maps/'+tile):
    #     print(file)
    #     if(file.endswith(".zip")):
    #         print("UNZIPPING %s..." %(file))
    #         with zipfile.ZipFile('./maps/'+tile+'/'+file, 'r') as zip_ref:
    #             zip_ref.extractall('./maps/'+tile)
    #         #os.remove(file)

    # # we have top 3 cloudless days for each tile in 'tiles' list: download to its own folder
    # if not os.path.exists('./maps/'+tile+"/"+pp[least_clouds_1_id]['title']+'.SAFE'):
    #     print(pp[least_clouds_1_id]['title'] +
    #           " doesn't exist yet. Downloading...")
    #     try:
    #         # sensat automatically skips downloading previously downloaded files??
    #         api.download(least_clouds_1_id, './maps/'+tile)
    #     except:
    #         # if connection gets interrupted
    #         pass
    # else:
    #     print(pp[least_clouds_1_id]['title']+" already exists.")

    # if not os.path.exists('./maps/'+tile+"/"+pp[least_clouds_2_id]['title']+'.SAFE'):
    #     print(pp[least_clouds_2_id]['title'] +
    #           " doesn't exist yet. Downloading...")
    #     try:
    #         api.download(least_clouds_2_id, './maps/'+tile)
    #     except:
    #         # if connection gets interrupted
    #         pass
    # else:
    #     print(pp[least_clouds_2_id]['title']+" already exists.")

    # print('./maps/'+tile+"/"+pp[least_clouds_3_id]['title']+'.SAFE')
    # if not os.path.exists('./maps/'+tile+"/"+pp[least_clouds_3_id]['title']+'.SAFE'):
    #     print(pp[least_clouds_3_id]['title'] +
    #           " doesn't exist yet. Downloading...")
    #     try:
    #         api.download(least_clouds_3_id, './maps/'+tile)
    #     except:
    #         pass
    # else:
    #     print(pp[least_clouds_3_id]['title']+" already exists.")


    # for file in os.listdir('./maps/'+tile):
    #     print(file)
    #     if(file.endswith(".zip")):
    #         print("UNZIPPING %s..." %(file))
    #         with zipfile.ZipFile('./maps/'+tile+'/'+file, 'r') as zip_ref:
    #             zip_ref.extractall('./maps/'+tile)
            #os.remove(file)

"""
# GeoJSON FeatureCollection containing footprints and metadata of the scenes
api.to_geojson(products)

# GeoPandas GeoDataFrame with the metadata of the scenes and the footprints as geometries
api.to_geodataframe(products)

# Get basic information about the product: its title, file size, MD5 sum, date, footprint and
# its download url
api.get_product_odata(< product_id > )

# Get the product's full metadata available on the server
api.get_product_odata(< product_id > , full=True)
"""
