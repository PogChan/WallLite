import json
from optparse import Option
from posixpath import expandvars
from pprint import pprint
from xml.etree.ElementInclude import include
from xml.etree.ElementTree import SubElement
from click import option
import time
import sys
import re
import urllib.parse
from itertools import tee
from config import CONSUMER_KEY, REDIRECT_URI, JSON_PATH
from td.client import TDClient
from datetime import datetime
from td.option_chain import OptionChain
from tickers import sectors, spy500

#AUTHOR ALLEN CHEN
#CREATED FOR EVOLVING FINANCE INSTITUTE
def optionInquiry(symbol, expDate, incVol, optionsType):
    #NEW SESSION

    time.sleep(1)
    optionChain = None
    try:
        optionQuery = OptionChain(symbol=symbol,strike_count=24, include_quotes=True, from_date=expDate, to_date=expDate, opt_range= optionsType)
        optionChain = tdClient.get_options_chain(optionQuery)
    except:
        print("ERROR: ", symbol, " ", expDate, " ", optionsType)

    if(optionChain is None):
        return (0,0)
    totalOICall = 0
    totalOIPut = 0
    print(symbol)
    status = optionChain["status"]

    print('\n' + "STATUS" +': ' +status )
    if(status == "FAILED"):
        return (0,0)

    priceOfSymbol = optionChain['underlying']['last']
    print("\nLAST: ", priceOfSymbol,"\n")
    if(priceOfSymbol is None):
        return (0,0)

    callMap = optionChain['callExpDateMap']
    putMap = optionChain['putExpDateMap']
    #if values are empty then we will return 0,0
    if(len(callMap.values()) == 0 or len(putMap.values()) == 0):
        return (0,0)

    callExpValue = next(iter(callMap.values()))
    putExpValue = next(iter(putMap.values()))

    for strike, optionsInfo in callExpValue.items():
        #for each strike and its options info, we will take the strike and then find the open interest
        #OPTIONS INFO IS A LIST OF DICTS
        volume = optionsInfo[0]['totalVolume']
        last = optionsInfo[0]['last'] * 100
        openInterest = optionsInfo[0]['openInterest']
        if((priceOfSymbol) >= optionsInfo[0]['strikePrice'] ):
            totalOICall += openInterest

        heatmap.update({strike: last*openInterest})
        heatmap1.update({strike: openInterest})
    for strike, optionsInfo in putExpValue.items():
        #for each strike and its options info, we will take the strike and then find the open interest
        #OPTIONS INFO IS A LIST OF DICTS

        volume = optionsInfo[0]['totalVolume']
        last = optionsInfo[0]['last'] * 100
        openInterest = optionsInfo[0]['openInterest']

        if((priceOfSymbol) <= optionsInfo[0]['strikePrice'] ):
            totalOIPut += openInterest
        putHeatMap.update({strike: last*openInterest})
        putHeatMap1.update({strike: openInterest})

    #CALLS - PUTS
    #NEGATIVE = More Puts
    #POSITIVE = More Calls

    # imbalanceList = [(strike, heatmap1[strike] - putHeatMap1[strike]) for strike in heatmap1.keys() & putHeatMap1.keys()]
    # print(imbalanceList)

    totalOICall = sum(heatmap.values())
    totalOIPut = sum(putHeatMap.values())
    print('totalOICall: ', totalOICall, 'totalOIPut: ', totalOIPut)
    return (totalOICall, totalOIPut)



if __name__ == "__main__":

    if(len(sys.argv) < 3):
        print("Not enough arguments")
        sys.exit("Format is py main.py (Symbol) (Exp Date yyyy-MM-dd)")


    #FORMAT CHECKING SYS
    if(len(sys.argv) >= 3):

        tdClient = TDClient(client_id = CONSUMER_KEY, redirect_uri = REDIRECT_URI, credentials_path = JSON_PATH)
        tdClient.login()
        f= open('result.txt', 'w')

        symbol = sys.argv[1]
        expDate = sys.argv[2]
        incVol = False

        if(len(sys.argv) == 4):
            temp = sys.argv[3].upper()
            if(temp.__eq__('TRUE')):
                incVol = True


        symbol = symbol.upper()

        if(re.match('(\d{4})[/.-](\d{2})[/.-](\d{2})$', expDate) is None):
            sys.exit("Invalid Exp Date (yyyy-MM-dd)")
        print('======OPTIONS HEATMAP======\n')
        print('TICKER: '+symbol +' |', 'EXP DATE: ' + expDate)

        heatmap = {}
        heatmap1 = {}
        putHeatMap = {}
        putHeatMap1 = {}


        if(symbol.__eq__('PC')):
            totalCalls = 0
            totalPuts = 0
            individualPCs = {}
            individualOIPCs = {}
            for ticker in spy500:

                x = optionInquiry(ticker, expDate, incVol, "ALL")
                if(not x):
                    print('nothing', x)
                    continue
                (callOI, putOI) = x

                if(callOI == 0):
                    callOI = 1
                elif(putOI == 0):
                    putOI = 1


                putHeatMap = dict(filter(lambda elem: elem[1] >= 1000000, putHeatMap.items()))
                heatmap = dict(filter(lambda elem: elem[1] >= 1000000, heatmap.items()))
                if(len(putHeatMap) == 0 and len(heatmap) == 0):
                    print(ticker)
                    continue

                individualOIPCs[ticker] = putOI/callOI


                if(incVol == True):
                    top10 = sorted(heatmap1,key=heatmap1.get, reverse = True)[:10]
                    for x in top10:
                        totalCalls+= float(heatmap1[x])


                    top10Put = sorted(putHeatMap1,key=putHeatMap1.get, reverse = True)[:10]
                    for x in top10Put:
                        totalPuts += float(putHeatMap1[x])
                else:
                    tempC = 0
                    tempP = 0
                    top10 = sorted(heatmap,key=heatmap.get, reverse = True)[:10]
                    for x in top10:
                       tempC += float(heatmap[x])
                       totalCalls+= float(heatmap[x])

                    top10Put = sorted(putHeatMap,key=putHeatMap.get, reverse = True)[:10]
                    for x in top10Put:
                       totalPuts += float(putHeatMap[x])
                       tempP += float(putHeatMap[x])

                    if(tempC == 0):
                        continue
                    individualPCs[ticker] = tempP/tempC

                    print(ticker + ' | ' + str(individualPCs[ticker]) + ' | ' + str(individualOIPCs[ticker]))


                heatmap = {}
                heatmap1 = {}
                putHeatMap = {}
                putHeatMap1 = {}


            print("========================\n\nTOP 50 PUT IMBALANCE")
            individualPut = sorted(individualPCs.items(), key=lambda item: item[1], reverse= True)
            RRGPut = ''
            for i in range(0,len(individualPut)):
                print(str(individualPut[i]) + ' | ' + str(individualOIPCs[individualPut[i][0]]))

                RRGPut += individualPut[i][0]
                if(i<49):
                    RRGPut+=","
            print(RRGPut)
            print("\nTOP 50 CALL IMBALANCE")
            RRGCall = ''
            individualCall = sorted(individualPCs.items(), key=lambda item: item[1])
            for i in range(0,len(individualCall)):

                print(str(individualCall[i]) + ' | ' + str(individualOIPCs[individualCall[i][0]]))

                RRGCall += individualCall[i][0]
                if(i<49):
                    RRGCall+=","
            print(RRGCall)


            print("PC is: ", totalPuts/totalCalls)
        elif(symbol.__eq__('PCOI') or symbol in sectors):
            totalCalls = 0
            totalPuts = 0
            individualOIPCs = {}
            tickersToSearch = spy500
            if(symbol in sectors):
               tickersToSearch = sectors[symbol]

            for ticker in tickersToSearch:
                x= optionInquiry(ticker, expDate, incVol, "ALL")
                if(not x):
                    print('nothing', x)
                    continue
                (callOI, putOI) = x
                if(tickersToSearch == spy500):
                    if(callOI < 1000):
                        print('CALL OI less than 1000')
                        continue
                    elif(putOI < 1000):
                        print('PUT OI less than 1000')
                        continue
                else:
                    if(callOI == 0):
                        print('CALL OI is 0')
                        callOI = 1
                    if(putOI == 0):
                        print('PUT OI is 0')
                        putOI = 1

                individualOIPCs[ticker] = putOI/(callOI+putOI)

                print(ticker + ' | ' + str(individualOIPCs[ticker]))
                heatmap = {}
                heatmap1 = {}
                putHeatMap = {}
                putHeatMap1 = {}


            sys.stdout = f
            print(sys.argv[1], " ", sys.argv[2])
            print("========================\n\nTOP 50 PUT OI IMBALANCE")
            individualPut = sorted(individualOIPCs.items(), key=lambda item: item[1], reverse= True)
            RRGPut = ''
            for i in range(0,len(individualPut)):
                print(individualPut[i])
                RRGPut += individualPut[i][0]
                if(i<49):
                    RRGPut+=","
                else:
                    break
            print(RRGPut)
            print("\nTOP 50 CALL OI IMBALANCE")
            RRGCall = ''
            individualCall = sorted(individualOIPCs.items(), key=lambda item: item[1])
            for i in range(0,len(individualCall)):

                print(individualCall[i])
                RRGCall += individualCall[i][0]
                if(i<49):
                    RRGCall+=","
                else:
                    break
            print(RRGCall)
        else:
            x = optionInquiry(symbol, expDate, incVol, 'ALL')

            if(x):
                (callOI, putOI) =x
                print("OI PC: "+ str(putOI/callOI))
                if(incVol == True):
                    top10 = sorted(heatmap1,key=heatmap1.get, reverse = True)[:10]
                    print('\n----CALL SIDE (Volume ADDED)----\n')
                    for x in top10:
                        print('Strike: ', x, '---> $', '{:,.2f}'.format(heatmap1[x]))


                    top10Put = sorted(putHeatMap1,key=putHeatMap1.get, reverse = True)[:10]
                    print('\n----PUT SIDE (Volume ADDED)----\n')
                    for x in top10Put:
                        print('Strike:', x, '---> $', '{:,.2f}'.format(putHeatMap1[x]))
                else:
                    top10 = sorted(heatmap,key=heatmap.get, reverse = True)[:10]
                    print('\n----CALL SIDE----\n')
                    for x in top10:
                        print('Strike: ', x, '---> $', '{:,.2f}'.format(heatmap[x]))


                    top10Put = sorted(putHeatMap,key=putHeatMap.get, reverse = True)[:10]
                    print('\n----PUT SIDE----\n')
                    for x in top10Put:
                        print('Strike:', x, '---> $', '{:,.2f}'.format(putHeatMap[x]))
    sys.stdout = sys.__stdout__
    f.close()
