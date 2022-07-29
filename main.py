import json
from pprint import pprint
from xml.etree.ElementTree import SubElement
from click import option
import requests
import sys
import re
import urllib.parse
from config import CONSUMER_KEY, REDIRECT_URI, JSON_PATH
from td.client import TDClient
from datetime import datetime
from td.option_chain import OptionChain

#AUTHOR ALLEN CHEN
#CREATED FOR EVOLVING FINANCE INSTITUTE
def optionInquiry(symbol, expDate, incVol):
    #NEW SESSION
    tdClient = TDClient(client_id = CONSUMER_KEY, redirect_uri = REDIRECT_URI, credentials_path = JSON_PATH)
    tdClient.login()

    


    opt_chain = {
        'symbol': symbol,
        'contractType': 'ALL',
        'strikeCount': 1000,
        'includeQuotes': True,
        'optionType': 'ALL',
        'fromDate': expDate,
        'toDate': expDate,
        'range':'ALL',
    }
    optionChain = tdClient.get_options_chain(option_chain=opt_chain)
    # pprint(optionChain)
    
    
 
    for key,value in optionChain.items():
        # print(key, ':', value, '\n')
        if(key.__eq__('status')):
            print('\n' + key +': ' + value)
            
        if(key.__eq__('callExpDateMap')):
            
            for optionExpStr, options in value.items():
                #since optionExpStr is only one then

                for strike, optionsInfo in options.items():
                    #for each strike and its options info, we will take the strike and then find the open interest
                    #OPTIONS INFO IS A LIST OF DICTS
                    for info in optionsInfo:
                        last = 0 
                        openInterest = 0 
                        volume = 0
                        for name, numeric in info.items():
                            if(name.__eq__('totalVolume')):
                                volume = numeric
                            if(name.__eq__('last')):
                                last = numeric * 100
                                # print(last,'\n')
                            if(name.__eq__('openInterest')):
                                openInterest = numeric
                                # print(openInterest, '\n')
                        
                        # print(strike, ':',last*openInterest, '\n')
                        heatmap.update({strike: last*openInterest})
                        heatmap1.update({strike: last*(openInterest + volume)})
        if(key.__eq__('putExpDateMap')):

            for optionExpStr, options in value.items():
                #since optionExpStr is only one then

                for strike, optionsInfo in options.items():
                    #for each strike and its options info, we will take the strike and then find the open interest
                    #OPTIONS INFO IS A LIST OF DICTS
                    for info in optionsInfo:
                        last = 0 
                        openInterest = 0 
                        volume = 0
                        for name, numeric in info.items():
                            # print(name)
                            if(name.__eq__('totalVolume')):
                                volume = numeric
                                # print(volume)
                            if(name.__eq__('last')):
                                last = numeric * 100
                                # print(last,'\n')
                            if(name.__eq__('openInterest')):
                                openInterest = numeric
                                # print(openInterest, '\n')
                        
                        # print(strike, ':',last*openInterest, '\n')
                        putHeatMap.update({strike: last*openInterest})
                        #MAX
                        putHeatMap1.update({strike: last*(openInterest + volume)})
    
    
if __name__ == "__main__":
 
    if(len(sys.argv) < 3):
        print("Not enough arguments")
        sys.exit("Format is py main.py (Symbol) (Exp Date yyyy-MM-dd)")
    
    
    #FORMAT CHECKING SYS
    if(len(sys.argv) >= 3):
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
        spy500 = ["AAPL", "MSFT",
            "AMZN",
            "TSLA",
            "BRK.B",
            "UNH",
            "V",
            "JNJ",
            "META",
            "NVDA",
            "XOM",
            "PG",
            "WMT",
            "JPM",
            "MA",
            "LLY",
            "HD",
            "PFE",
            "CVX",
            "KO",
            "ABBV",
            "BAC",
            "PEP",
            "MRK",
            "COST",
            "TMO",
            "AVGO",
            "DHR",
            "ORCL",
            "ABT",
            "CMCSA",
            "MCD",
            "ACN",
            "VZ",
            "DIS",
            "CSCO",
            "ADBE",
            "CRM",
            "QCOM",
            "TMUS",
            "NKE",
            "WFC",
            "INTC",
            "UPS",
            "BMY",
            "NEE",
            "PM",
            "TXN",
            "LIN",
            "MS",
            "AMD",
            "UNP",
            "RTX",
            "AMGN",
            "T",
            "CVS",
            "HON",
            "MDT",
            "SPGI",
            "LOW",
            "AMT",
            "COP",
            "IBM",
            "SCHW",
            "INTU",
            "ELV",
            "AXP",
            "GS",
            "LMT",
            "C",
            "DE",
            "CAT",
            "NFLX",
            "PLD",
            "BLK",
            "BA",
            "SBUX",
            "EL",
            "ADP",
            "PYPL",
            "CI",
            "MDLZ",
            "AMAT",
            "NOW",
            "ADI",
            "ZTS",
            "DUK",
            "CHTR",
            "MMM",
            "MO",
            "CB",
            "GE",
            "SO",
            "MMC",
            "ISRG",
            "SYK",
            "GILD",
            "CCI",
            "VRTX",
            "CME",
            "BKNG",
            "GOOG",
            "GOOGL",
            "TJX",
            "USB",
            "TGT",
            "BDX",
            "NOC",
            "PNC",
            "CSX",
            "FISV",
            "SHW",
            "IWM",
            "MU",
            "CL",
            "DIA",
            "SPY",
            "QQQ",
            "MRNA"]
        
        
        heatmap = {}
        heatmap1 = {}
        putHeatMap = {}
        putHeatMap1 = {}
        
        if(symbol.__eq__('PC')):
            totalCalls = 0
            totalPuts = 0 
            for ticker in spy500:
                optionInquiry(ticker, expDate, incVol)
                if(incVol == True):
                    top10 = sorted(heatmap1,key=heatmap1.get, reverse = True)[:10]
                    for x in top10:
                        totalCalls+= float(heatmap1[x])    
                          
                    top10Put = sorted(putHeatMap1,key=putHeatMap1.get, reverse = True)[:10]
                    for x in top10Put:
                        totalPuts += float(putHeatMap1[x])
                else:
                    top10 = sorted(heatmap,key=heatmap.get, reverse = True)[:10]
                    for x in top10:
                       totalCalls+= float(heatmap[x])    
           
                    top10Put = sorted(putHeatMap,key=putHeatMap.get, reverse = True)[:10]
                    for x in top10Put:
                       totalPuts += float(putHeatMap[x])
                print(heatmap)
                heatmap = {}
                heatmap1 = {}
                putHeatMap = {}
                putHeatMap1 = {}
            print("PC is: ", totalPuts/totalCalls)
                
        else:
            optionInquiry(symbol, expDate, incVol)
            
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

            

    


                            
                        
                        
            
        # if(key.__eq__('putExpDateMap')):
        #     print(value, "--------------------\n")
        

    