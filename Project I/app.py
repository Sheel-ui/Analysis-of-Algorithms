import operator
from heapq import heappop, heappush, heapify
import sys


# Transforming list to dict
def transformHouseList(houseList):
    houseDictList = []
    for i in range(len(houseList)):
        houseDictList.append({
            'startDate': houseList[i][0],
            'endDate': houseList[i][1],
            'index': i + 1
        })
    return houseDictList


def strat1(n,m,houseList):
    day = 1
    paintedIndexList = []
    currentHouse = 0

    # Loop till either all days or all houses are not processed
    while day <= n and currentHouse < m:
        # If currentHouse can be painted on currentDay then paint it and increment currentHouse and currentDay
        if day>=houseList[currentHouse]['startDate'] and day <= houseList[currentHouse]['endDate']:
            paintedIndexList.append(houseList[currentHouse]['index'])
            currentHouse += 1
            day += 1
        # If currentHouse cannot be painted on day
        else:
            # Increment day if day is before the start date of currentHouse, if not increment house since it is not eligible to be painted anymore
            if day < houseList[currentHouse]['startDate']:
                day += 1
            else:
                currentHouse += 1
    return paintedIndexList

def strat2(n,m,houseList):
    heap = []
    paintedIndexList = []
    currentHouse = 0
    day = 1

    # Iterate day from 1 to n
    for day in range(1,n+1):
        isPainted = False
        # Add all the houses which become available at currentDay to min heap ordered by (descending) startDate and (ascending) endDate
        while currentHouse < m and houseList[currentHouse]['startDate'] <= day:
            heappush(heap, 
            (
                -houseList[currentHouse]['startDate'],
                houseList[currentHouse]['endDate'],
                houseList[currentHouse]['index'],
                houseList[currentHouse],
            ))
            currentHouse += 1
        # While there are houses available to be painted and no houses have been painted on currentDay
        while len(heap)>0 and isPainted == False:
            # Pop house from min heap and see if it can be painted and paint it
            house = heappop(heap)[-1]
            if day >= house['startDate'] and day <= house['endDate']:
                paintedIndexList.append(house['index'])
                isPainted = True
    return paintedIndexList

def strat3(n,m,houseList):
    heap = []
    paintedIndexList = []
    currentHouse = 0
    day = 1

    # Iterate day from 1 to n
    for day in range(1,n+1):
        isPainted = False
        # Add all the houses which become available at currentDay to min heap ordered by (ascending) endDate-StartDate and (ascending) startDate
        while currentHouse < m and houseList[currentHouse]['startDate'] <= day:
            heappush(heap, 
            (
                houseList[currentHouse]['endDate']-houseList[currentHouse]['startDate'],
                houseList[currentHouse]['startDate'],
                houseList[currentHouse]['index'],
                houseList[currentHouse]
            ))
            currentHouse += 1
        # While there are houses available to be painted and no houses have been painted on currentDay
        while len(heap)>0 and isPainted == False:
            # Pop house from min heap and see if it can be painted and paint it
            house = heappop(heap)[-1]
            if day >= house['startDate'] and day <= house['endDate']:
                paintedIndexList.append(house['index'])
                isPainted = True
    return paintedIndexList

def strat4(n,m,houseList):
    heap = []
    paintedIndexList = []
    currentHouse = 0
    day = 1
    
    # Iterate day from 1 to n
    for day in range(1,n+1):
        isPainted = False
        # Add all the houses which become available at currentDay to min heap ordered by (ascending) endDate and (ascending) startDate
        while currentHouse < m and houseList[currentHouse]['startDate'] <= day:
            heappush(heap, 
            (
                houseList[currentHouse]['endDate'],
                houseList[currentHouse]['startDate'],
                houseList[currentHouse]['index'],
                houseList[currentHouse]
            ))
            currentHouse += 1
        # While there are houses available to be painted and no houses have been painted on currentDay
        while len(heap)>0 and isPainted == False:
            # Pop house from min heap and see if it can be painted and paint it
            house = heappop(heap)[-1]
            if day >= house['startDate'] and day <= house['endDate']:
                paintedIndexList.append(house['index'])
                isPainted = True
    return paintedIndexList

def strat(n,m,houseList):
    heap = []
    paintedIndexList = []

    # Start currentDay with StartDate of first house
    currentDay = houseList[0]['startDate']

    # House 1 to m
    for house in houseList:

        # add to heap if StartDate of house is equal to currentDay
        if house['startDate'] == currentDay:
            heappush(heap,
                (
                house['endDate'],
                house['startDate'],
                house['index'],
                house
                ) 
            )
        # if StartDate is not equal to CurrentDate
        else:
            nextDay = house['startDate']
            # pop from the heap and paint the house (if valid) and increment the current day until you reach the nextDay
            while currentDay < nextDay and len(heap)>0 :
                paintedHouse = heappop(heap)[-1]
                if currentDay <= paintedHouse['endDate']:
                    if currentDay > n:
                        return paintedIndexList
                    paintedIndexList.append(paintedHouse['index'])
                    currentDay += 1
            currentDay = nextDay
            heappush(heap,
                (
                house['endDate'],
                house['startDate'],
                house['index'],
                house
                ) 
            )
    # after iterating through houses, pop the heap and paint the house (if valid) until the heap is empty
    while(len(heap)>0):
        paintedHouse = heappop(heap)[-1]
        if currentDay >= paintedHouse['startDate'] and currentDay <= paintedHouse['endDate']:
            if currentDay > n:
                return paintedIndexList
            paintedIndexList.append(paintedHouse['index'])
            currentDay += 1
    return paintedIndexList


if __name__ == "__main__":
    n_m = input()

    # STDIN
    houseList = []
    n,m = int(n_m.split(' ')[0]),int(n_m.split(' ')[1])

    for i in range(m):
        house = input()
        houseList.append([int(house.split(' ')[0]),int(house.split(' ')[1])])

    # Transform user input to supported format
    houseList = transformHouseList(houseList)


    # Invoking Strategies
    if sys.argv[1] == "strat1":
        paintedIndexList = strat1(n,m,houseList)
    elif sys.argv[1] == "strat2":
        paintedIndexList = strat2(n,m,houseList)
    elif sys.argv[1] == "strat3":
        paintedIndexList = strat3(n,m,houseList)
    elif sys.argv[1] == "strat4":
        paintedIndexList = strat4(n,m,houseList)
    elif sys.argv[1] == "stratop":
        paintedIndexList = strat(n,m,houseList)

    print(" ".join(str(item) for item in paintedIndexList))