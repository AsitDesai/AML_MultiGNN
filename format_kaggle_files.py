import numpy as np
import datatable as dt
from datetime import datetime
from datatable import f, join, sort
import sys
import os

# Check if the input path has been provided
n = len(sys.argv)
if n == 1:
    print("No input path")
    sys.exit()

# Set input and output file paths
inPath = sys.argv[1]
outPath = os.path.dirname(inPath) + "/formatted_transactions.csv"

# Read the raw data from the input CSV file
raw = dt.fread(inPath, columns=dt.str32)

# Initialize dictionaries to map values to unique IDs
currency = dict()
paymentFormat = dict()
bankAcc = dict()
account = dict()


def get_dict_val(name, collection):
    """
    Get a unique ID for a given name, and add it to the collection if not present.

    Args:
    - name (str): The name to look up or add.
    - collection (dict): The collection to store unique IDs.

    Returns:
    - int: The unique ID for the given name.
    """
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val


# Define the header for the output CSV file
header = "EdgeID,from_id,to_id,Timestamp,\
Amount Sent,Sent Currency,Amount Received,Received Currency,\
Payment Format,Is Laundering\n"

# Initialize the variable to track the first timestamp
firstTs = -1

# Open the output CSV file for writing
with open(outPath, 'w') as writer:
    writer.write(header)  # Write the header to the file
    for i in range(raw.nrows):
        # Convert the timestamp from string to UNIX timestamp
        datetime_object = datetime.strptime(
            raw[i, "Timestamp"], '%Y/%m/%d %H:%M')
        ts = datetime_object.timestamp()
        day = datetime_object.day
        month = datetime_object.month
        year = datetime_object.year
        hour = datetime_object.hour
        minute = datetime_object.minute

        # Initialize the start time and the first timestamp
        if firstTs == -1:
            startTime = datetime(year, month, day)
            firstTs = startTime.timestamp() - 10

        # Calculate the time difference from the first timestamp
        ts = ts - firstTs

        # Get unique IDs for currencies and payment formats
        cur1 = get_dict_val(raw[i, "Receiving Currency"], currency)
        cur2 = get_dict_val(raw[i, "Payment Currency"], currency)
        fmt = get_dict_val(raw[i, "Payment Format"], paymentFormat)

        # Create unique account IDs based on bank and account numbers
        fromAccIdStr = raw[i, "From Bank"] + raw[i, 2]
        fromId = get_dict_val(fromAccIdStr, account)
        toAccIdStr = raw[i, "To Bank"] + raw[i, 4]
        toId = get_dict_val(toAccIdStr, account)

        # Convert amounts to floats and get laundering flag
        amountReceivedOrig = float(raw[i, "Amount Received"])
        amountPaidOrig = float(raw[i, "Amount Paid"])
        isl = int(raw[i, "Is Laundering"])

        # Write the formatted line to the output CSV file
        line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n' % \
            (i, fromId, toId, ts, amountPaidOrig,
             cur2, amountReceivedOrig, cur1, fmt, isl)
        writer.write(line)

# Read the formatted CSV file and sort by timestamp
formatted = dt.fread(outPath)
formatted = formatted[:, :, sort(3)]  # Sort by the timestamp column

# Save the sorted data back to the CSV file
formatted.to_csv(outPath)
