#  "house_prices/AmesHousing.csv"

def check_first_field(file):

    f = open(file)

    for line in f:
        if line[0] == 10:
            print(line)