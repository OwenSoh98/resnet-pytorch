import csv

row = ['2', 'Marie', 'California']


with open('test_CSV.csv', 'a', newline='') as writeFile: #File darf nicht offen in excel sein, sonst keine permission
    writer = csv.writer(writeFile)
    writer.writerow(row)


writeFile.close()

row = ['3', 'Kappa', 'NY']


with open('test_CSV.csv', 'a', newline='') as writeFile: #File darf nicht offen in excel sein, sonst keine permission
    writer = csv.writer(writeFile)
    writer.writerow(row)

writeFile.close()