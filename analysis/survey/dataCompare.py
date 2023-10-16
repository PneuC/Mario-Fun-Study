import csv


if __name__ == '__main__':
    with open('questionare.csv', newline='') as csvfile:
        questionRow = csv.reader(csvfile, delimiter=' ', quotechar='|')
        id_dict_q = {}
        id_dict_a = {}

        for question in questionRow:
            questionarray = question[0].split(',')
            ans = 'Null'
            if questionarray[3] == 'A':
                ans = 'r'
            elif questionarray[3] == 'B':
                ans = 'k'
            elif questionarray[3] == 'C':
                ans = 'c'
            id_dict_q[questionarray[0]] = ans

    with open('annotation2.csv', newline='') as csvfile:
        annotationRow = csv.reader(csvfile, delimiter=' ', quotechar='|')

        for annotation in annotationRow:
            annotationarray = annotation[0].split(',')
            id_dict_a[annotationarray[0]] = [annotationarray[1], annotationarray[2], annotationarray[3]]

    with open('rank.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for id in id_dict_q.keys():
            ans = "Null"
            if id_dict_q[id] == 'c':
                ans = 'Collector '
            elif id_dict_q[id] == 'k':
                ans = 'Killer    '
            elif id_dict_q[id] == 'r':
                ans = 'Runner    '
            if id in id_dict_a.keys():
                rank = id_dict_a[id].index(id_dict_q[id])
                spamwriter.writerow([id, ans, rank+1, id_dict_a[id]])
            else:
                spamwriter.writerow([id, ans, 0, 'Null'])
