import sys, csv
import urllib.request, urllib.parse
from bs4 import BeautifulSoup

for i in range(1,2):
	urlo = "https://data.j-league.or.jp/SFRT01/?search=search&yearId=2017&yearIdLabel=2017年&competitionId=428&competitionIdLabel=明治安田生命Ｊ１リーグ&competitionSectionId=3&competitionSectionIdLabel=第３節&homeAwayFlg=%s"%(i)
	url = urllib.parse.quote(urlo, safe=":/?=&")

	html = urllib.request.urlopen(url)
	soup = BeautifulSoup(html,"html.parser")

	table = soup.findAll("table",{"class":"standings-table00"})[0]
	rows = table.findAll("tr")

	csvFile = open("ranking.csv", 'wt', newline='', encoding='utf-8')
	writer = csv.writer(csvFile)

	try:
		for row in rows:
			csvRow = []
			list = ["02","03","04","05","08","09","12","13"]
			for i in list:
				for cell in row.findAll("td",{"class":"wd%s"%(i)}):
					csvRow.append(cell.get_text())
					print(i)
					print(cell.get_text())
				writer.writerow(csvRow)
	finally:
		csvFile.close()
