#install BeautifulSoup  and requests library
#Import Them
from bs4 import BeautifulSoup
import requests
class WebScrapper():
   wikiDoc = requests.get("https://en.wikipedia.org/wiki/Deep_learning");
   parsedDoc = BeautifulSoup(wikiDoc.content, "html.parser")
   def getTitle(self):
     title = self.parsedDoc.title.string
     return title
   def getWikiLinks(self):
       list =[]
       for link in self.parsedDoc.find_all('a'):
          list.append(link.get('href'))
       return list
          #print(link.get('href'))
#print(bsObj.title.string)
webs = WebScrapper()
print(webs.getTitle())
print(webs.getWikiLinks())


