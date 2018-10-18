import feedparser 
import re 

def getWords(html):
    # Remove all the HTML tags  
    txt=re.compile(r'<[^>]+>').sub('',html)
    # Split words by all non-alpha characters  
    words=re.compile(r'[^A-Z^a-z]+').split(txt)
    # Convert to lowercase  
    return [word.lower() for word in words if word!='']

# returns title and dictionary of word counts for an RSS feeds 
def getWordCounts(url):
    d=feedparser.parse(url)
    wc={}

    for e in d.entries:
        if 'summary' in e: 
            summary=e.summary
        else:
            summary=e.description
        
        # extract a list of words
        words=getWords(e.title+' '+summary)
        for word in words:
            wc.setdefault(word, 0)
            wc[word]+=1
    return d.feed.title, wc

apcount={}
wordcounts={}
for feedurl in file('feedlist.txt'):
    title, wc=getWordCounts(feedurl)
    wordcounts[title]=wc
    for word, count in wc.items():
        apcount.setdefault(word, 0)
        if count>1:
            apcount[word]+=1

wordlist=[] 
for w,bc in apcount.items():  
    frac=float(bc)/len(feedlist)  
    if frac>0.1 and frac<0.5: 
        wordlist.append(w)
