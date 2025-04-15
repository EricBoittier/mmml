import requests
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.found = None  # Use instance attribute
    
    def handle_starttag(self, tag, attrs):
        if "button" in tag:
            for attr in attrs:
                if "onclick" in attr[0]:
                    self.found = attr[1]
parser = MyHTMLParser()

def download_paper(doi, resid):
    url = f"https://sci-hub.se/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        with open("paper.pdf", "wb") as f:
            f.write(response.content)
        print("Paper downloaded successfully!")
        print(url)
    else:
        print("Failed to download. Check access rights.")

    parser.feed(str(response.content))
    download_url = parser.found.split("\\")[1][1:]
    download_url = "https:"+download_url
    response = requests.get(download_url)
    if response.status_code == 200:
        _doi = doi.replace("/", ".")
        save_location = f"papers/{resid}-{_doi}.pdf"
        with open(save_location, "wb") as f:
            f.write(response.content)
        print("Saved: ", save_location)
    else:
        print("Failed: ", download_url)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python download_paper.py <doi> <resid>")
        sys.exit(1)

    doi = sys.argv[1]
    resid = sys.argv[2]
    download_paper(doi, resid)

