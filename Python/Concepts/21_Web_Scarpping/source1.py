 #! /usr/bin/env python3

''' Simple Web Scrapping script '''

from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

# Part-1: Simply making the GET request part
def simple_get(url):
    ''' 
    Attempts to get the current at `url` by making an HTTP GET request. 
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    '''
    try:
        # header is given to avoid error-403
        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
        with closing(get(url, stream=True, headers=header)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None
    except RequestException as e:
        log_error(f'Error during requests to {url} : {str(e)}')
        return None

def is_good_response(resp):
    ''' Returns True if response seems to be HTML, False otherwise.'''
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None
            and content_type.find('html') > -1)

def log_error(e):
    '''
    It is always a good idea to log errors. This function just prints them,
    but we can make it do anything.
    '''
    print(e)

# Part-2: Function to extract single list of names by implementation similar to that of Part-2
# This function downloads the page and iterates over the <li> elements, picking out each name
# that occurs. Next, we add each name to a Python set, which ensures that you don't end up with
# duplicate names. Finally we convert it into a list and return it.
def get_names():
    '''
    Downloads the page where the list of mathematiciansis found and return the list of strings,
    one per mathematician.
    '''
    url = 'http://www.fabpedigree.com/james/mathmen.htm'
    response = simple_get(url)
    if response is not None:
        html = BeautifulSoup(response, 'html.parser')
        names = set()
        for li in html.select('li'):
            for name in li.text.split('\n'):
                if len(name) > 0:
                    names.add(name.strip())
        return list(names)
    
    # Raise an exception if we failed to get any data from the url
    raise Exception(f'Error retrieving contents at {url}')

# Part-3: Pciking up the pageviews for each of the name in the names list
def pageviews(name):
    '''
    Accepts a `name` of a mathematician and returns the number of pageviews on that
    Mathematician's wikipedia received in the last 60 days, as an `int`.
    '''
    # url_root is a template stirng that is used to build a URL.
    url_root = 'https://xtools.wmflabs.org/articleinfo/en.wikipedia.org/{}'
    response = simple_get(url_root.format(name))
    if response is not None:
        html = BeautifulSoup(response, 'html.parser')
        hit_link = [a for a in html.select('a') if a['href'].find('latest-60') > -1]

        if len(hit_link) > 0:
            link_text = hit_link[0].text.replace(',', '')
            try:
                print(f'{int(link_text)} Pageviews found for {name}')
                return int(link_text)
            except:
                log_error(f'Could not parse {link_text} as an `int`.')
    log_error(f'No Pageviews found for {name}')
    return None

def main():
    ''' Driver Program '''
    print('Getting the list of names...')
    names = get_names()
    print('...done.\n')

    results = []
    print('Getting stats for each name...')
    for name in names:
        try:
            hits = pageviews(name)
            if hits is None:
                hits = -1
            results.append((hits, name))
        except:
            results.append((-1, name))
            log_error(f'Error encountered while processing {name}, skipping')
    print('...done\n')

    results.sort()
    results.reverse()

    if len(results) > 5:
        top_marks = results[:5]
    else:
        top_marks = results
    
    print('\nThe most popular Mathematicians are:\n')
    for (mark, mathematician) in top_marks:
        print(f'{mathematician} with {mark} pageviews')
    
    no_results = len([res for res in results if res[0] == -1])
    print(f'\nBut we did not found the results for {no_results} Mathematicians on the list')

if __name__ == '__main__':
    main()