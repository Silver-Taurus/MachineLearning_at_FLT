#! /usr/bin/env python3

''' Use of requests module '''

import requests
import os

def halt_reset():
    ''' Just a utility function '''
    while(True):
        response = input('\nPress Enter to continue: ') or '\n'
        if(response):
            break
    os.system('clear')


# Few attributes
R = requests.get('https://api.github.com/')
print(R.status_code)
print(R.headers['content-type'])
print(R.encoding)
print(R.text)
print(R.json())


halt_reset()


# Making a request

# Let's try getting a webpage, for ex: getting github's public timeline
R = requests.get('https://api.github.com/events')

# After requesting we had got a response
print(R.status_code)

# request's simple API means that all forms of HTTP request are obvious. For example:
# this is how you make an http post request
# R = requests.post('http://httpbin.org/post', data={'key': 'value'})
# Similarly
# R = requests.put('http://httpbin.org/put', data={'key': 'value'})
# R = requests.delete('http://httpbin.org/delete')
# R = requests.head('http://httpbin.org/head')
# R = requests.options('http://httpbin.org/options')


# Passing arguments in URLs
PAYLOAD = {'key1': 'value1', 'key2': 'value2'}
R = requests.get('https://httpbin.org/get', params=PAYLOAD)
print(R.url)
# Note that any dictionary key whose value is None will not be added to the URL's query string

# You can also add a list of items as a value.
PAYLOAD = {'key1': 'value1', 'key2': ['value2', 'value3']}
R = requests.get('https://httpbin.org/get', params=PAYLOAD)
print(R.url)


# Response Content
R = requests.get('https://api.github.com/events')
print(R.text)
print()


# Using online bin
# using get and post commands


# To get the info about any user
# r = requests.get('https://api.github.com/user/user_name)

# To get the info about the repos of any user
# r = requests.get('https://api.github.com/user/user_name/repos')


# To create a git repo
# g = requests.post('https://api.github.com/user/repos', data={'name': 'repo_name', 'description': 'details'},
# auth=('user_name', 'password'))



# Simple Program

from requests import get
from requests.exceptions import RequestException
from contextlib import closing

def simple_get(url):
    ''' 
    Attempts to get the current at `url` by making an HTTP GET request. 
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    '''
    try:
        with closing(get(url, stream=True)) as resp:
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

def main():
    ''' Driver Program '''
    raw_html = simple_get('http://www.fabpedigree.com/james/mathmen.htm')
    print(len(raw_html))
    print()
    no_html = simple_get('https://realpython.com.blog/nope-not-gonna-find-it')
    print(no_html is None)

if __name__ == '__main__':
    main()